"""Direct visit-alignment measurements for toy looped residual networks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Protocol, Sequence

try:
    import torch
    from torch import Tensor, nn
    from torch.func import functional_call, jvp, vjp
except ImportError as exc:  # pragma: no cover - optional neural dependency
    raise ImportError("lsa.kappa_probe requires the optional 'neural' extra") from exc


@dataclass(frozen=True)
class KappaEstimate:
    rounds: int
    visible_rounds: int
    kappa: float
    u_norms: tuple[float, ...]
    g_norms: tuple[float, ...]
    u_sum_norm: float
    g_sum_norm: float
    beta_alpha_ratio_sq: float
    power_iterations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PowerLawFit:
    gamma: float
    intercept: float
    r_squared: float
    points: int

    def predict(self, rounds: float) -> float:
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        return math.exp(self.intercept) * rounds**self.gamma

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ToyResidualBlock(nn.Module):
    """Two-linear-layer residual block with explicit alpha/beta scaling."""

    def __init__(self, hidden_size: int, *, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        if hidden_size <= 0 or alpha <= 0 or beta < 0:
            raise ValueError("invalid residual block dimensions or scaling")
        self.hidden_size = int(hidden_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, state: Tensor) -> Tensor:
        branch = self.out_proj(torch.tanh(self.in_proj(state)))
        return self.alpha * state + self.beta * branch


class ToyLoop(nn.Module):
    """A finite schedule of tied or visit-untied residual blocks."""

    def __init__(
        self,
        hidden_size: int,
        rounds: int,
        *,
        tied: bool,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        self.hidden_size = int(hidden_size)
        self.rounds = int(rounds)
        self.tied = bool(tied)
        count = 1 if tied else rounds
        self.blocks = nn.ModuleList(
            ToyResidualBlock(hidden_size, alpha=alpha, beta=beta) for _ in range(count)
        )

    @property
    def alpha(self) -> float:
        return self.blocks[0].alpha

    @property
    def beta(self) -> float:
        return self.blocks[0].beta

    def block_at(self, visit: int) -> ToyResidualBlock:
        if visit not in range(self.rounds):
            raise IndexError(visit)
        return self.blocks[0 if self.tied else visit]

    def forward(self, state: Tensor) -> Tensor:
        for visit in range(self.rounds):
            state = self.block_at(visit)(state)
        return state

    def states(self, state: Tensor) -> tuple[Tensor, ...]:
        result = [state]
        for visit in range(self.rounds):
            state = self.block_at(visit)(state)
            result.append(state)
        return tuple(result)

    def suffix(self, state: Tensor, after_visit: int) -> Tensor:
        for visit in range(after_visit + 1, self.rounds):
            state = self.block_at(visit)(state)
        return state


class ResidualLoop(Protocol):
    """Structural interface required by the visit-alignment estimator."""

    rounds: int
    alpha: float
    beta: float

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def block_at(self, visit: int) -> nn.Module: ...

    def states(self, state: Tensor) -> tuple[Tensor, ...]: ...

    def suffix(self, state: Tensor, after_visit: int) -> Tensor: ...


def _flatten(tensors: Iterable[Tensor]) -> Tensor:
    values = [tensor.reshape(-1) for tensor in tensors]
    if not values:
        raise ValueError("cannot flatten an empty tensor collection")
    return torch.cat(values)


def alignment_coefficient(
    sensitivities: Sequence[Tensor],
    gradients: Sequence[Tensor],
) -> float:
    """Compute directional visit alignment in the required interval ``[0, R]``.

    Residual scale is deliberately excluded here and applied once by the
    stability functional. Including it both in kappa and in B double-counts
    scaling and makes even a one-visit coefficient exceed one.
    """

    if len(sensitivities) != len(gradients) or not sensitivities:
        raise ValueError("U and G must be non-empty and have the same visit count")
    u_norms = torch.stack([value.norm() for value in sensitivities])
    g_norms = torch.stack([value.norm() for value in gradients])
    denominator = (
        len(sensitivities)
        * float(u_norms.max().item())
        * float(g_norms.max().item())
    )
    if denominator == 0:
        raise ValueError("kappa is undefined when a visit sensitivity or gradient scale is zero")
    numerator = float(torch.stack(tuple(sensitivities)).sum(dim=0).norm().item())
    numerator *= float(torch.stack(tuple(gradients)).sum(dim=0).norm().item())
    kappa = numerator / denominator
    rounds = len(sensitivities)
    if kappa < -1e-6 or kappa > rounds + 1e-5:
        raise RuntimeError(f"kappa range invariant failed: {kappa} not in [0,{rounds}]")
    return max(0.0, min(float(rounds), kappa))


def _suffix_sensitivity(
    model: ResidualLoop,
    state_after_visit: Tensor,
    visit: int,
    *,
    power_iterations: int,
    generator: torch.Generator,
) -> Tensor:
    """Return a deterministic top-direction JVP through the remaining suffix."""

    if power_iterations <= 0:
        raise ValueError("power_iterations must be positive")
    base = state_after_visit.detach()
    direction = torch.randn(
        base.shape,
        dtype=base.dtype,
        device=base.device,
        generator=generator,
    )
    direction = direction / direction.norm().clamp_min(torch.finfo(base.dtype).eps)

    def suffix_fn(value: Tensor) -> Tensor:
        return model.suffix(value, visit)

    for _ in range(power_iterations):
        _, tangent = jvp(suffix_fn, (base,), (direction,))
        tangent_norm = tangent.norm().clamp_min(torch.finfo(base.dtype).eps)
        output_direction = tangent / tangent_norm
        _, pullback = vjp(suffix_fn, base)
        direction = pullback(output_direction)[0]
        direction = direction / direction.norm().clamp_min(torch.finfo(base.dtype).eps)
    _, tangent = jvp(suffix_fn, (base,), (direction,))
    return tangent.detach().reshape(-1)


def _frozen_call(block: nn.Module, state: Tensor) -> Tensor:
    parameters = {name: value.detach() for name, value in block.named_parameters()}
    buffers = {name: value.detach() for name, value in block.named_buffers()}
    return functional_call(block, (parameters, buffers), (state,))


def _visit_gradient(
    model: ResidualLoop,
    inputs: Tensor,
    targets: Tensor,
    target_visit: int,
) -> Tensor:
    """Recompute with exactly one module application live to its parameters."""

    model.zero_grad(set_to_none=True)
    state = inputs.detach()
    live_block = model.block_at(target_visit)
    for visit in range(model.rounds):
        block = model.block_at(visit)
        state = block(state) if visit == target_visit else _frozen_call(block, state)
    loss = torch.nn.functional.mse_loss(state, targets.detach())
    parameters = tuple(live_block.parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    return _flatten(gradient.detach() for gradient in gradients)


def estimate_kappa(
    model: ResidualLoop,
    inputs: Tensor,
    targets: Tensor,
    *,
    visible_visits: Iterable[int] | None = None,
    power_iterations: int = 5,
    seed: int = 0,
) -> KappaEstimate:
    """Measure visit alignment with O(R_g) JVP/VJP and backward recomputations."""

    positions = tuple(range(model.rounds) if visible_visits is None else visible_visits)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("visible visits must be a non-empty set of unique positions")
    if any(position not in range(model.rounds) for position in positions):
        raise ValueError("visible visit is outside the loop")
    ratio_sq = (model.beta / model.alpha) ** 2
    if ratio_sq == 0:
        raise ValueError("kappa is undefined for beta=0")

    with torch.no_grad():
        states = model.states(inputs.detach())
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    sensitivities = tuple(
        _suffix_sensitivity(
            model,
            states[position + 1],
            position,
            power_iterations=power_iterations,
            generator=generator,
        )
        for position in positions
    )
    gradients = tuple(
        _visit_gradient(model, inputs, targets, position) for position in positions
    )
    kappa = alignment_coefficient(sensitivities, gradients)
    u_norms = tuple(float(value.norm().item()) for value in sensitivities)
    g_norms = tuple(float(value.norm().item()) for value in gradients)
    return KappaEstimate(
        rounds=model.rounds,
        visible_rounds=len(positions),
        kappa=kappa,
        u_norms=u_norms,
        g_norms=g_norms,
        u_sum_norm=float(torch.stack(sensitivities).sum(dim=0).norm().item()),
        g_sum_norm=float(torch.stack(gradients).sum(dim=0).norm().item()),
        beta_alpha_ratio_sq=ratio_sq,
        power_iterations=power_iterations,
    )


def fit_power_law(rounds: Sequence[float], kappas: Sequence[float]) -> PowerLawFit:
    """Fit log(kappa)=intercept+gamma*log(R) with an explicit R-squared."""

    if len(rounds) != len(kappas) or len(rounds) < 2:
        raise ValueError("power-law fitting requires paired values at two or more rounds")
    if any(value <= 0 or not math.isfinite(value) for value in (*rounds, *kappas)):
        raise ValueError("power-law inputs must be finite and positive")
    x = [math.log(value) for value in rounds]
    y = [math.log(value) for value in kappas]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0:
        raise ValueError("round values must not all be equal")
    gamma = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator
    intercept = y_mean - gamma * x_mean
    predicted = [intercept + gamma * value for value in x]
    residual = sum((actual - estimate) ** 2 for actual, estimate in zip(y, predicted))
    total = sum((actual - y_mean) ** 2 for actual in y)
    r_squared = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
    return PowerLawFit(gamma, intercept, r_squared, len(rounds))


def geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def spread_ratio(values: Sequence[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("spread ratio requires finite positive values")
    return max(values) / min(values)
