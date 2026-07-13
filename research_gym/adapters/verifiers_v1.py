from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import os
from pathlib import Path


TARGET_VERIFIERS_VERSION = "0.1.14"
MINIMUM_UV_VERSION = "0.11.1"
DEFAULT_V1_ENV_PATH = Path("environments/hybrid_sequencer_v1")


def _version_tuple(value: str) -> tuple[int, ...]:
    numeric = value.split("+", 1)[0].split(".dev", 1)[0]
    parts = []
    for item in numeric.split("."):
        digits = "".join(character for character in item if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


@dataclass(frozen=True)
class VerifiersV1Integration:
    env_path: Path = DEFAULT_V1_ENV_PATH
    env_id: str = "hybrid-sequencer-v1"
    target_version: str = TARGET_VERIFIERS_VERSION
    minimum_uv_version: str = MINIMUM_UV_VERSION

    @property
    def installed_version(self) -> str | None:
        try:
            return importlib.metadata.version("verifiers")
        except importlib.metadata.PackageNotFoundError:
            return None

    @property
    def installed_version_compatible(self) -> bool:
        version = self.installed_version
        return version is not None and _version_tuple(version) >= _version_tuple(self.target_version)

    @property
    def installed_uv_version(self) -> str | None:
        try:
            return importlib.metadata.version("uv")
        except importlib.metadata.PackageNotFoundError:
            return None

    @property
    def installed_uv_compatible(self) -> bool:
        version = self.installed_uv_version
        return version is not None and _version_tuple(version) >= _version_tuple(self.minimum_uv_version)

    @property
    def package_complete(self) -> bool:
        return all(
            (self.env_path / name).exists()
            for name in ("hybrid_sequencer_v1.py", "pyproject.toml", "README.md", "data/replay_tasks.jsonl")
        )

    @property
    def native_windows_prime_blocked(self) -> bool:
        if os.name != "nt":
            return False
        try:
            distribution = importlib.metadata.distribution("prime-tunnel")
        except importlib.metadata.PackageNotFoundError:
            return False
        tunnel_path = Path(distribution.locate_file("prime_tunnel/tunnel.py"))
        return tunnel_path.exists() and "import fcntl" in tunnel_path.read_text(encoding="utf-8")

    def prime_eval_command(self, config_path: Path = Path("configs/eval/hybrid_sequencer_v1.toml")) -> list[str]:
        return ["prime", "eval", "run", self.env_id, "-c", str(config_path)]

    def exact_release_smoke_command(self) -> list[str]:
        return [
            "uv",
            "run",
            "--isolated",
            "--with",
            f"verifiers=={self.target_version}",
            "python",
            "scripts/smoke_verifiers_v1.py",
        ]

    def to_jsonable(self) -> dict[str, object]:
        value = asdict(self)
        value.update(
            {
                "env_path": str(self.env_path),
                "installed_version": self.installed_version,
                "installed_version_compatible": self.installed_version_compatible,
                "installed_uv_version": self.installed_uv_version,
                "installed_uv_compatible": self.installed_uv_compatible,
                "package_complete": self.package_complete,
                "native_windows_prime_blocked": self.native_windows_prime_blocked,
                "prime_eval_command": self.prime_eval_command(),
                "exact_release_smoke_command": self.exact_release_smoke_command(),
            }
        )
        return value
