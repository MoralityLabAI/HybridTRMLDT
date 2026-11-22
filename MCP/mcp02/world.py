"""Compositional multi-server MCP worlds with a hidden oracle.

Each case is one synthetic organisation served by several MCP servers with
overlapping capabilities. A task names an entity by display name and region,
asks for a field reached through one to three relation hops, and may combine
two such chains. The world adds ambiguous names, replica and archive copies,
moved documents, unit variants, nested sections, unknown registry metadata,
paginated collections, a mid-episode catalog epoch change and large distractor
catalogs. Expected answers and required documents live only in ``oracle``.

Splits are grouped. Entity and server names come from disjoint word pools per
split, and request templates (intro and constraint phrasing) are assigned to
exactly one split. The concept vocabulary (field synonyms, server role
descriptions) is shared language across splits.
"""
from __future__ import annotations

import random
from typing import Any

from .textutil import digest, stable_int

SPLITS = ("train", "dev", "test")
FAMILIES = ("alias_lookup", "join_chain", "join_arith", "stale_moved",
            "epoch_roll", "conflict_authority", "missing", "paginated")
REGIONS = ("eu-west", "us-east", "ap-south", "sa-east", "eu-north", "us-west")

# Shared language: each concept has interchangeable surface names.
CONCEPTS = {
    "owner_team": ("owner team", "owning team", "maintained by", "steward group"),
    "cost_center": ("cost center", "budget unit", "finance code", "charge code"),
    "vendor": ("primary vendor", "supplier", "contracted vendor"),
    "monthly_budget": ("monthly budget", "monthly allocation", "spend ceiling"),
    "p95_latency": ("p95 latency", "tail latency", "slow request time"),
    "headcount": ("headcount", "staff count", "team size"),
    "contract_value": ("contract value", "annual commitment", "contract total"),
}
ROLE_WORDS = {
    "catalog": ("service catalog", "system inventory", "application registry"),
    "org": ("people directory", "team roster", "org chart"),
    "finance": ("general ledger", "budget book", "finance records"),
    "procurement": ("vendor contracts", "purchasing records", "supplier agreements"),
    "observability": ("telemetry store", "metrics warehouse", "monitoring data"),
    "mirror": ("replica cache", "read mirror", "synced copy"),
    "archive": ("archived snapshots", "historical export", "cold storage"),
}
SCOPE = {"catalog": "inventory", "org": "people", "finance": "ledger", "procurement": "contracts",
         "observability": "telemetry", "mirror": "replica", "archive": "archive"}
KIND_HOME = {"service": "catalog", "team": "org", "cost_center": "finance",
             "budget_section": "finance", "vendor": "procurement", "service_metrics": "observability"}
KIND_WORDS = {
    "service": ("service profile", "application entry", "system record"),
    "team": ("team page", "group profile", "roster entry"),
    "cost_center": ("cost center sheet", "budget unit record", "finance code entry"),
    "budget_section": ("budget section", "allocation detail", "spend schedule"),
    "vendor": ("vendor contract", "supplier agreement", "purchasing record"),
    "service_metrics": ("metrics panel", "telemetry summary", "performance sheet"),
}
KIND_UNIT = {"service": None, "team": "people", "cost_center": None, "budget_section": "kUSD",
             "vendor": "kUSD", "service_metrics": "ms"}
TARGET_KIND = {"monthly_budget": "budget_section", "p95_latency": "service_metrics",
               "headcount": "team", "contract_value": "vendor"}
TARGET_UNIT = {"monthly_budget": "kUSD", "p95_latency": "ms", "headcount": "people", "contract_value": "kUSD"}
RELATION_KIND = {"owner_team": "team", "cost_center": "cost_center", "vendor": "vendor"}
CHAINS = {
    "budget": {"hops": ("owner_team", "cost_center"), "target": "monthly_budget"},
    "latency": {"hops": (), "target": "p95_latency"},
    "headcount": {"hops": ("owner_team",), "target": "headcount"},
    "contract": {"hops": ("vendor",), "target": "contract_value"},
}
VALUE_RANGE = {"monthly_budget": (50, 5000), "p95_latency": (20, 900),
               "headcount": (3, 120), "contract_value": (10, 9000)}
INTROS = ("Find", "Report", "Look up", "Determine", "Retrieve", "Work out")
CLOSINGS = (
    "Use only current authoritative records and cite every record you rely on.",
    "Rely on live records from the owning system and cite each one.",
    "Ignore replicas and archives; cite the records that establish the result.",
    "Answer from the system of record and list the supporting records.",
    "Base the answer on current home-system records with citations.",
    "Only authoritative, unsuperseded records count; cite all of them.",
)
TEMPLATE_SPLIT = {0: "train", 1: "train", 2: "train", 3: "train", 4: "dev", 5: "test"}
_SYLLABLES = ("ka", "lo", "mir", "ves", "tor", "pa", "nel", "dru", "si", "qua", "ber", "os",
              "fen", "ra", "zul", "mo", "ti", "gar", "el", "vin", "sha", "cor", "pe", "lum",
              "dax", "ru", "ny", "bel", "ho", "tri", "jas", "on", "wex", "il", "ma", "sto")


def _word_pool(split: str) -> list[str]:
    words = sorted({a + b for a in _SYLLABLES for b in _SYLLABLES if a != b})
    index = SPLITS.index(split)
    return [w for w in words if stable_int("mcp02-word-split", w) % 3 == index]


class _Names:
    """Unique display names drawn from the split's word pool."""

    def __init__(self, split: str, rng: random.Random):
        self.words = _word_pool(split)
        self.rng = rng
        self.used: set[str] = set()

    def fresh(self, share: str | None = None) -> str:
        for _ in range(10000):
            first, second = self.rng.choice(self.words), self.rng.choice(self.words)
            if share is not None:
                if self.rng.random() < 0.5:
                    first = share.split()[0].lower()
                else:
                    second = share.split()[1].lower()
            name = f"{first.capitalize()} {second.capitalize()}"
            if first != second and name not in self.used:
                self.used.add(name)
                return name
        raise RuntimeError("name pool exhausted")

    def server(self) -> str:
        for _ in range(10000):
            name = self.rng.choice(self.words) + self.rng.choice(self.words)
            if name not in self.used:
                self.used.add(name)
                return name
        raise RuntimeError("name pool exhausted")


class _Builder:
    def __init__(self, split: str, seed: int, index: int, family: str):
        self.group = f"mcp02-world-v1/{split}/{seed}/{index:06d}"
        self.rng = random.Random(stable_int(self.group))
        self.split, self.family = split, family
        self.names = _Names(split, self.rng)
        self.stem = digest(self.group)[:12]
        self.epoch = f"epoch-{self.stem[:6]}-1"
        self.old_epoch = f"epoch-{self.stem[:6]}-0"
        self.docs: list[dict] = []
        self.uris: set[str] = set()
        self.servers = {role: self.names.server() + "-mcp" for role in ROLE_WORDS}
        self.server_role_words = {role: self.rng.choice(words) for role, words in ROLE_WORDS.items()}
        # Each server names a concept with its own synonym.
        self.surface = {role: {concept: self.rng.choice(names) for concept, names in CONCEPTS.items()}
                        for role in ROLE_WORDS}
        self.counter = 0

    def entity_id(self, kind: str) -> str:
        self.counter += 1
        prefix = {"service": "svc", "team": "tm", "cost_center": "cc", "vendor": "vd"}[kind]
        return f"{prefix}-{digest([self.group, 'entity', self.counter])[:6]}"

    def add(self, *, kind: str, role: str, collection: str, entity_id: str, name: str,
            fields: dict, links: dict, status: str = "active", superseded_by: str | None = None,
            unit: str | None = "default", region: str | None = None, epoch: str | None = None,
            late: bool = False) -> dict:
        while True:
            self.counter += 1
            slug = digest([self.group, "doc", self.counter])[:10]
            if late and slug[0] not in "ef":
                continue
            uri = f"mcp://{self.servers[role]}/{collection}/{slug}"
            if uri not in self.uris:
                break
        self.uris.add(uri)
        doc = {
            "uri": uri, "server": self.servers[role], "role": role, "collection": collection,
            "kind": kind, "entity_id": entity_id, "name": name, "region": region,
            "fields": dict(fields), "links": dict(links), "status": status,
            "superseded_by": superseded_by, "scope": SCOPE[role],
            "unit": KIND_UNIT[kind] if unit == "default" else unit,
            "epoch": epoch or (self.old_epoch if role == "archive" else self.epoch),
            "revision": f"r{1 + self.rng.randrange(9)}",
            "kind_word": self.rng.choice(KIND_WORDS[kind]),
        }
        # Some registry entries omit status, scope or unit; reading reveals them.
        registry = {"status": status, "scope": doc["scope"], "unit": doc["unit"], "epoch": doc["epoch"]}
        if self.rng.random() < 0.12:
            registry[self.rng.choice(("status", "scope", "unit"))] = None
        doc["registry"] = registry
        self.docs.append(doc)
        return doc

    def value(self, field: str) -> int:
        low, high = VALUE_RANGE[field]
        return self.rng.randint(low, high)

    # -- entity groups -----------------------------------------------------------------
    def shard(self, base: str, key: str, shards: int) -> str:
        return f"{base}-{1 + stable_int(self.group, key) % shards}"

    def service_group(self, name: str, region: str, *, shards: int, late_team: bool = False) -> dict:
        """One service with its team, cost center, budget section, vendor and metrics."""
        g = {"service": self.entity_id("service"), "team": self.entity_id("team"),
             "cost_center": self.entity_id("cost_center"), "vendor": self.entity_id("vendor"),
             "name": name, "region": region}
        g["team_name"], g["cc_name"], g["vendor_name"] = (self.names.fresh() for _ in range(3))
        g["values"] = {field: self.value(field) for field in VALUE_RANGE}
        g["service_doc"] = self.add(
            kind="service", role="catalog", collection=self.shard("services", g["service"], shards),
            entity_id=g["service"], name=name, region=region, fields={},
            links={"owner_team": {"id": g["team"], "name": g["team_name"]},
                   "vendor": {"id": g["vendor"], "name": g["vendor_name"]}})
        g["team_doc"] = self.add(
            kind="team", role="org", collection=self.shard("teams", g["team"], shards),
            entity_id=g["team"], name=g["team_name"], fields={"headcount": g["values"]["headcount"]},
            links={"cost_center": {"id": g["cost_center"], "name": g["cc_name"]}}, late=late_team)
        section = self.add(
            kind="budget_section", role="finance", collection="budget-sections",
            entity_id=g["cost_center"], name=g["cc_name"],
            fields={"monthly_budget": g["values"]["monthly_budget"]}, links={})
        g["section_doc"] = section
        g["cc_doc"] = self.add(
            kind="cost_center", role="finance", collection="cost-centers", entity_id=g["cost_center"],
            name=g["cc_name"], fields={}, links={"monthly_budget": {"uri": section["uri"]}})
        g["vendor_doc"] = self.add(
            kind="vendor", role="procurement", collection="vendors", entity_id=g["vendor"],
            name=g["vendor_name"], fields={"contract_value": g["values"]["contract_value"]}, links={})
        g["metrics_doc"] = self.add(
            kind="service_metrics", role="observability", collection="service-metrics",
            entity_id=g["service"], name=name, fields={"p95_latency": g["values"]["p95_latency"]}, links={})
        return g

    def replica(self, doc: dict, value_field: str, *, bait: bool = False) -> dict:
        fields = {key: (None if val is None else val + self.rng.choice((-1, 1)) * self.rng.randint(1, 40))
                  for key, val in doc["fields"].items()}
        if not fields:
            fields = {value_field: self.value(value_field)}
        collection = "replica-" + ("finance" if doc["role"] in ("finance", "procurement") else "telemetry"
                                   if doc["role"] == "observability" else "people")
        copy = self.add(kind=doc["kind"], role="mirror", collection=collection, entity_id=doc["entity_id"],
                        name=doc["name"], fields=fields, links={}, unit=doc["unit"], region=doc["region"])
        if bait:
            copy["kind_word"] = "current authoritative " + doc["kind_word"]
        return copy

    def archive(self, doc: dict) -> dict:
        fields = {key: (None if val is None else val + self.rng.randint(1, 60)) for key, val in doc["fields"].items()}
        return self.add(kind=doc["kind"], role="archive", collection=f"snapshot-{self.old_epoch[-8:]}",
                        entity_id=doc["entity_id"], name=doc["name"], fields=fields, links=dict(doc["links"]),
                        status="archived", unit=doc["unit"], region=doc["region"])


def _chain_text(kind: str, g: dict, pick) -> str:
    who = f"the service {g['name']} in {g['region']}"
    if kind == "budget":
        return (f"the {pick('monthly_budget')} of the {pick('cost_center')} of the "
                f"{pick('owner_team')} of {who}")
    if kind == "latency":
        return f"the {pick('p95_latency')} of {who}"
    if kind == "headcount":
        return f"the {pick('headcount')} of the {pick('owner_team')} of {who}"
    return f"the {pick('contract_value')} with the {pick('vendor')} of {who}"


def _required(kind: str, g: dict) -> tuple[list[str], int | None]:
    if kind == "budget":
        docs = [g["service_doc"], g["team_doc"], g["cc_doc"], g["section_doc"]]
        value = g["section_doc"]["fields"]["monthly_budget"]
    elif kind == "latency":
        docs = [g["service_doc"], g["metrics_doc"]]
        value = g["metrics_doc"]["fields"]["p95_latency"]
    elif kind == "headcount":
        docs = [g["service_doc"], g["team_doc"]]
        value = g["team_doc"]["fields"]["headcount"]
    else:
        docs = [g["service_doc"], g["vendor_doc"]]
        value = g["vendor_doc"]["fields"]["contract_value"]
    return [d["uri"] for d in docs], value


def make_world(split: str = "test", seed: int = 7, index: int = 0, catalog_size: int = 256) -> dict:
    """Build one reproducible case. Only ``task`` is public."""
    if split not in SPLITS:
        raise ValueError("split must be train, dev or test")
    if type(catalog_size) is not int or catalog_size < 64:
        raise ValueError("catalog_size must be an integer of at least 64")
    family = FAMILIES[index % len(FAMILIES)]
    b = _Builder(split, seed, index, family)
    rng = b.rng
    shards = max(1, catalog_size // 96)
    chain_kinds = {"alias_lookup": ["latency"], "join_chain": ["budget"],
                   "join_arith": [rng.choice(("headcount", "latency"))] * 2,
                   "stale_moved": ["headcount"], "epoch_roll": ["contract"],
                   "conflict_authority": [rng.choice(("budget", "latency"))],
                   "missing": [rng.choice(("latency", "headcount"))], "paginated": ["headcount"]}[family]
    groups = []
    for chain in chain_kinds:
        name, region = b.names.fresh(), rng.choice(REGIONS)
        g = b.service_group(name, region, shards=shards, late_team=family == "paginated")
        groups.append(g)
        # Ambiguous naming: a twin service shares the display name in another region.
        if family in ("alias_lookup", "join_chain") or rng.random() < 0.35:
            twin_region = rng.choice([r for r in REGIONS if r != region])
            b.service_group(name, twin_region, shards=shards)
        # Confusable neighbours share one word of the name.
        for _ in range(2):
            b.service_group(b.names.fresh(share=name), rng.choice(REGIONS), shards=shards)
    hidden: dict[str, Any] = {}
    for chain, g in zip(chain_kinds, groups):
        target_doc = {"budget": g["section_doc"], "latency": g["metrics_doc"],
                      "headcount": g["team_doc"], "contract": g["vendor_doc"]}[chain]
        field = CHAINS[chain]["target"]
        b.replica(target_doc, field, bait=family == "conflict_authority")
        if family == "conflict_authority":
            b.archive(target_doc)
            b.replica(target_doc, field)
        b.archive(g["service_doc"])
        if chain == "budget":
            regional = b.add(kind="budget_section", role="finance", collection="regional-ledger",
                             entity_id=g["cost_center"], name=g["cc_name"],
                             fields={"monthly_budget": g["values"]["monthly_budget"] * 9 // 10}, links={},
                             unit="kEUR")
            regional["kind_word"] = "regional " + regional["kind_word"]
    if family == "stale_moved":
        g = groups[0]
        old = g["team_doc"]
        moved = b.add(kind="team", role="org", collection="teams-relocated", entity_id=old["entity_id"],
                      name=old["name"], fields=dict(old["fields"]), links=dict(old["links"]))
        old.update(status="superseded", superseded_by=moved["uri"])
        old["registry"]["status"] = None if rng.random() < 0.5 else "superseded"
        g["team_doc"] = moved
    if family == "missing":
        g = groups[0]
        doc = g["metrics_doc"] if chain_kinds[0] == "latency" else g["team_doc"]
        field = "p95_latency" if chain_kinds[0] == "latency" else "headcount"
        doc["fields"][field] = None
    if family == "epoch_roll":
        hidden["epoch_roll"] = {"after_calls": rng.randint(3, 7), "uri": groups[0]["vendor_doc"]["uri"],
                                "new_value": b.value("contract_value"),
                                "new_epoch": f"epoch-{b.stem[:6]}-2"}
    # Distractor catalog: unrelated groups until the requested size is met exactly.
    while len(b.docs) < catalog_size:
        g = b.service_group(b.names.fresh(), rng.choice(REGIONS), shards=shards)
        if rng.random() < 0.5:
            b.replica(g["metrics_doc"], "p95_latency")
        if rng.random() < 0.3:
            b.archive(g["team_doc"])
    core = {uri for chain, g in zip(chain_kinds, groups) for uri in _required(chain, g)[0]}
    core |= {d["superseded_by"] for d in b.docs if d["superseded_by"]}
    core |= {d["uri"] for d in b.docs if d["superseded_by"] in core}
    # Trim surplus distractors from the end, never a required or linked document.
    keep_links = {link["uri"] for d in b.docs if d["uri"] in core for link in d["links"].values() if "uri" in link}
    while len(b.docs) > catalog_size:
        for i in range(len(b.docs) - 1, -1, -1):
            if b.docs[i]["uri"] not in core and b.docs[i]["uri"] not in keep_links:
                del b.docs[i]
                break
        else:
            raise ValueError("catalog_size is smaller than the task's required core")
    if family == "paginated":
        hidden["sketch_exclude"] = [groups[0]["team_doc"]["uri"]]
    required, values = [], []
    for chain, g in zip(chain_kinds, groups):
        uris, value = _required(chain, g)
        required.append(uris)
        values.append(value)
    combine = None
    if family == "join_arith":
        combine = rng.choice(("sum", "difference"))
    answer = None if any(v is None for v in values) else (
        values[0] if combine is None else values[0] + values[1] if combine == "sum" else values[0] - values[1])
    template = rng.choice([i for i, s in TEMPLATE_SPLIT.items() if s == split])
    pick = lambda concept: rng.choice(CONCEPTS[concept])  # noqa: E731
    parts = [_chain_text(chain, g, pick) for chain, g in zip(chain_kinds, groups)]
    if combine == "sum":
        body = f"the sum of {parts[0]} and {parts[1]}"
    elif combine == "difference":
        body = f"{parts[0]} minus {parts[1]}"
    else:
        body = parts[0]
    request = (f"{INTROS[template]} {body}. {CLOSINGS[template]} "
               "If a required value is not reported by the system of record, abstain and cite the records read.")
    plan = {"chains": [{"anchor": {"kind": "service", "name": g["name"], "region": g["region"]},
                        "hops": list(CHAINS[chain]["hops"]), "target": CHAINS[chain]["target"],
                        "unit": TARGET_UNIT[CHAINS[chain]["target"]]}
                       for chain, g in zip(chain_kinds, groups)],
            "combine": combine}
    task = {"id": b.stem, "request": request, "plan": plan,
            "constraints": {"status": "active", "authority": "home system of each record kind",
                            "home_scopes": {kind: SCOPE[role] for kind, role in KIND_HOME.items()}},
            "catalog_epoch": b.epoch, "template": template}
    docs = sorted(b.docs, key=lambda d: d["uri"])
    return {
        "id": b.stem, "family": family, "split": split, "group_id": b.group,
        "template_group": f"template-{template}", "name_pool": f"words-{split}",
        "provenance": {"kind": "synthetic", "generator": "mcp02-world-v1", "seed": seed},
        "task": task, "docs": docs,
        "servers": {role: {"name": name, "description": b.server_role_words[role],
                           "surface": b.surface[role]} for role, name in b.servers.items()},
        "hidden": hidden,
        "oracle": {"answer": answer, "required_uris": required, "abstain": answer is None,
                   "chain_values": values},
    }


def make_cases(split: str = "test", seed: int = 7, n: int = 16, catalog_size: int = 256,
               families: tuple[str, ...] | None = None) -> list[dict]:
    cases = []
    index = 0
    while len(cases) < n:
        family = FAMILIES[index % len(FAMILIES)]
        if families is None or family in families:
            cases.append(make_world(split, seed, index, catalog_size))
        index += 1
    return cases
