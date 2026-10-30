"""Balance targets for one head (SPEC "Autopilot" > "Balance target"). Pure arithmetic, no I/O.

- Prior ``p_l``: the real labelled prior per label group, Laplace-smoothed; with fewer than
  ``real_prior_min`` (30) real labelled rows, the synthetic train prior instead.
- Target share ``t_l = max(0.5/K, 0.5*(1/K) + 0.5*p_l)``: real priors bend the target, never erase it.
- Rows are only ever added (real rows are never dropped): required size ``N* = max_l(n_l / t_l)`` and the
  deficit ``d_l = max(ceil(t_l*N*) - n_l, min_count - n_l, 0)``.
- At most ``gen_cap`` rows per head per round are generated, allocated in proportion to the deficits
  (largest remainder); the rest is carried to the next round or run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BalancePlan:
    shares: dict[str, float]
    prior: dict[str, float]
    prior_source: str                       # "real" | "synthetic"
    required_size: int
    deficits: dict[str, int]                # full deficit per label group
    allocation: dict[str, int]              # rows to generate this round (sum <= gen_cap)
    carry: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.allocation.values())


def laplace(counts: dict[str, int], labels: list[str]) -> dict[str, float]:
    tot = sum(counts.get(l, 0) for l in labels)
    k = len(labels)
    return {l: (counts.get(l, 0) + 1) / (tot + k) for l in labels}


def target_shares(labels: list[str], real: dict[str, int], synthetic: dict[str, int],
                  real_prior_min: int = 30) -> tuple[dict[str, float], dict[str, float], str]:
    if not labels:
        raise ValueError("balance needs at least one label group")
    k = len(labels)
    use_real = sum(real.get(l, 0) for l in labels) >= real_prior_min
    prior = laplace(real if use_real else synthetic, labels)
    shares = {l: max(0.5 / k, 0.5 * (1 / k) + 0.5 * prior[l]) for l in labels}
    return shares, prior, "real" if use_real else "synthetic"


def allocate(deficits: dict[str, int], cap: int) -> tuple[dict[str, int], dict[str, int]]:
    total = sum(deficits.values())
    if total <= cap:
        return dict(deficits), {l: 0 for l in deficits}
    raw = {l: d * cap / total for l, d in deficits.items()}
    alloc = {l: int(math.floor(v)) for l, v in raw.items()}
    left = cap - sum(alloc.values())
    for l in sorted(raw, key=lambda l: (-(raw[l] - alloc[l]), l))[:left]:
        alloc[l] += 1
    return alloc, {l: deficits[l] - alloc[l] for l in deficits}


def plan(labels: list[str], have: dict[str, int], real: dict[str, int], synthetic: dict[str, int], *,
         min_count: int = 150, gen_cap: int = 600, real_prior_min: int = 30) -> BalancePlan:
    """``have``: train rows per label group, every source; ``real``: real labelled train rows;
    ``synthetic``: generated train rows (the fallback prior)."""
    shares, prior, src = target_shares(labels, real, synthetic, real_prior_min)
    n_star = max((have.get(l, 0) / shares[l] for l in labels), default=0.0)
    deficits = {l: max(math.ceil(shares[l] * n_star - 1e-9) - have.get(l, 0), min_count - have.get(l, 0), 0)
                for l in labels}
    alloc, carry = allocate(deficits, gen_cap)
    return BalancePlan(shares=shares, prior=prior, prior_source=src, required_size=int(math.ceil(n_star - 1e-9)),
                       deficits=deficits, allocation=alloc, carry=carry)
