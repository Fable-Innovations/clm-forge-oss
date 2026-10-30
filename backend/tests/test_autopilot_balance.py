"""Balance target maths: shares bend toward the real prior, rows are only added, the round cap carries."""
import math

import pytest

from forge.autopilot import balance


def test_target_share_formula_and_prior_choice():
    labels = ["a", "b", "c"]
    shares, prior, src = balance.target_shares(labels, {"a": 80, "b": 15, "c": 5}, {"a": 1, "b": 1, "c": 1})
    assert src == "real"
    assert prior == pytest.approx({"a": 81 / 103, "b": 16 / 103, "c": 6 / 103})
    for l in labels:
        assert shares[l] == pytest.approx(max(0.5 / 3, 0.5 / 3 + 0.5 * prior[l]))
    assert sum(shares.values()) == pytest.approx(1.0)
    # fewer than 30 real labelled rows: the synthetic train prior
    _s, prior2, src2 = balance.target_shares(labels, {"a": 10, "b": 5}, {"a": 100, "b": 100, "c": 100})
    assert src2 == "synthetic" and prior2["a"] == pytest.approx(1 / 3)


def test_rows_only_added_and_min_count():
    labels = ["a", "b"]
    p = balance.plan(labels, have={"a": 900, "b": 100}, real={"a": 300, "b": 30}, synthetic={"a": 600, "b": 70},
                     min_count=150, gen_cap=10_000)
    n_star = max(900 / p.shares["a"], 100 / p.shares["b"])
    assert p.required_size == math.ceil(n_star)
    assert p.deficits["a"] == 0                                    # never negative: nothing is dropped
    assert p.deficits["b"] == math.ceil(p.shares["b"] * n_star) - 100
    assert p.carry == {"a": 0, "b": 0} and p.allocation == p.deficits
    small = balance.plan(labels, have={"a": 100, "b": 100}, real={}, synthetic={"a": 100, "b": 100})
    assert small.deficits == {"a": 50, "b": 50}                  # balanced, but under min_count


def test_round_cap_allocates_proportionally_and_carries_the_rest():
    alloc, carry = balance.allocate({"a": 900, "b": 300, "c": 0}, 600)
    assert sum(alloc.values()) == 600 and alloc == {"a": 450, "b": 150, "c": 0}
    assert carry == {"a": 450, "b": 150, "c": 0}
    alloc, carry = balance.allocate({"a": 1, "b": 1, "c": 1}, 2)
    assert sum(alloc.values()) == 2 and sum(carry.values()) == 1
