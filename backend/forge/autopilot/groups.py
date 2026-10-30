"""Call-group readiness, built from the adapter's ``groups`` output.

A call group (classify, same-intent, done-match, ...) is a set of heads one call site uses together. It is
ready to switch to the candidate encoder when every head in it has a promoted head on that encoder that
passes its gate and is within ``tol`` of its 4B score. When every group shares one encoder setting in
the app that ships the heads, an encoder switch is a shipping decision for all groups at once: the report says so
whenever the adapter names the shared setting.
"""
from __future__ import annotations

from forge.autopilot.targets import Policy


def readiness(out: dict, policy: Policy, encoder: str) -> dict:
    """``out`` = the ``groups`` verb's data: ``{encoder, groups: {name: [GroupHead]}, call_cooccurrence,
    cross_group_calls, shared_encoder: {setting, note}}``. Retired heads do not block a group."""
    groups = []
    for name, heads in (out.get("groups") or {}).items():
        rows, reasons = [], []
        for info in heads:
            h = info.get("head")
            if info.get("retired"):
                rows.append({"head": h, "ready": True, "retired": True, "why": "retired: not required"})
                continue
            tol = policy.tol_for(h)
            score, base = info.get("macro_f1"), info.get("reference_macro_f1")
            within = score is not None and base is not None and score >= base - tol - 1e-12
            on_enc = info.get("encoder") == encoder
            ok = bool(info.get("promoted")) and on_enc and bool(info.get("gate_passed")) and within
            why = [] if ok else [w for w, bad in (
                ("not promoted", not info.get("promoted")),
                (f"encoder {info.get('encoder')!r} != {encoder!r}", not on_enc),
                ("gate failed", not info.get("gate_passed")),
                (f"test macro-F1 {score} vs 4B {base} beyond tolerance {tol}", not within)) if bad]
            rows.append({"head": h, "promoted": bool(info.get("promoted")), "encoder": info.get("encoder"),
                         "gate_passed": bool(info.get("gate_passed")), "test_macro_f1": score,
                         "baseline_4b_macro_f1": base, "within_tol": within, "ready": ok,
                         "provenance": info.get("provenance"), "why": "; ".join(why)})
            reasons += [f"{h}: {w}" for w in why]
        groups.append({"name": name, "heads": rows, "ready": bool(rows) and all(r["ready"] for r in rows),
                       "reasons": reasons})
    shared = (out.get("shared_encoder") or {}).get("setting")
    return {"encoder": encoder, "groups": groups, "co_occurrence": out.get("call_cooccurrence") or {},
            "cross_group_calls": out.get("cross_group_calls") or {}, "shared_encoder_setting": shared,
            "notice": (f"every group shares {shared}: switching the encoder is a shipping decision for all groups"
                       if shared else None)}
