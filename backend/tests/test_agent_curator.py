"""Curator mechanics with a fake LLM (no encoder): relabel drops, balance, stratified split, recurate."""
from collections import Counter

from forge.curator import Curator, label_menu, norm_label
from forge.models import TaskSchema

from test_agent_fakes import TASK, FakeLLM


async def run(previous=None, target=150, guidance=None, llm=None):
    events = []
    c = Curator(llm or FakeLLM(), TASK, events.append, target=target, previous=previous, guidance=guidance,
                use_embeddings=False, concurrency=4)
    rows = await c.run()
    return c, rows, events


async def test_pipeline_relabel_split_and_progress():
    c, rows, events = await run()
    labels = Counter(r.label for r in rows)
    assert set(labels) == {"billing", "technical", "account"}
    assert all(v <= 50 for v in labels.values())             # balanced cap = ceil(150/3)
    assert c.stats["disagreements"] > 0                       # every 10th relabel disagreed and was dropped
    splits = Counter(r.split for r in rows)
    assert splits["train"] > splits["val"] > 0 and splits["test"] > 0
    for lab in labels:                                        # stratified: every label in every split
        assert {r.split for r in rows if r.label == lab} == {"train", "val", "test"}
    assert len({r.id for r in rows}) == len(rows)
    phases = [e["phase"] for e in events if e["type"] == "curation"]
    assert phases[-1] == "done" and "generating" in phases and "labelling" in phases
    assert {r.source for r in rows} <= {"synthetic", "hard_negative"}
    assert any(r.source == "hard_negative" for r in rows)


async def test_recurate_keeps_previous_and_freezes_test():
    _, first, _ = await run()
    test_ids = {r.id for r in first if r.split == "test"}
    c, rows, _ = await run(previous=first, guidance="more billing vs account hard negatives")
    assert {r.id for r in rows if r.split == "test"} == test_ids
    assert len(rows) > len(first)


async def test_generate_fans_out_one_worker_per_label():
    c, _, events = await run()
    workers = [e for e in events if e["type"] == "tool" and e["name"] == "generate"]
    assert sorted(w["input"] for w in workers) == ["account", "billing", "technical"]
    last = [e for e in events if e["type"] == "curation" and e["phase"] == "generating"][-1]
    assert set(last["workers"]) == {"account", "billing", "technical"}
    assert all(w["done"] == w["total"] for w in last["workers"].values())


async def test_generate_workers_share_slots_instead_of_running_in_turn():
    started = []                                                    # labels in the order their batches finish

    def on_event(e):
        if e["type"] == "curation" and e["phase"] == "generating" and e["done"]:
            started.append(next(lab for lab, w in e["workers"].items() if w["done"] > seen.get(lab, 0)))
            seen.update({lab: w["done"] for lab, w in e["workers"].items()})
    seen: dict[str, int] = {}
    c = Curator(FakeLLM(), TASK, on_event, target=600, use_embeddings=False, concurrency=1, gen_concurrency=1)
    await c.generate({"billing": 100, "technical": 100, "account": 100})
    assert set(started[:3]) == {"billing", "technical", "account"}  # one batch from each label before any second


async def test_qa_gate_runs_before_labelling_and_regenerates_failing_label():
    llm = FakeLLM(qa_fail={"billing"})
    c, rows, events = await run(llm=llm)
    first_label = llm.calls.index("LabelBatch")
    assert "QAReview" in llm.calls[:first_label]                 # reviewed before labelling
    assert c.stats["qa"]["billing"]["attempt"] == 2 and c.stats["qa"]["billing"]["passed"]
    assert c.stats["qa"]["technical"]["attempt"] == 1
    assert c.stats["qa_rejected"] > 0
    # the regeneration prompts carry the orchestrator's feedback
    assert any("Fix this: make every text clearly about this queue" in p for p in llm.prompts)
    assert any(r.label == "billing" for r in rows)
    phases = [e["phase"] for e in events if e["type"] == "curation"]
    assert phases.index("reviewing") < phases.index("labelling")


def test_label_helpers():
    assert norm_label(TASK, " Billing ") == "billing"
    assert norm_label(TASK, "nope") is None
    score = TaskSchema(name="s", question_id="s", type="score", instructions="sev?", criteria=["low", "mid", "high"],
                       state_description="x", state_example="y")
    assert score.labels() == ["0", "1", "2"] and norm_label(score, "level 2") == "2"
    noul = TaskSchema(name="n", question_id="n", type="noul", instructions="ok?", state_description="x", state_example="y")
    assert norm_label(noul, "Yes") == "true" and "true" in label_menu(noul)


async def test_generators_use_task_derived_voices():
    llm = FakeLLM()
    c, _, _ = await run(llm=llm)
    assert llm.calls.index("Voices") < llm.calls.index("GenBatch")    # planned once, before generation
    assert llm.calls.count("Voices") == 1
    gen = [p for name, p in zip(llm.calls, llm.prompts) if name == "GenBatch"]
    assert c.personas == ["a terse power user", "a polite first-time user", "an admin"]
    assert all(any(f"Voice: {v}." in p for v in c.personas) for p in gen)


async def test_embedding_dedup_keeps_minimal_pairs(monkeypatch):
    """Near-identical states with different labels are a minimal pair, not duplicates."""
    import numpy as np
    from forge import encoder as encoder_mod
    from forge.models import Example

    class Enc:
        def embed(self, texts, on_progress=None, **_kw):
            v = np.array([[1.0, 0.001 * i] for i in range(len(texts))], dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(encoder_mod, "get_encoder", lambda *a, **k: Enc())
    labs = ["billing", "technical", "billing", "account", "technical", "billing", "account", "account",
            "technical", "billing"]
    rows = [Example(id=f"r{i}", state=f"same long context, differing tail {i}", label=lab, source="synthetic",
                    split="train") for i, lab in enumerate(labs)]
    c = Curator(FakeLLM(), TASK, lambda e: None, target=30, encoder_spec="mlx:test")
    kept = await c.dedup(rows)
    # all ten embeddings are near-identical: only one row per label survives, never one row overall
    assert sorted(r.label for r in kept) == ["account", "billing", "technical"]
