"""Importing real rows from exported tables: discovery (incl. gitignored clm_data/), mapping
validation, deterministic rendering, and real rows filling the test split."""
import csv
import re
import random
import subprocess
from collections import Counter

from forge import importer
from forge.curator import Curator
from forge.importer import Table, TableMapping

from test_agent_fakes import TASK, FakeLLM

QUEUES = {"Billing": "I was charged twice for invoice", "Tech": "The app crashes with error", "Acct": "Cannot reset my password for"}


def make_repo(tmp_path, per_queue=40):
    root = tmp_path / "repo"
    (root / "clm_data").mkdir(parents=True)
    (root / ".gitignore").write_text("clm_data/\n")
    (root / "app.py").write_text("def route(ticket): ...\n")
    (root / "other.csv").write_text("a,b\n1,2\n")
    with open(root / "clm_data" / "tickets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "subject", "body", "queue"])
        i = 0
        for q, text in QUEUES.items():
            for _ in range(per_queue):
                i += 1
                w.writerow([i, f"Ticket {i}", f"{text} #{i}", q])
        w.writerow([999, "Ticket 999", "Something odd happened here", "Spam"])  # unmapped label value
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def test_discover_reads_gitignored_clm_data(tmp_path):
    root = make_repo(tmp_path)
    found = importer.discover(root)
    assert found[0] == "clm_data/tickets.csv" and "other.csv" in found


def test_render_and_validate():
    row = {"subject": "Refund", "body": "", "queue": "Billing"}
    assert importer.render("Subject: {subject}\nBody: {body}", row) == "Subject: Refund"
    assert importer.render("Body: {body}", row) is None
    t = Table(rel="t.csv", columns=["subject", "body", "queue"], rows=[row], total_rows=1)
    leak = TableMapping(relevant=True, reason="", template="{subject} {queue}", label_column="queue", label_map={"Billing": "billing"})
    assert importer.validate(leak, t, TASK)[0] is None
    bad = TableMapping(relevant=True, reason="", template="{nope}")
    assert "unknown columns" in importer.validate(bad, t, TASK)[1]
    ok = TableMapping(relevant=True, reason="", template="{subject}\n{body}", label_column="queue",
                      label_map={"Billing": "Billing", "Other": "not-a-label"})
    m, _ = importer.validate(ok, t, TASK)
    assert m.label_map == {"billing": "billing"}


def test_read_table_samples_large_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "SAMPLE_ROWS", 10)
    (tmp_path / "t.jsonl").write_text("".join(f'{{"text": "row {i}", "n": {i}}}\n' for i in range(100)))
    t = importer.read_table(tmp_path, "t.jsonl", random.Random(0))
    assert t.columns == ["text", "n"] and t.total_rows == 100 and len(t.rows) == 10


async def test_curation_uses_imported_rows_for_test(tmp_path):
    root = make_repo(tmp_path)
    events = []
    c = Curator(FakeLLM(), TASK, events.append, target=150, repo_path=str(root), use_embeddings=False)
    rows = await c.run()
    imported = [r for r in rows if r.source == "imported"]
    assert imported and all(r.meta["file"] == "clm_data/tickets.csv" for r in imported)
    # text is copied verbatim from the export, labels come from the queue column
    assert any(r.state == "Subject: Ticket 1\n\nI was charged twice for invoice #1" and r.label == "billing" for r in imported)
    assert not any("Something odd" in r.state for r in rows)
    test = [r for r in rows if r.split == "test"]
    assert test and all(r.source == "imported" for r in test)
    assert c.stats["test_real"] == len(test)
    # generation only fills the gap left by the 40 real rows per label (target 50)
    assert Counter(r.source for r in rows)["synthetic"] + Counter(r.source for r in rows)["hard_negative"] < 60
    tools = [e for e in events if e["type"] == "tool" and e["name"] == "import"]
    assert {t["input"] for t in tools} == {"clm_data/tickets.csv", "other.csv"}
    assert "skipped" in next(t["output"] for t in tools if t["input"] == "other.csv")


def test_discover_ignores_symlinks_out_of_repo(tmp_path):
    root = make_repo(tmp_path)
    secret = tmp_path / "secret.csv"
    secret.write_text("key,value\naws,SECRET\n")
    (root / "clm_data" / "leak.csv").symlink_to(secret)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "more.csv").write_text("a,b\n1,2\n")
    (root / "clm_data" / "linked_dir").symlink_to(outside, target_is_directory=True)
    found = importer.discover(root)
    assert "clm_data/tickets.csv" in found
    assert not any("leak" in f or "linked_dir" in f for f in found)


def test_discover_ignores_symlinked_clm_data(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "t.csv").write_text("a,b\n1,2\n")
    (root / "clm_data").symlink_to(real, target_is_directory=True)
    assert importer.discover(root) == []


async def test_recurate_imports_newly_exported_rows_into_train_val(tmp_path):
    root = make_repo(tmp_path, per_queue=10)
    first = await Curator(FakeLLM(), TASK, lambda e: None, target=150, repo_path=str(root), use_embeddings=False).run()
    test_ids = {r.id for r in first if r.split == "test"}
    # the user exports more rows after the first curation pass
    with open(root / "clm_data" / "tickets.csv", "a", newline="") as f:
        w = csv.writer(f)
        for i in range(500, 530):
            w.writerow([i, f"Ticket {i}", f"Late export about invoice #{i}", "Billing"])
    rows = await Curator(FakeLLM(), TASK, lambda e: None, target=150, repo_path=str(root), previous=first,
                         use_embeddings=False).run()
    late = [r for r in rows if "Late export" in r.state]
    assert late and all(r.source == "imported" and r.split in ("train", "val") for r in late)
    assert {r.id for r in rows if r.split == "test"} == test_ids
    assert len({r.state for r in rows}) == len(rows)   # earlier imports are not imported twice


def make_split_repo(tmp_path, per_queue=40, test_per_queue=12):
    """Like make_repo, but the export carries a `split` column pinning some rows to a frozen test set."""
    root = make_repo(tmp_path, per_queue=per_queue)
    with open(root / "clm_data" / "tickets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "subject", "body", "queue", "split"])
        i = 0
        for q, text in QUEUES.items():
            for _ in range(per_queue):
                i += 1
                w.writerow([i, f"Ticket {i}", f"{text} #{i}", q, "train"])
            for _ in range(test_per_queue):
                i += 1
                w.writerow([i, f"Ticket {i}", f"Held-out {text} #{i}", q, "test"])
        # a train row duplicating a frozen test row must not survive into train
        w.writerow([i, f"Ticket {i}", f"Held-out {text} #{i}", q, "train"])
    return root


def test_split_column_is_hidden_from_the_mapping():
    t = Table(rel="t.csv", columns=["subject", "body", "queue", "Split"],
              rows=[{"subject": "s", "body": "b", "queue": "Billing", "Split": "test"}], total_rows=1)
    assert "Split" not in importer.profile(t)
    m = TableMapping(relevant=True, reason="", template="{subject} {Split}")
    assert "unknown columns" in importer.validate(m, t, TASK)[1]


async def test_split_column_pins_a_frozen_test_split(tmp_path):
    root = make_split_repo(tmp_path)
    # target 60 -> 20 per label: train rows are sampled/balanced down, test rows are all kept
    c = Curator(FakeLLM(), TASK, lambda e: None, target=60, repo_path=str(root), use_embeddings=False)
    rows = await c.run()
    test = [r for r in rows if r.split == "test"]
    assert len(test) == 36 and all("Held-out" in r.state and r.meta.get("frozen_test") for r in test)
    assert Counter(r.label for r in test) == {"billing": 12, "technical": 12, "account": 12}
    others = [r for r in rows if r.split != "test"]
    assert others and not any("Held-out" in r.state for r in others)
    assert {r.split for r in others} == {"train", "val"}
    assert c.stats["test_real"] == 36 and c.stats["fixed_test"] == 36

    # a recurate keeps exactly that test split
    again = await Curator(FakeLLM(), TASK, lambda e: None, target=60, repo_path=str(root), previous=rows,
                          use_embeddings=False).run()
    assert {r.id for r in again if r.split == "test"} == {r.id for r in test}
    assert not any("Held-out" in r.state for r in again if r.split != "test")


async def test_split_column_refuses_a_sampled_table(tmp_path, monkeypatch):
    root = make_split_repo(tmp_path, per_queue=10, test_per_queue=2)
    monkeypatch.setattr(importer, "SAMPLE_ROWS", 5)
    events = []
    c = Curator(FakeLLM(), TASK, events.append, target=60, repo_path=str(root), use_embeddings=False)
    await c.import_data(20)
    out = next(e["output"] for e in events if e["type"] == "tool" and e["input"] == "clm_data/tickets.csv")
    assert out.startswith("skipped") and "split" in out


class TeacherFake(FakeLLM):
    """Labels imported rows by keyword, so unlabelled real rows can be teacher-labelled; counts the calls."""

    KEYS = {"charged": "billing", "crashes": "technical", "password": "account"}

    def __init__(self):
        super().__init__()
        self.real_label_calls = 0

    async def structured(self, prompt, schema, system=None, **kw):
        if schema.__name__ == "LabelBatch" and "Unlabelled export" in prompt:
            self.real_label_calls += 1
            items = re.findall(r"\[(\d+)\]\n([^\n]*)\n\n?([^\n]*)", prompt)
            return schema(labels=[{"i": int(i), "label": next((v for k, v in self.KEYS.items() if k in a + b), "billing"),
                                   "confidence": 0.7, "reason": "fake"} for i, a, b in items])
        return await super().structured(prompt, schema, system, **kw)


def make_unlabelled_repo(tmp_path, labelled=30, unlabelled=6, test_unlabelled=0):
    """Tickets with the queue column filled for `labelled` rows per queue and blank for `unlabelled`;
    `test_unlabelled` rows per queue are pinned to the frozen test split without a label."""
    root = make_repo(tmp_path)
    with open(root / "clm_data" / "tickets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "subject", "body", "queue", "split"])
        i = 0
        for q, text in QUEUES.items():
            for _ in range(labelled):
                i += 1
                w.writerow([i, f"Ticket {i}", f"{text} #{i}", q, "train"])
            for _ in range(unlabelled):
                i += 1
                w.writerow([i, f"Unlabelled export {i}", f"{text} #{i}", "", "train"])
            for _ in range(test_unlabelled):
                i += 1
                w.writerow([i, f"Unlabelled export {i}", f"Held-out {text} #{i}", "", "test"])
    return root


async def test_user_policy_queues_unlabelled_real_rows_without_a_teacher(tmp_path):
    from forge.labelling import LabelQueue
    root = make_unlabelled_repo(tmp_path, test_unlabelled=2)
    llm, q, events = TeacherFake(), LabelQueue(tmp_path / "q"), []
    c = Curator(llm, TASK, events.append, target=90, repo_path=str(root), use_embeddings=False,
                label_policy="user", queue=q)
    rows = await c.run()
    assert llm.real_label_calls == 0
    assert not any("Unlabelled export" in r.state for r in rows)          # needs_label never reaches the dataset
    pending = q.pending()
    assert pending and all("Unlabelled export" in r.state and r.label is None and r.label_source is None for r in pending)
    assert {r.split for r in pending} == {"train", "test"}                  # frozen-test rows keep their split
    assert sum(r.split == "test" for r in pending) == 6
    assert c.stats["needs_label"] == len(pending)
    assert events[-1]["counts"]["needs_label"] == len(pending)
    # export-labelled rows carry their source
    assert all(r.label_source == "export" for r in rows if r.source == "imported")


async def test_no_policy_means_the_labelling_agent(tmp_path):
    """No policy chosen = the default `agent` policy: curation queues the rows for the labelling stage's agent
    (no teacher call during curation, nothing reaches the dataset unlabelled)."""
    root = make_unlabelled_repo(tmp_path)
    llm, events = TeacherFake(), []
    c = Curator(llm, TASK, events.append, target=90, repo_path=str(root), use_embeddings=False)
    assert c.label_policy == "agent"
    await c.run()
    assert llm.real_label_calls == 0 and c.stats["needs_label"] == 18
    assert any("go to the labelling agent" in e.get("message", "") for e in events if e["type"] == "log")


async def test_mixed_policy_suggests_and_still_needs_a_label(tmp_path):
    from forge.labelling import LabelQueue
    root = make_unlabelled_repo(tmp_path, test_unlabelled=2)
    llm, q = TeacherFake(), LabelQueue(tmp_path / "q")
    rows = await Curator(llm, TASK, lambda e: None, target=90, repo_path=str(root), use_embeddings=False,
                         label_policy="mixed", queue=q).run()
    assert llm.real_label_calls >= 2                                        # train/val rows and frozen-test rows
    assert not any("Unlabelled export" in r.state for r in rows)
    pending = q.pending()
    assert len(pending) == 24 and all(r.suggested_label and r.teacher_conf == 0.7 for r in pending)
    assert all(r.label_state == "needs_label" for r in pending)
    assert next(r for r in pending if "charged" in r.state).suggested_label == "billing"


async def test_teacher_policy_labels_and_mirrors_for_review(tmp_path):
    from forge.labelling import LabelQueue
    root = make_unlabelled_repo(tmp_path, test_unlabelled=2)
    llm, q = TeacherFake(), LabelQueue(tmp_path / "q")
    c = Curator(llm, TASK, lambda e: None, target=90, repo_path=str(root), use_embeddings=False,
                label_policy="teacher", queue=q)
    rows = await c.run()
    taught = [r for r in rows if "Unlabelled export" in r.state]
    assert taught and all(r.label_source == "teacher" and r.meta["label_origin"] == "teacher" for r in taught)
    assert sum(r.split == "test" for r in taught) == 6
    assert q.pending() == [] and c.stats["needs_label"] == 0
    mirrored = {r.rid: r for r in q.rows()}
    assert {r.id for r in taught} <= set(mirrored)
    assert all(mirrored[r.id].label_state == "labelled" and mirrored[r.id].label_source == "teacher" for r in taught)
    # a recurate re-imports the same export and leaves the queue as it was
    assert await __import__("asyncio").to_thread(q.add, [mirrored[taught[0].id]]) == 0


class ShapingFake(FakeLLM):
    """Generates, per batch, one copy of a real exported ticket and one text with a leftover placeholder next to
    novel texts; records the generation prompts."""

    def __init__(self):
        super().__init__()
        self.gen_prompts = []

    async def structured(self, prompt, schema, system=None, **kw):
        if schema.__name__ == "GenBatch":
            self.gen_prompts.append(prompt)
            lab = re.search(r"correct label is (\w+)", prompt).group(1)
            n = int(re.search(r"Write (\d+)", prompt).group(1))
            import random
            novel = [f"{lab} issue #{random.random():.8f} needs help" for _ in range(max(0, n - 2))]
            return schema(states=novel + [f"{lab} issue Ticket 5 I was charged twice for invoice #5 again",
                                          f"{lab} issue from <NAME> about the account"])
        return await super().structured(prompt, schema, system, **kw)


async def test_generation_is_shaped_on_scrubbed_real_rows_train_only_and_leak_checked(tmp_path):
    from forge.models import REAL_SHAPED
    root = make_repo(tmp_path)
    llm = ShapingFake()
    c = Curator(llm, TASK, lambda e: None, target=150, repo_path=str(root), use_embeddings=False)
    rows = await c.run()
    p = llm.gen_prompts[0]
    assert "Real examples of this state text, scrubbed" in p and p.count("[exemplar ") == 4
    # exemplars are scrubbed: ticket numbers became placeholders, no raw "#<digits>" survives
    block = p.split("Real examples")[1].split("Write ")[0]
    assert "<NUM>" in block and not re.search(r"#\d", block)
    shaped = [r for r in rows if r.provenance == REAL_SHAPED]
    assert shaped and all(r.split == "train" for r in shaped)
    assert all(len(r.meta["exemplar_hashes"]) == 4 and r.meta["batch"] for r in shaped)
    drops = c.stats["shaped_drops"]
    assert drops.get("placeholder", 0) >= 1                         # a leftover placeholder is never kept
    assert not any("charged twice for invoice #5 again" in r.state for r in rows)   # the copy was dropped
    assert sum(v for k, v in drops.items() if k != "placeholder") >= 1
    assert c.stats["shaped"] == len(shaped)


def test_scrub_and_leak_guard():
    from forge import shaping
    t = "Hi, this is Maria Lopez (maria@example.com, +1 415 555 0100). Order 8f14e45fceea167a5a36 failed, see https://x.io/a"
    s = shaping.scrub(t)
    assert "Maria" not in s and "<EMAIL>" in s and "<PHONE>" in s and "<URL>" in s and "<ID>" in s
    assert "<NAME>" in s and "example.com" not in s
    g = shaping.LeakGuard(forbidden=["the quick brown fox jumps over the lazy dog today"],
                          real=["please refund the duplicate charge on my march invoice right away thanks",
                                "Contact Maria about it"])
    assert g.check("the quick brown fox jumps over the lazy dog today") == "forbidden"
    assert g.check("could you refund the duplicate charge on my march invoice right away thanks", real_shaped=True) \
        == "shared_8gram"
    assert g.check("a note for Maria regarding the plan", real_shaped=True) == "identifier"
    assert g.check("a note for Maria regarding the plan", real_shaped=False) is None
    assert g.check("my card was billed twice this month, please fix it") is None


def test_scrub_covers_field_values_sentence_starts_and_the_template():
    """a first name after a colon ("sender: Jordan Pike") and a name opening a sentence reached
    the prompt, and the structure descriptor was read from the unscrubbed text (a chat line "Jordan Pike: ..."
    puts the name in the template)."""
    from forge import shaping
    a = "sender: Jordan Pike\ncontent: Jordan will join late. Can you hold the room?"
    b = "We spoke with Jordan yesterday about the venue."
    names = shaping.pool_names([a, b])
    out = shaping.scrub(a, names)
    assert "Jordan" not in out and "Pike" not in out and out.startswith("sender: <NAME> <NAME>")
    chat = "Jordan Pike: running late\nBrandon: ok"
    clean = shaping.scrub(chat, shaping.pool_names([chat, "talked to Jordan Pike and Casey"]))
    assert "jordan" not in shaping.key_template(clean).lower() and "casey" not in shaping.key_template(clean).lower()
    assert shaping.scrub("Monday works. Thanks!") == "Monday works. Thanks!"             # common words stay


class FailingShapingFake(ShapingFake):
    """A generation call that fails quoting its prompt back (as a provider error may)."""

    async def structured(self, prompt, schema, system=None, **kw):
        if schema.__name__ == "GenBatch":
            from forge.llm.base import LLMError
            raise LLMError("structured output invalid: " + prompt.split("Real examples")[1][:2000])
        return await super().structured(prompt, schema, system, **kw)


async def test_a_failed_generation_never_logs_real_text(tmp_path):
    root = make_repo(tmp_path)
    logs = []
    c = Curator(FailingShapingFake(), TASK, lambda e: logs.append(e), target=150, repo_path=str(root),
                use_embeddings=False)
    try:
        await c.run()
    except Exception:  # noqa: BLE001 - nothing generated: the run may refuse later; only the log matters here
        pass
    fails = [e["message"] for e in logs if e.get("type") == "log" and "generation batch" in e.get("message", "")]
    assert fails and all(len(m) < 400 and not re.search(r"#\d", m) for m in fails)
