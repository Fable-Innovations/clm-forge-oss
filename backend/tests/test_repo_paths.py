"""Repo reads stay inside the repo: no symlinks out, no `..`, no absolute paths (repos may be cloned
from untrusted URLs and file refs come from an LLM)."""
import subprocess

from forge import repo
from forge.analyst import _ground

from test_agent_fakes import proposal


def make(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "router.py").write_text("def route(t):\n    return 'billing'\n")
    secret = tmp_path / "secret.txt"
    secret.write_text("AWS_SECRET=hunter2\n")
    (root / "notes.txt").symlink_to(secret)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.py").write_text("SECRET = 1\n")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root, secret


def test_safe_path(tmp_path):
    root, secret = make(tmp_path)
    assert repo.safe_path(root, "src/router.py") == root / "src" / "router.py"
    for bad in ["notes.txt", "linked/x.py", "../secret.txt", "src/../../secret.txt", str(secret), ""]:
        assert repo.safe_path(root, bad) is None, bad
    assert repo.read_text(root, "notes.txt") == ""


def test_list_files_skips_symlinks(tmp_path):
    root, _ = make(tmp_path)
    files = repo.list_files(root)
    assert "src/router.py" in files
    assert not any(f.startswith(("notes", "linked")) for f in files)


def test_ground_drops_unsafe_llm_file_refs(tmp_path):
    root, _ = make(tmp_path)
    p = proposal()
    o = p.opportunities[0]
    o.files[0].path = "src/router.py"
    extra = o.files[0].model_copy(update={"path": "../secret.txt"})
    linked = o.files[0].model_copy(update={"path": "notes.txt"})
    o.files += [extra, linked]
    p, warnings = _ground(p, str(root))
    paths = [f.path for f in p.opportunities[0].files]
    assert paths == ["src/router.py"]
    assert not any("hunter2" in (f.snippet or "") for f in p.opportunities[0].files)
    assert sum("unsafe" in w for w in warnings) == 2


def test_grep_and_read_reach_the_tail_of_large_listed_files(tmp_path):
    from forge.llm.openai import RepoTools
    root = tmp_path / "repo"
    root.mkdir()
    body = "x = 1  # filler line to push the target past 200k chars\n" * 5000   # ~290k chars, < MAX_FILE_BYTES
    (root / "big.py").write_text(body + "def on_vad_state(state):\n    pass\n")
    assert "big.py" in repo.list_files(root)
    assert "def on_vad_state" in repo.read_text(root, "big.py")
    out, _ = RepoTools(root).run("grep", {"pattern": "def on_vad_state"})
    assert "big.py:5001:" in out
