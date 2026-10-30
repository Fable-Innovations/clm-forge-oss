"""forge.encoder_choice: the encoder Forge recommends for a new experiment or target (task profile, measured memory
peak vs this machine's model budget, backends, a preset's own encoder). No model is loaded."""
from forge import encoders
from forge.encoder_choice import profile_of, recommend
from forge.memory import GiB
from forge.models import TaskSchema


def _task(qtype: str, criteria) -> TaskSchema:
    return TaskSchema(name="t", question_id="q", type=qtype, instructions="?", criteria=criteria,
                      state_description="s", state_example="s")


def test_profiles():
    assert profile_of(None) == "general"
    assert profile_of(_task("noul", None)) == "simple"
    assert profile_of(_task("choice", {"a": "A", "b": "B", "c": "C"})) == "simple"
    assert profile_of(_task("choice", {k: k for k in "abcde"})) == "rich"          # more than four options
    assert profile_of(_task("choice", None)) == "rich"                               # options come with each call
    assert profile_of(_task("score", ["low", "mid", "high"])) == "rich"


def test_general_and_simple_pick_the_shipping_4b_build():
    r = recommend(budget_bytes=int(28.8 * GiB))
    assert r.key == "qwen3-4b-gguf-q4km" and r.profile == "general"
    assert "2560-d" in r.reason and "GGUF build is the one packages ship on" in r.reason and "28.8 GB" in r.reason
    assert recommend(_task("noul", None), budget_bytes=int(28.8 * GiB)).key == "qwen3-4b-gguf-q4km"
    winnow = next(c for c in r.considered if c.key == "winnow-12b-gguf-q8")
    assert not winnow.fits and "never measured" in winnow.why_not               # listed, never recommended


def test_rich_picks_the_widest_that_fits_and_prefers_the_reference_encoder():
    r = recommend(_task("choice", None), budget_bytes=int(28.8 * GiB))
    assert r.key == encoders.REFERENCE and "warm-starts" in r.reason
    # a budget the 8B peak exceeds: the widest that still fits
    r = recommend(_task("choice", None), budget_bytes=int(5 * GiB))
    assert encoders.get(r.key).hidden == 2560
    assert next(c for c in r.considered if c.key == encoders.REFERENCE).why_not.startswith("its measured peak")


def test_backends_preferred_and_nothing_fits():
    r = recommend(backends=("mlx",), preferred=("qwen3-4b",), budget_bytes=int(28.8 * GiB))
    assert r.key == "qwen3-4b" and "built for" in r.reason
    assert all(encoders.get(c.key).backend == "mlx" for c in r.considered if c.fits)
    # the preferred one does not fit: the profile ranking decides among those that do
    r = recommend(backends=("mlx",), preferred=("qwen3-4b",), budget_bytes=int(3.0 * GiB))
    assert r.key in {"qwen3-0.6b", "qwen3-0.6b-8bit", "qwen3-0.6b-bf16"} and "built for" not in r.reason
    none = recommend(budget_bytes=int(1 * GiB))
    assert none.key is None and "no registered encoder fits the 1.0 GB" in none.reason
