"""Head-set presets (forge.autopilot.presets): the presets file (missing, malformed, valid, re-read on change), the
preset schema's checks, and a head-set target's sources validated against its preset. No adapter runs."""
import json

import pytest
from pydantic import ValidationError

import headset_presets
from forge.autopilot import presets
from forge.autopilot.targets import Target
from forge.autopilot.validation import plain_errors


def _target(root, **hs):
    p = presets.get("test-heads")
    return Target.model_validate({"id": "t", "kind": "headset",
                                  "headset": {**p.headset("qwen3-0.6b"), **hs}})


def _fields(e: ValidationError) -> dict:
    return {x["field"]: x["message"] for x in plain_errors(e)}


def test_a_missing_file_means_no_presets(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_HEADSET_PRESETS", str(tmp_path / "nope.json"))
    assert presets.load() == {} and presets.all() == [] and presets.env_passthrough() == set()
    with pytest.raises(KeyError, match="known: none; add one to"):
        presets.get("anything")


def test_the_default_path_is_in_the_data_folder(monkeypatch, tmp_path):
    from forge import paths
    monkeypatch.delenv("FORGE_HEADSET_PRESETS")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    assert presets.path() == tmp_path / "headset_presets.json"
    headset_presets.use(monkeypatch, tmp_path / "headset_presets.json", headset_presets.preset(tmp_path))
    monkeypatch.delenv("FORGE_HEADSET_PRESETS")
    assert list(presets.load()) == ["test-heads"]


@pytest.mark.parametrize("text, where", [
    ("{not json", "not valid JSON"),
    (json.dumps({"presets": [{"key": "x"}]}), "presets.0.name"),
    (json.dumps({"preset": []}), "preset"),
])
def test_a_malformed_file_is_named(tmp_path, monkeypatch, text, where):
    f = tmp_path / "presets.json"
    f.write_text(text)
    monkeypatch.setenv("FORGE_HEADSET_PRESETS", str(f))
    with pytest.raises(presets.PresetFileError) as e:
        presets.load()
    assert str(f) in str(e.value) and where in str(e.value)
    assert presets.env_passthrough() == set()


@pytest.mark.parametrize("change, message", [
    ({"key": "Bad Key"}, "key 'Bad Key'"),
    ({"cwd": "relative/dir"}, "cwd must be an absolute path"),
    ({"protected_files": ["frozen.jsonl"]}, "protected_files must be an absolute path"),
    ({"env_passthrough": ["NOT-A-NAME"]}, "not an environment variable name"),
    ({"sources": [headset_presets.SOURCES[0], headset_presets.SOURCES[0]]}, "each source kind can be declared once"),
    ({"sources": [{"kind": "Event Log"}]}, "source kind 'Event Log'"),
    ({"sources": [{"kind": "s", "settings": {"stage": {"values": ["a"], "default": "b"}}}]}, "not among values"),
    ({"sources": [{"kind": "s", "settings": {"stage": {"values": ["a"], "default": "a", "multiple": True}}}]},
     "a multiple setting's default is a list"),
    ({"sources": [{"kind": "s", "settings": {"stage": {"values": ["a"], "default": ["a"]}}}]},
     "a single setting's default is one value"),
    ({"adapter": []}, "presets.0.adapter"),
])
def test_preset_schema_checks(tmp_path, monkeypatch, change, message):
    headset_presets.use(monkeypatch, tmp_path / "p.json", headset_presets.preset(tmp_path, **change))
    with pytest.raises(presets.PresetFileError, match="p.json") as e:
        presets.load()
    assert message in str(e.value)


def test_duplicate_keys_are_refused(tmp_path, monkeypatch):
    p = headset_presets.preset(tmp_path)
    headset_presets.use(monkeypatch, tmp_path / "p.json", p, p)
    with pytest.raises(presets.PresetFileError, match="preset keys must be unique: test-heads"):
        presets.load()


def test_a_valid_file_loads_and_is_reread_when_it_changes(tmp_path, monkeypatch):
    f = headset_presets.use(monkeypatch, tmp_path / "p.json",
                            headset_presets.preset(tmp_path, env_passthrough=["MY_TOKEN_CMD"]))
    p = presets.get("test-heads")
    assert p.adapter == ("x",) and [s.kind for s in p.sources] == ["event_log", "shared_export"]
    assert p.adapter_config_keys == ("epochs",) and presets.env_passthrough() == {"MY_TOKEN_CMD"}
    body = p.headset("qwen3-0.6b")
    assert body["sources"] == [{"kind": "event_log", "enabled": True, "settings": {"stage": "qa"}},
                               {"kind": "shared_export", "enabled": False, "settings": {"stages": ["qa", "prod"]}}]
    headset_presets.use(monkeypatch, f, headset_presets.preset(tmp_path),
                        headset_presets.preset(tmp_path, key="second-set", name="Second"))
    assert set(presets.load()) == {"test-heads", "second-set"}


def test_sources_are_checked_against_the_preset(tmp_path, monkeypatch):
    headset_presets.use(monkeypatch, tmp_path / "p.json", headset_presets.preset(tmp_path))
    # settings left out get the preset's defaults
    t = _target(tmp_path, sources=[{"kind": "event_log"}, {"kind": "shared_export", "settings": {"stages": ["prod"]}}])
    assert [(s.kind, s.enabled, s.settings) for s in t.headset.sources] == [
        ("event_log", True, {"stage": "qa"}), ("shared_export", True, {"stages": ["prod"]})]
    assert t.headset.target_config()["sources"][1] == {"kind": "shared_export", "enabled": True,
                                                       "settings": {"stages": ["prod"]}}
    for sources, want in (
        ([{"kind": "rss"}], {"headset.sources.0.kind"}),
        ([{"kind": "event_log", "settings": {"stage": "staging"}}], {"headset.sources.0.settings.stage"}),
        ([{"kind": "event_log", "settings": {"stage": ["qa"]}}], {"headset.sources.0.settings.stage"}),
        ([{"kind": "event_log", "settings": {"region": "eu"}}], {"headset.sources.0.settings.region"}),
        ([{"kind": "shared_export", "settings": {"stages": []}}], {"headset.sources.0.settings.stages"}),
        ([{"kind": "shared_export", "settings": {"stages": ["qa", "qa"]}}], {"headset.sources.0.settings.stages"}),
        ([{"kind": "shared_export", "settings": {"stages": "qa"}}], {"headset.sources.0.settings.stages"}),
        ([{"kind": "event_log"}, {"kind": "event_log"}], {"headset.sources"}),
        ([{"kind": "Event-Log"}], {"headset.sources.0.kind"}),
    ):
        with pytest.raises(ValidationError) as e:
            _target(tmp_path, sources=sources)
        assert set(_fields(e.value)) == want, sources


def test_only_the_presets_data_keys_may_be_set(tmp_path, monkeypatch):
    headset_presets.use(monkeypatch, tmp_path / "p.json", headset_presets.preset(tmp_path))
    assert _target(tmp_path, adapter_config={"epochs": 2}).headset.target_config()["epochs"] == 2
    with pytest.raises(ValidationError) as e:
        _target(tmp_path, adapter_config={"python": "/tmp/evil"})
    msg = _fields(e.value)["headset.adapter_config"]
    assert "python cannot be set on a target (allowed: epochs)" in msg


def test_a_stored_row_takes_its_server_fields_from_the_preset(tmp_path, monkeypatch):
    """Whatever a stored row holds for the adapter argv, folder or environment, the preset's are what load."""
    from forge.autopilot.targets import Targets
    from forge.store import Store
    headset_presets.use(monkeypatch, tmp_path / "p.json", headset_presets.preset(tmp_path, env={"A": "1"}))
    store = Store(tmp_path / "f.db")
    targets = Targets(store)
    targets.put(_target(tmp_path))
    assert "adapter" not in store.targets()[0]["config"]["headset"]                 # never stored
    cfg = store.targets()[0]["config"]
    store.save_target("t", "headset", {**cfg, "headset": {**cfg["headset"], "adapter": ["sh", "-c", "evil"],
                                                          "cwd": "/", "env": {"PYTHONPATH": "/x"}}}, True)
    hs = targets.get("t").headset
    assert hs.adapter == ["x"] and hs.cwd == str(tmp_path) and hs.env == {"A": "1"}
