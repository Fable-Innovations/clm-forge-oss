"""tests/conftest.py: a ``model_loading`` test that Forge's memory admission refuses is skipped with the reason; any
other failure, and an unmarked test, still fail. No model is loaded (the refusal is raised directly)."""
from pathlib import Path

pytest_plugins = ["pytester"]

CONFTEST = (Path(__file__).with_name("conftest.py")).read_text()


def test_refusal_skips_only_marked_tests(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_x='''
import pytest
from forge.memory import MemoryAdmissionError

@pytest.mark.model_loading
def test_refused():
    try:
        raise MemoryAdmissionError("encoder needs ~6.4 GB but only 2.0 GB is free after the 7.2 GB reserve")
    except MemoryAdmissionError as e:
        raise RuntimeError("embedding failed") from e        # wrapped: still a refusal

@pytest.mark.model_loading
def test_other_error_fails():
    raise ValueError("a real bug")

def test_unmarked_refusal_fails():
    raise MemoryAdmissionError("not a model-loading test")
''')
    res = pytester.runpytest("-rs", "-p", "no:cacheprovider")
    res.assert_outcomes(skipped=1, failed=2)
    res.stdout.fnmatch_lines(["*model memory refused on this machine: encoder needs ~6.4 GB*"])
