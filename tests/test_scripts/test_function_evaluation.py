import pytest
from tests.test_utils.test_funs_git import get_model_from_optistim_repo


def test_evaluate_function():
    model = get_model_from_optistim_repo()
    results = model(**{i: 0.0 for i in model.__annotations__["inputs"].keys()})
    assert results["activation"] == pytest.approx(0.0)
    assert results["energy"] == pytest.approx(0.0)
    assert results["maxamp"] == pytest.approx(0.0)


# def test_sweep_function():
#     ...
