import pytest
from tests.test_utils.test_funs_git import get_model_from_optistim_repo


def test_evaluate_function():
    model = get_model_from_optistim_repo()
    results = model(**{i: 0.0 for i in model.__annotations__["inputs"].keys()})
    assert results["activation"] == pytest.approx(0.0)
    assert results["energy"] == pytest.approx(0.0)
    assert results["maxamp"] == pytest.approx(0.0)


@pytest.mark.skip("Not implemented yet")
def test_sweep_function():
    """Manually create & perform a sweep through the function; plot the results"""


# ## sweep
# import numpy as np
# import pandas as pd

# NVARS = import_function_from_repo(
#     temp_dir, module_name="get_pulse.py", function_name="NVARS"
# )
# SEGMENT_PW = import_function_from_repo(
#     temp_dir, module_name="get_pulse.py", function_name="SEGMENT_PW"
# )


# def create_comfort_amplitude_sweep(pw=1.0):
#     var_names = [f"p{i+1}" for i in range(NVARS)]
#     series = []
#     nseg_per_pw = round(pw / SEGMENT_PW)
#     for amp in np.arange(-1, 1.01, 0.1):
#         amp = np.round(amp, 1)
#         vars = []
#         for _ in range(5):
#             for _ in range(nseg_per_pw):
#                 vars.append(amp)
#             for _ in range(nseg_per_pw):
#                 vars.append(0)
#         for _ in range(5):
#             for _ in range(nseg_per_pw):
#                 vars.append(-amp)
#         series.append(pd.Series(vars, index=var_names))
#         ## NB: if pw is not 1.0, will need to fill up with 0s
#     df = pd.concat(series, axis=1).T
#     df.to_csv(script_dir / "COMFORT_1ms_amplitudesweep.csv", index=False, sep=" ")
#     return df


# for  ## TODO the idea would be to generate a map with multithread
## first, see how to interface w dakota, that is more important. Then, move from there.


## TODO also port here the evaluation through Dakota of whatever samples in a file
## (overkill for some ocassions, but useful overall, specially if allows to re-use well-optimized code and parallelization)
