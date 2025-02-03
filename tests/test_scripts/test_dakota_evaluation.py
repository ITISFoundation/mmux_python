import pytest
from pathlib import Path
from tests.test_utils.test_funs_git import get_model_from_optistim_repo, create_run_dir
from utils.funs_create_dakota_conf import create_function_sampling
from utils.dakota_object import DakotaObject, Map


## TODO also port here the evaluation through Dakota of whatever samples in a file
## (overkill for some ocassions, but useful overall, specially if allows to re-use well-optimized code and parallelization)
@pytest.mark.skip(reason="Dakota interface Not implemented yet")
def test_sweep_dakota():
    "use the 'evaluate' workflow I designed for Dakota"
    assert NotImplementedError


def test_create_sampling_dakota_file():
    """
    Create the Dakota file and compare to a previous version. If not equal, throw an error.
    If REFERENCE gets set to True, then overwrite instead.
    """
    REFERENCE = True
    REFERENCE_FILE = Path(__file__).parent / "dakota_sampling.in"
    model = get_model_from_optistim_repo()
    dakota_conf = create_function_sampling(
        model,
        dakota_conf_file=REFERENCE_FILE if REFERENCE else None,
        # TODO think. What is best for Dakota interfacing? All in the runfolder, just use file names? Or absolute paths?
        ## Probably full, declarative paths is most consistent...
    )
    if not REFERENCE:
        ref_dakota_file = Path(REFERENCE_FILE)
        with ref_dakota_file.open("r") as ref_file:
            ref_dakota_conf = ref_file.read()
        assert (
            ref_dakota_conf == dakota_conf
        ), "Generated Dakota file does not match the reference file."
    else:
        ref_dakota_file = Path(REFERENCE_FILE)
        with ref_dakota_file.open("w") as ref_file:
            ref_file.write(dakota_conf)


@pytest.mark.parametrize("n_runners", [1, 10])
def test_run_sampling_with_dakota(n_runners):
    run_dir = create_run_dir(Path.cwd(), "sampling")
    model = get_model_from_optistim_repo(run_dir)
    map = Map(model, n_runners=n_runners)
    dakobj = DakotaObject(map)
    dakota_conf = create_function_sampling(fun=model, num_samples=2, batch_mode=True)
    dakobj.run(dakota_conf, output_dir=run_dir)
