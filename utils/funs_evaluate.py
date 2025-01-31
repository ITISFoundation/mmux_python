### Allows to evaluate in different modes - batch, single set, ...
### also in OSPARC or local deployment
from typing import Callable, List
from pathlib import Path
import dakota.environment as dakenv
import datetime
import os


## TODO switch to test-based development!!


### Otherwise could create a class that I can instantiate and give a "model" at initialization time,
# which is given to "run dakota" as input parameter
def batch_evaluator(model: Callable, batch_input: List[dict]):
    return map(model, batch_input)  # FIXME not sure this will work


def batch_evaluator_local(model: Callable, batch_input: List[dict]):
    return [
        {"fns": [v for v in response.values()]} for response in map(model, batch_input)
    ]


def single_evaluator(model: Callable, input: dict):
    return model(input)


def create_run_dir(script_dir: Path, dir_name: str = "sampling"):
    ## part 1 - setup
    main_runs_dir = script_dir / "runs"
    current_time = datetime.datetime.now().strftime("%Y%m%d.%H%M%S%d")
    temp_dir = main_runs_dir / "_".join(["dakota", current_time, dir_name])
    print(str(temp_dir))
    os.makedirs(temp_dir, exist_ok=True)
    print("temp_dir: ", temp_dir)
    return temp_dir


def run_dakota(dakota_conf_path: Path, batch_mode: bool = True):
    print("Starting dakota")
    dakota_conf = dakota_conf_path.read_text()
    callbacks = (
        {"batch_evaluator": batch_evaluator}
        if batch_mode
        else {"evaluator": single_evaluator}  # not sure this will work
    )
    study = dakenv.study(
        callbacks=callbacks,
        input_string=dakota_conf,
    )
    study.execute()
    ## TODO access documentation of dakenv.study -- cannot, also cannot find in https://github.com/snl-dakota/dakota/tree/devel/packages
    # would need to ask Werner
    """
    Help on class study in module dakota.environment.environment:

class study(pybind11_builtins.pybind11_object)
 |  Method resolution order:
 |      study
 |      pybind11_builtins.pybind11_object
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __init__(...)
 |      __init__(*args, **kwargs)
 |      Overloaded function.
 |
 |      1. __init__(self: dakota.environment.environment.study, callback: object, input_string: str, read_restart: str = '') -> None
 |
 |      2. __init__(self: dakota.environment.environment.study, callbacks: dict, input_string: str, read_restart: str = '') -> None
 |
 |  execute(...)
 |      execute(self: dakota.environment.environment.study) -> None
 |
 |  response_results(...)
 |      response_results(self: dakota.environment.environment.study) -> dakota.environment.environment.Response
 |
 |  variables_results(...)
 |      variables_results(self: dakota.environment.environment.study) -> dakota.environment.environment.Variables
 |
 |  ----------------------------------------------------------------------
 |  Static methods inherited from pybind11_builtins.pybind11_object:
 |
 |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
 |      Create and return a new object.  See help(type) for accurate signature.

    """


if __name__ == "__main__":
    print("DONE")
