from typing import Callable, List
import uuid
import traceback
import contextlib
import os
from pathlib import Path
import dakota.environment as dakenv
import logging
import wiofiles as wio
import sys

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


class Map:
    ## TODO should the "model" Callable be given here or in DakotaObject?
    def __init__(self, model: Callable, n_runners: int = 1) -> None:
        logger.info("Creating caller map")
        self.model = model
        self.uuid = str(uuid.uuid4())
        self.map_uuid = None
        self.n_runners = n_runners
        logger.info(f"Optimizer uuid is {self.uuid}")
        pass

    def evaluate(self, params_set: List[dict]):
        outputs_set = []
        logger.info(f"Evaluating {len(params_set)} parameter sets")
        logger.debug(f"Evaluating: {params_set}")
        assert (
            self.n_runners > 0
        ), "A negative (or zero) number of runners is not allowed."
        if self.n_runners == 1:
            for param_set in params_set:
                outputs_set.append(self.model(**param_set))
            # raise ValueError(f"This is the output: {outputs_set}")
        else:
            # TODO use multiprocessing; return in strict order
            raise NotImplementedError(f"params_set: {params_set}")

        return outputs_set


class DakotaObject:
    def __init__(self, map_object: Map | None) -> None:
        self.map_object = map_object
        logger.info("DakotaObject created")

    def model_callback(self, dak_inputs: List[dict]) -> List[dict]:
        try:
            logger.info("Into model_callback")
            param_sets = [
                {
                    **{
                        label: value
                        for label, value in zip(dak_input["cv_labels"], dak_input["cv"])
                    },
                    **{
                        label: value
                        for label, value in zip(
                            dak_input["div_labels"], dak_input["div"]
                        )
                    },
                }
                for dak_input in dak_inputs
            ]
            all_response_labels = [
                dak_input["function_labels"] for dak_input in dak_inputs
            ]
            assert (
                self.map_object is not None
            ), "model_callback should not be executed if map_object is None"
            obj_sets = self.map_object.evaluate(param_sets)
            dak_outputs = [
                {"fns": [obj_set[response_label] for response_label in response_labels]}
                for obj_set, response_labels in zip(obj_sets, all_response_labels)
            ]
            return dak_outputs
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def run(self, dakota_conf: str, output_dir: Path):
        print("Starting dakota")
        with working_directory(output_dir):
            stdout, stderr = self.future_exec(
                func=self.model_callback if self.map_object else None,
                conf=dakota_conf,
            )
            print("Dakota run finished")
            with open("dakota_stdout.txt", "w") as f_out, open("dakota_stderr.txt", "w") as f_err:
                if stdout:
                    f_out.write(stdout)
                if stderr:
                    f_err.write(stderr)
            if stderr:
                print(stderr, file=sys.stderr)
    
    def dak_exec(self, func, conf):
        study = dakenv.study(callbacks={'map': func}, input_string=conf) # type: ignore
        stdoutstr, stderrstr = None, None
        with wio.capture_to_file(stdout='./stdout', stderr='./stderr') as (stdout, stderr):
            study.execute()
        with open(stdout) as outf, open(stderr) as errf:
            stdoutstr = outf.read()
            stderrstr = errf.read()
        del study
        return stdoutstr, stderrstr

    def future_exec(self, func, conf):
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(1) as pool:
            future = pool.submit(self.dak_exec, func, conf)
        return future.result()