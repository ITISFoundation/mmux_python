from typing import Callable, List
import logging
import uuid
import traceback
import contextlib
import os
import sys
from pathlib import Path
import dakota.environment as dakenv

from pathos.pools import ProcessPool as Pool

# from multiprocessing.pool import ThreadPool

logger = logging.getLogger(__name__)
logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


class DummyFile:
    def write(self, x):
        pass

    def flush(self):
        pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


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

    def model_wrapper(self, param_set):
        return self.model(**param_set)

    def evaluate(self, params_set: List[dict]):
        outputs_set = []
        logger.info(f"Evaluating {len(params_set)} parameter sets")
        logger.debug(f"Evaluating: {params_set}")
        assert (
            self.n_runners > 0
        ), "A negative (or zero) number of runners is not allowed."
        if self.n_runners == 1:
            for param_set in params_set:
                outputs_set.append(self.model_wrapper(param_set))
        else:
            # with ThreadPool(self.n_runners) as pool:
            #     outputs_set = pool.map(func=self.model_wrapper, iterable=params_set)
            with Pool(self.n_runners) as pool:
                outputs_set = pool.map(self.model_wrapper, params_set)
                # pool.close()
                # pool.join()

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
            with nostdout():
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
        if self.map_object:
            # callbacks = {"model": self.model_callback}
            callback = self.model_callback
        else:
            logger.info(
                "No Map object was provided to DakotaObject. "
                "Therefore, it is assumed this is a Dakota-internal calculation"
                "and no callback is necessary."
            )
            # callbacks = {}
            callback = None
        print("Starting dakota")
        dakota_restart_path = output_dir / "dakota.rst"
        with working_directory(output_dir):
            study = dakenv.study(  # type: ignore
                # callbacks=callbacks,
                callback=callback,
                input_string=dakota_conf,
                read_restart=(
                    str(dakota_restart_path) if dakota_restart_path.exists() else ""
                ),
            )
            study.execute()


# if __name__ == "__main__":
# import logging

# logger = logging.getLogger(__name__)
# logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)
# logger.debug("This message should go to the log file")
# logger.info("So should this")
# logger.warning("And this, too")
# logger.error("And non-ASCII stuff, too, like Øresund and Malmö")
