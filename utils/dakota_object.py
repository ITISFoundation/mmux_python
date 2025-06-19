from typing import Callable, List, Dict, Any, Tuple, Optional
import uuid
import traceback
import contextlib
import os
from pathlib import Path
import dakota.environment as dakenv
import logging
import wiofiles as wio
import sys
import concurrent.futures
import functools

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


# Static function to execute Dakota - needs to be at module level to be picklable
def _dak_exec_static(func, conf):
    """Static version of dak_exec that can be pickled for multiprocessing."""
    study = dakenv.study(callbacks={'map': func}, input_string=conf) # type: ignore
    stdoutstr, stderrstr = None, None
    with wio.capture_to_file(stdout='./stdout', stderr='./stderr') as (stdout, stderr):
        study.execute()
    with open(stdout) as outf, open(stderr) as errf:
        stdoutstr = outf.read()
        stderrstr = errf.read()
    del study
    return stdoutstr, stderrstr


# Wrapper function for model_callback that can be pickled
def _model_callback_wrapper(param_sets: List[Dict[str, Any]], 
                           evaluate_func: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Wrapper for model_callback that takes only serializable arguments."""
    try:
        logger.info("Into model_callback wrapper")
        obj_sets = evaluate_func(param_sets)
        
        all_response_labels = [dak_input.get("function_labels", []) for dak_input in param_sets]
        
        dak_outputs = [
            {"fns": [obj_set[response_label] for response_label in response_labels]}
            for obj_set, response_labels in zip(obj_sets, all_response_labels)
        ]
        return dak_outputs
    except Exception as e:
        print(traceback.format_exc())
        raise e


class Map:
    def __init__(self, model: Callable, n_runners: int = 1) -> None:
        logger.info("Creating caller map")
        self.model = model
        self.uuid = str(uuid.uuid4())
        self.map_uuid = None
        self.n_runners = n_runners
        logger.info(f"Optimizer uuid is {self.uuid}")
        
        # Check if model is picklable
        if n_runners > 1:
            import pickle
            try:
                pickle.dumps(model)
            except (TypeError, pickle.PicklingError):
                logger.warning("The model function is not picklable. Forcing n_runners=1")
                self.n_runners = 1

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
        else:
            # Use ProcessPoolExecutor for parallel evaluation
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_runners) as executor:
                # Submit each parameter set as a separate task
                futures = [executor.submit(self._run_single_model, param_set) for param_set in params_set]
                # Collect results in order
                outputs_set = [future.result() for future in futures]

        return outputs_set
        
    def _run_single_model(self, param_set):
        """Helper method to run a single model evaluation - can be used with ProcessPoolExecutor."""
        return self.model(**param_set)


class DakotaObject:
    def __init__(self, map_object: Optional[Map] = None) -> None:
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
            # Create a picklable version of the callback
            if self.map_object:
                # Instead of passing the instance method directly, we pass the 
                # necessary data to the static function
                callback = functools.partial(_model_callback_wrapper, 
                                           evaluate_func=self.map_object.evaluate)
            else:
                callback = None
            
            stdout, stderr = self.future_exec(
                func=callback,
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
                
    def future_exec(self, func, conf):
        # Use the static function directly rather than the instance method
        with concurrent.futures.ProcessPoolExecutor(1) as pool:
            future = pool.submit(_dak_exec_static, func, conf)
        return future.result()