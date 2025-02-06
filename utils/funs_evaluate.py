from pathlib import Path
import datetime
import os
from typing import List
from utils.dakota_object import DakotaObject
from utils.funs_create_dakota_conf import create_sumo_evaluation
from utils.funs_plotting import plot_response_curves
from utils.funs_data_processing import (
    create_samples_along_axes,
    extract_predictions_along_axes,
)
import pandas as pd


def create_run_dir(script_dir: Path, dir_name: str = "sampling"):
    ## part 1 - setup
    main_runs_dir = script_dir / "runs"
    current_time = datetime.datetime.now().strftime("%Y%m%d.%H%M%S%d")
    temp_dir = main_runs_dir / "_".join(["dakota", current_time, dir_name])
    print(str(temp_dir))
    os.makedirs(temp_dir, exist_ok=True)
    print("temp_dir: ", temp_dir)
    return temp_dir


def evaluate_sumo_along_axes(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: List[str],
    response_vars: List[str],
    NSAMPLESPERVAR: int = 21,
    ## TODO be able to load / query SuMo directly; or simply be able to do on any function (although prob better as separate function, that)
):
    # create sweeps data
    data = pd.read_csv(PROCESSED_TRAINING_FILE, sep=" ")
    PROCESSED_SWEEP_INPUT_FILE = create_samples_along_axes(
        run_dir, data, input_vars, NSAMPLESPERVAR
    )

    # create dakota file
    dakota_conf = create_sumo_evaluation(
        build_file=PROCESSED_TRAINING_FILE,
        ### TODO be able to save & load surrogate models (start w GP) rather than create them every time
        samples_file=PROCESSED_SWEEP_INPUT_FILE,
        input_variables=input_vars,
        output_responses=response_vars,
    )

    # run dakota
    dakobj = DakotaObject(
        map_object=None
    )  # no need to evaluate any function (only the SuMo, internal to Dakota)
    dakobj.run(dakota_conf, run_dir)

    for RESPONSE in response_vars:
        results = extract_predictions_along_axes(
            run_dir, RESPONSE, input_vars, NSAMPLESPERVAR
        )
        plot_response_curves(results, RESPONSE, input_vars, savedir=run_dir)

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

    """ Parallel Runner code

            input_batches = self.batch_input_tasks(input_tasks, n_of_batches)
            def batch_input_tasks(self, input_tasks, n_of_batches):
                batches = [{"batch_i": None, "tasks": []} for _ in range(n_of_batches)]
                for task_i, input_task in enumerate(input_tasks):
                    batch_id = task_i % n_of_batches
                    batches[batch_id]["batch_i"] = batch_id
                    batches[batch_id]["tasks"].append((task_i, input_task))
            return batches
            output_tasks = input_tasks.copy()
            for output_task in output_tasks:
                output_task["status"] = "SUBMITTED"
            output_tasks_content = json.dumps(
                {"uuid": tasks_uuid, "tasks": output_tasks}
            )
            self.output_tasks_path.write_text(output_tasks_content)
            output_batches = self.run_batches(
                tasks_uuid, input_batches, number_of_workers
            )
            for output_batch in output_batches:
                output_batch_tasks = output_batch["tasks"]

                for output_task_i, output_task in output_batch_tasks:
                    output_tasks[output_task_i] = output_task
                    # logging.info(output_task["status"])

                output_tasks_content = json.dumps(
                    {"uuid": tasks_uuid, "tasks": output_tasks}
                )
                self.output_tasks_path.write_text(output_tasks_content)
                logger.info(f"Finished a batch of {len(output_batch_tasks)} tasks")
            logger.info(f"Finished a set of {len(output_tasks)} tasks")
            logger.debug(f"Finished a set of tasks: {output_tasks_content}")



        def run_batches(self, tasks_uuid, input_batches, number_of_workers):
            logger.info(f"Evaluating {len(input_batches)} batches")
            logger.debug(f"Evaluating: {input_batches}")

            self.n_of_finished_batches = 0

            def map_func(batch_with_uuid, trial_number=1):
                return asyncio.run(async_map_func(batch_with_uuid, trial_number))

            def set_batch_status(batch, message):
                for task_i, task in batch["tasks"]:
                    task["status"] = "FAILURE"

            async def async_map_func(batch_with_uuid, trial_number=1):
                batch_uuid, batch = batch_with_uuid
                try:
                    logger.info(
                        "Running worker for a batch of "
                        f"{len(batch["tasks"])} tasks"
                    )
                    logger.debug(f"Running worker for batch: {batch}")
                    self.jobs_file_write_status_change(
                        id=batch_uuid,
                        status="running",
                    )

                    task_input = self.transform_batch_to_task_input(batch)

                    job_timeout = (
                        self.settings.job_timeout
                        if self.settings.job_timeout > 0
                        else None
                    )

                    output_batch = await asyncio.wait_for(
                        self.run_job(task_input, batch), timeout=job_timeout
                    )

                    self.jobs_file_write_status_change(
                        id=batch_uuid,
                        status="done",
                    )

                    self.n_of_finished_batches += 1
                    logger.info(
                        "Worker has finished batch "
                        f"{self.n_of_finished_batches} of {len(input_batches)}"
                    )

        async def run_job(self, task_input, input_batch):
            job_inputs = self.create_job_inputs(task_input)

            logger.debug(f"Sending inputs: {job_inputs}")
            if self.test_mode:
                import datetime

                print(f"Start run job: {datetime.datetime.now()}")
                logger.info("Map in test mode, just returning input")

                done_batch = self.process_job_outputs(
                    job_inputs, input_batch, "SUCCESS"
                )
                time.sleep(1)
                print(f"Stop run job: {datetime.datetime.now()}")

                return done_batch

            with self.create_study_job(
                self.settings.template_id, job_inputs, self.studies_api
            ) as job:
                logger.info(f"Calling start study api for job {job.id}")
                with self.lock:
    ##############################################################################
                    job_status = self.studies_api.start_study_job(
                        study_id=self.settings.template_id, job_id=job.id
                    )
    ##############################################################################
                logger.info(f"Start study api for job {job.id} done")

                while job_status.stopped_at is None:
                    job_status = self.studies_api.inspect_study_job(
                        study_id=self.settings.template_id, job_id=job.id
                    )
                    time.sleep(1)

                if job_status.state != "SUCCESS":
                    logger.error(
                        f"Batch failed with {job_status.state}: " f"{job_inputs}"
                    )
                    raise Exception(
                        f"Job returned a failed status: {job_status.state}"
                    )
                else:
                    with self.lock:
                        job_outputs = self.studies_api.get_study_job_outputs(
                            study_id=self.settings.template_id, job_id=job.id
                        ).results

                done_batch = self.process_job_outputs(
                    job_outputs, input_batch, job_status.state
                )

            return done_batch

    """


if __name__ == "__main__":
    print("DONE")
