import contextlib
import os
from pathlib import Path
import dakota.environment as dakenv
import logging
import mmux_python.utils.wiofiles as wio # type: ignore
import sys
import concurrent.futures

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
def _dak_exec_static(conf):
    """Static version of dak_exec that can be pickled for multiprocessing."""
    study = dakenv.study(callback=None, input_string=conf) # type: ignore
    stdoutstr, stderrstr = None, None
    with wio.capture_to_file(stdout='./stdout', stderr='./stderr') as (stdout, stderr):
        study.execute()
    with open(stdout) as outf, open(stderr) as errf:
        stdoutstr = outf.read()
        stderrstr = errf.read()
    del study
    return stdoutstr, stderrstr

class DakotaObject:
    def __init__(self) -> None:
        logger.info("DakotaObject created")

    def run(self, dakota_conf: str, output_dir: Path):
        print("Starting dakota")
        with working_directory(output_dir):
            # Create a picklable version of the callback
            stdout, stderr = self.future_exec(conf=dakota_conf)
            print("Dakota run finished")
            with open("dakota_stdout.txt", "w") as f_out, open("dakota_stderr.txt", "w") as f_err:
                if stdout:
                    f_out.write(stdout)
                if stderr:
                    f_err.write(stderr)
            if stderr:
                print(stderr, file=sys.stderr)
                
    def future_exec(self, conf):
        # Use the static function directly rather than the instance method
        with concurrent.futures.ProcessPoolExecutor(1) as pool:
            future = pool.submit(_dak_exec_static, conf)
        return future.result()