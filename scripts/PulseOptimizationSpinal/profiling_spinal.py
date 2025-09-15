from pathlib import Path
import sys, shutil
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from tests.test_utils.test_funs_git import create_run_dir
from utils.funs_create_dakota_conf import create_function_sampling
from utils.funs_data_processing import load_data, get_non_dominated_indices
from utils.dakota_object import DakotaObject, Map
from utils.funs_plotting import plot_objective_space
from utils_spinal import get_model_from_spinal_repo, postpro_spinal_samples


MAXAMP = 1.0
NUM_SAMPLES = 1
N_RUNNERS = 1

run_dir = create_run_dir(Path.cwd(), "sampling")
model = get_model_from_spinal_repo(run_dir)
shutil.copytree("GAF_kernels", run_dir / "GAF_kernels")
outputs = model(
    **{f"p{i+1}": r for i, r in enumerate(np.random.rand(15).tolist())}
)  ## run the model itsself

print(f"Outputs: {outputs}")
