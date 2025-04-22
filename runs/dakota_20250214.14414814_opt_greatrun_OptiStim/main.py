### This file is the access point for "make python_example"
## It will sample for a simple function - to demonstrate feasibility of the approach
## When this is uploaded to OSPARC, the same dakota.in will be executed inside DakotaService
## and model evaluations will be executed within the ParallelRunner

import pathlib as pl
import sys
import os
import json


script_dir = pl.Path(__file__).parent
sys.path.append(str(script_dir))

#

from evaluation import model


def main():
    print(list(os.environ.keys()))
    input_path = pl.Path(os.environ["INPUT_FOLDER"])
    output_path = pl.Path(os.environ["OUTPUT_FOLDER"])

    input_file_path = input_path / "input.json"
    output_file_path = output_path / "output.json"

    inputs = json.loads(input_file_path.read_text())

    outputs = [model(**input) for input in inputs]

    output_file_path.write_text(json.dumps(outputs))


if __name__ == "__main__":
    main()
