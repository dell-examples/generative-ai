# Created by scalers.ai for Dell
"""
This is a helper script for converting safetensors files in
model directories to bin files.

This can be used from the command line by running
python3 sft2bin.py -d /path/to/model/dir

or it can be used as a module by importing the
sft2bin_worker function and passing the /path/to/model/dir
to this function as an argument.
"""

import json
import os
import shutil
import uuid
from argparse import ArgumentParser

import torch
from safetensors.torch import load_file


def parse_args():
    """Command-line argument creator"""
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    return args


def to_bin_and_save(filename: str) -> None:
    """
    Converts a safetensors file to a bin file, and saves it in the
    same directory with the same name, with extension as .bin instead
    of .safetensors
        Parameters:
            filename: The filepath of the safetensors file
        Returns:
            None
    """

    # Check if file is a safetensors file
    if filename.split(".")[-1] != "safetensors":
        print("Non-safetensors file passed for conversion. Skipping...")
        return

    root_name = filename.split(".")[0]

    # Load safetensors file and save as bin file.
    sft = load_file(filename)
    torch.save(sft, f"{root_name}.bin")
    return


def modify_index_json(filename: str) -> None:
    """
    Parses the model.safetensors.index.json file to replace all
    occurences of .safetensors files with .bin files. Will rewrite
    the file with the modified json.
        Parameters:
            filename: The filepath to the model.safetensors.index.json file.
        Returns:
            None
    """

    # Read json file as dictionary
    index_file = None
    with open(filename, "r", encoding="utf-8") as file:
        index_file = json.loads(file.read())

    # Change all values in safetensor values in weight_map to bin
    for key in index_file["weight_map"].keys():
        value = index_file["weight_map"][key]
        if value.endswith(".safetensors"):
            value = value.split(".")[0]
            index_file["weight_map"][key] = f"{value}.bin"

    # Rewrite the modified json to same file
    with open(filename, "w") as file:
        file.seek(0)
        index_file_str = json.dumps(index_file)
        file.write(index_file_str)

    return


def sft2bin_worker(dirname: str):
    """
    This function does the end-to-end of converting all safetensors files
    to bin files, and updating the model.safetensors.index.json file as well.
        Parameters:
            dirname: The name of the directory where the safetensors files
                are located.
        Returns:
            None
    """

    # Checks to make sure we can work with dirname
    full_name = os.path.join("/", "models", dirname)
    if not os.path.exists(full_name):
        print(
            "Path doesn't exist. Assuming HuggingFace model file, and skipping..."
        )
        return dirname

    dirname = full_name
    if not os.path.isdir(dirname):
        print(f"The path {dirname} is not a directory")
        return dirname

    # Copy the current directory into /tmp under new name
    new_dir = str(uuid.uuid4())
    shutil.copytree(dirname, f"/tmp/{new_dir}")

    dirname = os.path.join("/", "tmp", new_dir)

    all_files = os.listdir(dirname)
    sft_files = []

    # Get all safetensors files and convert to bin
    for f in all_files:
        if f.endswith(".safetensors"):
            sft_files.append(f)
            to_bin_and_save(os.path.join(dirname, f))

    # Remove all safetensors files from directory
    for f in sft_files:
        os.remove(os.path.join(dirname, f))

    # Modify the .safetensors.index.json file
    for f in all_files:
        if f.endswith(".safetensors.index.json"):
            modify_index_json(os.path.join(dirname, f))

    return dirname


if __name__ == "__main__":
    args = parse_args()
    sft2bin_worker(args.dir)
