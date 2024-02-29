# Created by scalers.ai for Dell
"""
This script is used for converting a Llama 2 model to ONNX format. The converted
model can then be used for inference with frameworks supporting ONNX models.

Additionally, it cleans up unnecessary files generated during the conversion process
and saves the config file and tokenizer in the ONNX model directory.
"""

import argparse
import glob
import os

from huggingface_hub import HfFolder
from transformers import AutoConfig, AutoTokenizer


def parse_arguments():
    """Parse command-line arguments for converting Llama 2 model to ONNX format.

    :returns: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Convert Llama 2 model to ONNX format"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Llama 2 model variant name",
    )
    parser.add_argument(
        "--hf_token", type=str, required=True, help="Hugging Face Token"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="int8",
        help="Data type for model precision",
    )

    return parser.parse_args()


def main():
    """Main function for converting Llama 2 model to ONNX format."""
    args = parse_arguments()

    save_directory = ""

    if "/" in args.model_name:
        # Extract model name from the URL for creating the save directory's name
        filename = args.model_name.split("/")
        save_directory = f"onnx_{filename[1]}-{args.data_type}"
    else:
        save_directory = f"onnx_{args.model_name}-{args.data_type}"

    # Save Hugging Face token
    HfFolder.save_token(os.environ.get("HF_TOKEN", args.hf_token))

    # Run the command to convert the model to ONNX format
    command = (
        f"python3 -m onnxruntime.transformers.models.llama.convert_to_onnx"
        f" -m {args.model_name}"
        f" --output {save_directory} --precision {args.data_type}"
        " --quantization_method quantize_dynamic --execution_provider cpu"
    )
    os.system(command)

    # Clean up unnecessary files generated during conversion
    files_to_delete = glob.glob(os.path.join(f"{save_directory}", "*fp32*"))
    for file in files_to_delete:
        os.remove(file)

    # Rename the ONNX file to a standard name
    onnx_files = glob.glob(os.path.join(f"{save_directory}", "*.onnx"))
    for old_filename in onnx_files:
        new_name = os.path.join(save_directory, "model.onnx")
        os.rename(old_filename, new_name)

    # Save config file in ONNX model directory
    config = AutoConfig.from_pretrained(args.model_name)
    config.save_pretrained(save_directory)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(save_directory)


if __name__ == "__main__":
    main()
