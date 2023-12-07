# Created by scalers.ai for Dell
"""
Preprocess and format the input PubMed QA dataset from huggingface.

The dataset is converted to the format
`{"question": ..., "context": ..., "answer": ...}`
"""

import os
import sys

from datasets import load_dataset


def formatting_func(datapoint):
    """Convert the datatype from sequence to string.

    :param datapoint: A dictionary containing data to be formatted.
    :type datapoint: dict

    :returns: The modified data point with 'context' having string datatype.
    :rtype: dict
    """
    datapoint["context"] = "".join(datapoint["context"])

    return datapoint


def main():
    """Main function for dataset preprocessing and conversion to JSON.

    This function performs the following steps:
    1. Load the dataset with the subset.
    2. Flatten the dataset.
    3. Remove unnecessary columns.
    4. Rename the columns according to the format required.
    5. Convert the datatype of the column from sequence to string datatype.
    6. Save the dataset as a JSON file.

    :returns: None
    """
    # Load the dataset
    try:
        dataset = load_dataset("pubmed_qa", "pqa_artificial")
    except Exception as error:
        sys.exit(f"An error occurred: {error}")

    # Flatten the dataset
    dataset = dataset.flatten()
    # Remove the unnecessary columns
    dataset = dataset.remove_columns(
        ["pubid", "context.labels", "context.meshes", "final_decision"]
    )
    # Rename the columns as per requirement
    dataset = dataset.rename_column("context.contexts", "context")
    dataset = dataset.rename_column("long_answer", "answer")
    # Convert the datatype of column 'context' from sequence to string
    dataset = dataset.map(formatting_func)

    # Create a folder to store the preprocessed json file
    if not os.path.exists("dataset_dir"):
        os.mkdir("dataset_dir")

    # Dump the dataset into a json file
    try:
        dataset["train"].to_json("dataset_dir/dataset.json")
    except FileNotFoundError as file_error:
        sys.exit(f"File not found: {file_error}")
    except PermissionError as permission_error:
        sys.exit(f"Permission error: {permission_error}")
    except Exception as error:
        sys.exit(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
