# Created by scalers.ai for Dell
"""Create train/test dataset for fine-tuning by applying prompts."""

import argparse
import json
import os
import sys

from datasets import load_dataset


def parse_args():
    """Parse command-line arguments for dataset preprocessing.

    :returns: Parsed command-line arguments.
    :rtype: argparse.Namespace

    :raises ArgumentParserError: If there is an error in parsing the arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parameters for dataset preprocessing"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="1",
        required=True,
        choices=["1", "2", "3"],
        help="Choice for the type of prompt to be used for training.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Full path of the dataset file to be used for training.",
    )

    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="The size of the test split to be generated from the dataset.",
    )

    args = parser.parse_args()

    return args


def create_prompt(data_point):
    """Format data points into a prompt.

    :param data_point: A dictionary containing question, context, and answer.
    :type data_point: dict

    :returns: A formatted prompt containing instruction, input, and response.
    :rtype: str
    """
    return f"""Below is an instruction that describes a task, paired with an input
    that provides further context. Write a response that appropriately completes
    the request.

    ### Instruction:
    {data_point["question"]}

    ### Input:
    {data_point["context"]}

    ### Response:
    {data_point["answer"]}

    """


def template_dataset(datapoint):
    """Template a dataset by formatting a data point into a prompt.

    :param datapoint: A dictionary containing data for formatting a prompt.
    :type datapoint: dict

    :returns: The modified data point with formatted text.
    :rtype: dict
    """
    datapoint["text"] = f"{create_prompt(datapoint)}"

    return datapoint


def text_prompt(qa_dataset):
    """Create prompts without special tokens.

    E.g: The dataset has an example as:
    question: Is pseudomonas aeruginosa in CF and non-CF homes found
              predominantly in drains?
    context: For patients with cystic fibrosis (CF) Pseudomonas aeruginosa
             infection is a major contributor to progressive lung disease.
    long_answer: These findings implicate drains as important potential sources
                 of P. aeruginosa infection.

    The formatted prompt will be as below:

    Below is an instruction that describes a task, paired with an input that
    provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Is pseudomonas aeruginosa in CF and non-CF homes found predominantly in drains?

    ### Input:
    For patients with cystic fibrosis (CF) Pseudomonas aeruginosa infection is a
    major contributor to progressive lung disease.

    ### Response:
    These findings implicate drains as important potential sources of P. aeruginosa infection.


    :param qa_dataset: A dictionary containing dataset splits.
    :type qa_dataset: dict
    """
    qa_dataset["train"] = qa_dataset["train"].shuffle().map(template_dataset)
    qa_dataset["test"] = qa_dataset["test"].shuffle().map(template_dataset)

    for key, datasplit in qa_dataset.items():
        try:
            with open(
                f"data/{key}.jsonl", "w", encoding="utf-8"
            ) as jsonl_file:
                try:
                    for item in datasplit:
                        newitem = {}
                        newitem["input"] = f"{item['text']}"
                        jsonl_file.write(json.dumps(newitem) + "\n")
                except Exception as error:
                    print(f"Error writing to {key}.jsonl: {error}")
        except FileNotFoundError as file_error:
            sys.exit(f"Error opening {key}.jsonl: {file_error}")


def prompt_with_tokens(qa_dataset):
    """Create prompts with special tokens.

    E.g: The dataset has an example as:
    question: Is pseudomonas aeruginosa in CF and non-CF homes found
              predominantly in drains?
    context: For patients with cystic fibrosis (CF) Pseudomonas aeruginosa
             infection is a major contributor to progressive lung disease.
    long_answer: These findings implicate drains as important potential sources
                 of P. aeruginosa infection.

    The formatted prompt will be as below:
    <START_Q>Is pseudomonas aeruginosa in CF and non-CF homes found predominantly
    in drains? For patients with cystic fibrosis (CF) Pseudomonas aeruginosa
    infection is a major contributor to progressive lung disease.<END_Q>
    <START_A> These findings implicate drains as important potential sources of
    P. aeruginosa infection.<END_A>

    :param qa_dataset: A dictionary containing dataset splits.
    :type qa_dataset: dict
    """
    try:
        with open("data/tokens.json", "w", encoding="utf-8") as json_file:
            try:
                tokens = {}
                tokens["tokens"] = [
                    "<START_Q>",
                    "<END_Q>",
                    "<START_A>",
                    "<END_A>",
                ]
                json_file.write(json.dumps(tokens))
            except Exception as error:
                print(f"Error writing to tokens.json: {error}")
    except FileNotFoundError as file_error:
        sys.exit(f"Error opening tokens.json: {file_error}")

    for key, datasplit in qa_dataset.items():
        try:
            with open(
                f"data/{key}.jsonl", "w", encoding="utf-8"
            ) as jsonl_file:
                try:
                    for item in datasplit:
                        newitem = {}
                        newitem["input"] = (
                            f"<START_Q>{item['question']} {item['context']}<END_Q>"
                            f"<START_A>{item['answer']}<END_A>"
                        )
                        jsonl_file.write(json.dumps(newitem) + "\n")
                except Exception as error:
                    sys.exit(f"Error writing to {key}.jsonl: {error}")
        except FileNotFoundError as file_error:
            sys.exit(f"Error opening {key}.jsonl: {file_error}")


def special_tokens_prompt(qa_dataset):
    """Create prompts with custom tokens.

    E.g: The dataset has an example as:
    question: Is pseudomonas aeruginosa in CF and non-CF homes
              found predominantly in drains?
    context: For patients with cystic fibrosis (CF) Pseudomonas aeruginosa
             infection is a major contributor to progressive lung disease.
    long_answer: These findings implicate drains as important potential
                 sources of P. aeruginosa infection.

    The formatted prompt will be as below:
    <START_Q>Is pseudomonas aeruginosa in CF and non-CF homes found predominantly
    in drains?<END_Q><START_C>For patients with cystic fibrosis (CF) Pseudomonas
    aeruginosa infection is a major contributor to progressive lung disease.<END_C>
    <START_A> These findings implicate drains as important potential sources of
    P. aeruginosa infection.<END_A>

    :param qa_dataset: A dictionary containing dataset splits.
    :type qa_dataset: dict
    """
    try:
        with open("data/tokens.json", "w", encoding="utf-8") as json_file:
            try:
                tokens = {}
                tokens["tokens"] = [
                    "<START_Q>",
                    "<END_Q>",
                    "<START_C>",
                    "<END_C>",
                    "<START_A>",
                    "<END_A>",
                ]
                json_file.write(json.dumps(tokens))
            except Exception as error:
                sys.exit(f"Error writing to tokens.json: {error}")
    except FileNotFoundError as file_error:
        sys.exit(f"Error opening tokens.json: {file_error}")

    for key, datasplit in qa_dataset.items():
        try:
            with open(
                f"data/{key}.jsonl", "w", encoding="utf-8"
            ) as jsonl_file:
                try:
                    for item in datasplit:
                        newitem = {}
                        newitem["input"] = (
                            f"<START_Q>{item['question']}<END_Q>"
                            f"<START_C>{item['context']}<END_C>"
                            f"<START_A>{item['answer']}<END_A>"
                        )
                        jsonl_file.write(json.dumps(newitem) + "\n")
                except Exception as error:
                    sys.exit(f"Error writing to {key}.jsonl: {error}")
        except FileNotFoundError as file_error:
            sys.exit(f"Error opening {key}.jsonl: {file_error}")


def main():
    """Main function for dataset preprocessing."""
    # Load the training configurations
    args = parse_args()
    try:
        # Load the dataset
        dataset = load_dataset("json", data_files={"data": args.dataset_path})
    except FileNotFoundError as file_error:
        sys.exit(f"Error: Unable to load dataset: {file_error}.")

    # Split the dataset according to the input ratio
    qa_dataset = dataset["data"].train_test_split(
        args.test_split, shuffle=True, seed=42
    )

    if not os.path.exists("data"):
        os.mkdir("data")

    # Input the type of prompt required
    term = args.prompt_type
    if term == "1":
        text_prompt(qa_dataset)
    elif term == "2":
        prompt_with_tokens(qa_dataset)
    elif term == "3":
        special_tokens_prompt(qa_dataset)
    else:
        text_prompt(qa_dataset)


if __name__ == "__main__":
    main()
