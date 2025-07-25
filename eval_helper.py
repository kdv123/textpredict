# Home to code we want to share from different scripts that perform evaluations using the language models.

from typing import List
from datasets import load_dataset

def load_phrases_plaintext(filename: str,
                           phrase_limit: int = None) -> List:
    """
    Load phrases from a plaintext file with a phrase on each line
    :param filename: Filename containing the phrases
    :param phrase_limit: Optional limit to the first so many lines
    :return: List of phrases
    """
    # Read in a plain text file that has a sentence on each line
    phrase_file = open(filename, "r")
    phrases = phrase_file.readlines()
    phrase_file.close()

    if phrase_limit:
        phrases = phrases[:phrase_limit]
    return phrases

def load_phrases_dataset(name: str,
                         split: str,
                         phrase_col: str = "text",
                         phrase_limit: int = None,
                         limit_col = None,
                         limit_val = None) -> List:
    """
    Load phrases from a dataset
    :param name: Name of the dataset
    :param split: Name of the split in the dataset, e.g. train or validation
    :param phrase_col: Name of the column in the dataset containing the phrases
    :param phrase_limit: Optional limit to the first so many rows
    :param limit_col: Optional column to use to select subset of rows
    :param limit_val: Value to use to select subset of rows
    :return: List of phrases
    """
    # Read the sentences from a Hugging Face dataset
    dataset = load_dataset(path=name, split=split)
    # Optional limiting to rows with a column matching a given value
    if limit_col:
        dataset = dataset.filter(
            function=lambda example:
            example[limit_col] == limit_val)
    phrases = dataset[phrase_col]

    if phrase_limit:
        phrases = phrases[:phrase_limit]
    return phrases