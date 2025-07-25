# Home to code we want to share from different scripts that perform evaluations using the language models.

from typing import List
from datasets import load_dataset
import re

def load_phrases_plaintext(filename: str,
                           phrase_limit: int = None) -> List[str]:
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
                         limit_val = None) -> List[str]:
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

def filter_phrases(phrases: List[str],
                   drop_numbers: bool = False,
                   drop_max_len: int = None) -> List[str]:
    """
    Remove entire phrases based on different criteria
    :param phrases: Original list of all the phrases
    :param drop_numbers: Drop any phrase with a character 0-9
    :param drop_max_len: Drop any phrase with more than this many characters
    :return: List of phrases
    """
    result = []
    for phrase in phrases:
        if (not drop_max_len or len(phrase) <= drop_max_len) and \
           (not drop_numbers or not re.search(r'\d', phrase)):
           result.append(phrase)
    return result

def normalize_phrases(phrases: List[str],
                      lower: bool = False,
                      strip: bool = False,
                      truncate_max_len: int = None) -> List[str]:
    """
    Perform text normalization on the phrases
    :param phrases: Original list of all the phrases
    :param lower: Lowercase all phrases
    :param strip: Converts characters besides A-Z and apostrophe to space then collapses contiguous whitespace
    :param truncate_max_len: Truncate any phrase with this many characters
    :return: List of phrases
    """
    result = []
    for phrase in phrases:
        if lower:
            phrase = phrase.lower()
        # Drop words until we meet the truncation max length
        if truncate_max_len and len(phrase) > truncate_max_len:
            # First cut it off
            phrase = phrase[:truncate_max_len]
            # Then remove characters until we reach a space
            while phrase[-1] != " ":
                phrase = phrase[:-1]
            phrase = phrase.strip()
        if strip:
            phrase = re.sub(r'[^a-zA-Z \']', ' ', phrase)
            phrase = re.sub(r'\s+', ' ', phrase).strip()
        result.append(phrase)
    return result

def count_words(phrases: List[str]) -> int:
    """
    Count the number of words in a list of phrases
    :param phrases: List of phrases
    :return: Integer count
    """
    count = 0
    for phrase in phrases:
        count += len(phrase.split())
    return count