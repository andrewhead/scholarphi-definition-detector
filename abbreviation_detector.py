import os
import sys
import shutil
import regex
import coloredlogs #, logging
import numpy
import torch
from pprint import pprint
from colorama import Fore,Style
from tqdm import tqdm, trange
from collections import Counter,defaultdict
from typing import Any, List, Dict, Tuple, Optional, DefaultDict, Union
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, logging
import spacy
from scispacy.abbreviation import AbbreviationDetector
from spacy.matcher import Matcher
from spacy.util import filter_spans
from dataclasses import dataclass

from utils import get_token_character_ranges,sanity_check_for_acronym_detection, mapping_feature_tokens_to_original

logger = logging.get_logger()


def get_abbreviation_pairs(
    input_tokens,
    nlp_model):

    # Extract an acronym as a term and expansion as a definition using abbreviation detector. E.g., [Term]: Expected Gradients (EG) -> [Term]: EG, [Definition]: Expected Gradients

    text = " ".join(input_tokens)
    doc = nlp_model.nlp(text)
    abbreviation_tokens = [str(t) for t in doc]

    # Make index from tokens to their character positions.
    abbreviation_ranges = get_token_character_ranges(text, input_tokens)

    predictions = []
    for abrv in doc._.abbreviations:
        # acronym (term).
        acronym_ranges = abbreviation_ranges[abrv.start:abrv.end]
        if len(acronym_ranges) == 0:
            continue
        acronym_start = min([r.start for r in acronym_ranges])
        acronym_end = max([r.end for r in acronym_ranges]) + 1

        # expansion (definition).
        expansion_ranges = abbreviation_ranges[abrv._.long_form.start:abrv._.long_form.end]
        if len(expansion_ranges) == 0:
            continue
        expansion_start = min([r.start for r in expansion_ranges])
        expansion_end = max([r.end for r in expansion_ranges]) + 1

        if not sanity_check_for_acronym_detection(
                str(abrv),
                str(abrv._.long_form),
                acronym_start, acronym_end,
                expansion_start, expansion_end):
            continue

        _, new_labels = mapping_feature_tokens_to_original(
            abbreviation_tokens,
            abbreviation_ranges,
            expansion_start, expansion_end,
            acronym_start, acronym_end
        )

        predictions.append(new_labels)

    return predictions








