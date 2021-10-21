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

from utils import get_token_character_ranges,mapping_feature_tokens_to_original
logger = logging.get_logger()


@dataclass(frozen=True)
class TermDefinitionPair:
    term_start: int
    term_end: int
    term_text: str
    term_type: str
    term_confidence: float
    definition_start: int
    definition_end: int
    definition_text: str
    definition_type: str
    definition_confidence: float



def search_symbol_nickname(tidx, pos_list, range_, ranges, direction):
    UNION_POS_SET = ["DT", "JJ", "NN", "NNS", "NNP", "NNPS"]
    pos_idxs = []
    pos_tags = []
    pos_ranges = []
    if direction=='RIGHT':
        current_idx = tidx + 1

    elif direction=='LEFT':
        current_idx = tidx - 1


    # Exit if the current id is greater than the length of the sentence or below zero
    if current_idx >= len(pos_list) or current_idx<0:
        return None, None, None, None

    # Search ahead until the POS tag is not in the union set
    while (0<=current_idx and current_idx < len(pos_list)) and pos_list[current_idx] in UNION_POS_SET:
        pos_idxs.append(current_idx)
        pos_ranges.append(ranges[current_idx])
        pos_tags.append(pos_list[current_idx])
        if direction=='RIGHT':
            current_idx += 1
        elif direction=='LEFT':
            current_idx -= 1

    # ignore when the POS tag is [DT] or [IN].
    if len(pos_tags) == 1 and (pos_tags[0] == "DT"):
        return None, None, None, None

    if len(pos_idxs) > 0 :
        symbol_idx = tidx
        symbol_range = range_

        if direction=='LEFT':
            nickname_idxs = list(reversed(pos_idxs))
            nickname_ranges = [ppr for ppr in reversed(pos_ranges)]
            nickname_tags = [ppt for ppt in reversed(pos_tags)]

        elif direction=='RIGHT':
            nickname_idxs = pos_idxs
            nickname_ranges = [npr for npr in pos_ranges]
            nickname_tags = [ppt for ppt in pos_tags]

        #Skip 'DT' or 'IN' as first or last nickname tokens
        if (nickname_tags[0] == "DT"):
            nickname_ranges = nickname_ranges[1:]
            nickname_idxs = nickname_idxs[1:]
        elif (nickname_tags[-1] == "DT"):
            nickname_ranges = nickname_ranges[:-1]
            nickname_idxs = nickname_idxs[:-1]

        return symbol_idx,symbol_range,nickname_idxs, nickname_ranges
    else:
        return None, None, None, None


def get_symbol_nickname_pairs(input_tokens, pos_list): #, symbol_texs : Dict[Any, Any]):
    # Check whether a symbol's definition is nickname of the symbol or not
    # using heuristic rules below, although they are not pefect for some cases.
    #  a/DT particular/JJ transcript/NN SYMBOL/NN
    #  the/DT self-interaction/JJ term/JJ SYMBOL/NN
    #  the/DT main/JJ effect/NN of/NN feature/JJ SYMBOL/JJ

    # Union set of POS tags in nicknames. Feel free to add more if you have new patterns
    text = " ".join(input_tokens)
    ranges = get_token_character_ranges(text, input_tokens)
    predictions = []
    if 'SYMBOL' in text:
        symbol_nickname_pairs = []
        for tidx, (token,pos,range_) in enumerate(zip(input_tokens, pos_list, ranges)):
            # 1. If of the form '*th', check RIGHT pf symbol
            # 2. If token is a symbol, then:
            #   a. If the symbol tex is present:
            #       i. If single length symbol, first check LEFT then RIGHT
            #       ii. If multi length symbol, check LEFT
            #   b. If symbol tex is not present, just check LEFT
            if (token == 'SYMBOLth'):
                symbol_idx,symbol_range,nickname_idxs, nickname_ranges = search_symbol_nickname(tidx, pos_list, range_, ranges, 'RIGHT')
                if symbol_idx != None:
                    symbol_nickname_pairs.append((symbol_idx,symbol_range,nickname_idxs, nickname_ranges))
            elif token == 'SYMBOL':
                # Decide the order of LEFT or RIGHT
                DIRS = []
                if tidx > 0:
                    if tidx < len(pos_list)-1:
                        if pos_list[tidx + 1] in ['NN']:
                            DIRS = ['RIGHT','LEFT']
                        else:
                            DIRS = ['LEFT','RIGHT']
                    else:
                        DIRS = ['LEFT']
                elif tidx < len(pos_list)-1:
                    DIRS = ['RIGHT']

                for direction in DIRS:
                    symbol_idx,symbol_range,nickname_idxs, nickname_ranges = search_symbol_nickname(tidx, pos_list , range_, ranges, direction)
                    if symbol_idx is not None:
                        symbol_nickname_pairs.append((symbol_idx,symbol_range,nickname_idxs,nickname_ranges))
                        break

        for symbol_idx,symbol_range, nickname_idxs, nickname_ranges in symbol_nickname_pairs:
            # logging.debug("Detected nicknames", symbol_idx, symbol_range, [featurized_text['tokens'][idx] for idx in nickname_idxs], nickname_ranges)

            symbol_start = symbol_range.start
            symbol_end = symbol_range.end+1
            nickname_start = min([r.start for r in nickname_ranges])
            nickname_end = max([r.end for r in nickname_ranges]) + 1

            # revert back the detected acronyms/nicknames to the original slot ensemble_predictions
            _, new_labels = mapping_feature_tokens_to_original(
                input_tokens,
                ranges,
                symbol_start, symbol_end,
                nickname_start, nickname_end,
            )
            # predictions.append({'type':'nickname','input':p['input_tokens'], 'pred':new_labels})
            predictions.append(new_labels)

    return predictions



