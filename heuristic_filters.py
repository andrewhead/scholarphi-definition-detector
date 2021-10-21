"""
^O Replacing occurrences of NN NN and NN NN NN with NN and vice versa.
^O Replacing occurrences of JJ NN with NN and vice versa.
^O Replace occurrences of 'the TARGET', 'a TARGET' and 'an TARGET' with
TARGET, and vice versa.
^O Replacing occurrences of 'the NN', 'a NN' and 'an NN' with NN, and vice
versa.
^O Replacing occurrences of JJ JJ with JJ and vice versa. (For example, a
wonderful beautiful place, a wonderful place)
^O Replacing NNS with NN and vice versa, where NNS is the tag for plural
nouns.
^O Replacing occurrences of 'the' with 'a' and vice versa.
^O Replacing occurrences of NN with NP and vice versa, where NP is the tag
for Proper Nouns.
^O Replacing occurrences of NP NP with NP and vice versa.
^O Replacing occurrences of RB VV with VV, where RB is the tag for Adverbs.
(For example: TARGET refers to e^Nciently studying for 5 hours, TARGET
refers to studying for 5 hours.)
"""

import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def common_filters(
    intent_preds: List[int],
    slot_preds: List[List[str]]
) -> Tuple[List[int], List[List[str]]]:
    """
    Apply common filters for the predictions
    """
    new_intent_preds, new_slot_preds = [], []

    for intent_pred, slot_pred, in zip(intent_preds, slot_preds):
        new_slot_pred = slot_pred
        new_intent_pred = intent_pred

        # 1. [slot] Filter out term / definition only cases.
        pred_counter = dict(Counter(slot_pred))
        term_exist, def_exist = False, False
        for c in pred_counter:
            if c.endswith("TERM"):
                term_exist = True
            if c.endswith("DEF"):
                def_exist = True
        if not (term_exist and def_exist):
            new_slot_pred = ["O" for p in slot_pred]

        # 2. [intent] Change intent label if no term + def detected.
        if not(term_exist and def_exist):
            new_intent_pred = 0

        # 3. [slot] Replace UNK with O.
        new_slot_pred = ["O" if sp == "UNK" else sp for sp in new_slot_pred]

        # 4. Change I-TERM I-DEF starting cases.
        temp_new_slot_pred = new_slot_pred.copy()
        term_start, def_start = False, False
        for sid, sp in enumerate(temp_new_slot_pred):
            if not term_start and sp == "I-TERM":
                new_slot_pred[sid] = "B-TERM"
            if sp.endswith("TERM"):
                term_start = True
            else:
                term_start = False

            if not def_start and sp == "I-DEF":
                new_slot_pred[sid] = "B-DEF"
            if sp.endswith("DEF"):
                def_start = True
            else:
                def_start = False

        new_intent_preds.append(new_intent_pred)
        new_slot_preds.append(new_slot_pred)

    return new_intent_preds, new_slot_preds

def term_def_filters(
    intent_preds: List[int],
    slot_preds : List[List[str]],
) -> Tuple[List[int], List[List[str]]]:
    new_intent_preds, new_slot_preds = [], []

    for intent_pred, slot_pred in zip(intent_preds, slot_preds):
        new_slot_pred = slot_pred
        new_intent_pred = intent_pred
        # [slot] Fill out missing term/def within threshold.
        temp_new_slot_pred = new_slot_pred.copy()
        for sid, sp in enumerate(temp_new_slot_pred):
            if sid < len(new_slot_pred) - 2 and sp.endswith("TERM"):
                if temp_new_slot_pred[sid + 1] == "O" and temp_new_slot_pred[
                    sid + 2
                ].endswith("TERM"):
                    new_slot_pred[sid + 1] = "I-TERM"
        temp_new_slot_pred = new_slot_pred.copy()
        for sid, sp in enumerate(temp_new_slot_pred):
            if sid < len(new_slot_pred) - 2 and sp.endswith("DEF"):
                if temp_new_slot_pred[sid + 1] == "O" and temp_new_slot_pred[
                    sid + 2
                ].endswith("DEF"):
                    new_slot_pred[sid + 1] = "I-DEF"
        temp_new_slot_pred = new_slot_pred.copy()
        for sid, sp in enumerate(temp_new_slot_pred):
            if sid < len(new_slot_pred) - 3 and sp.endswith("DEF"):
                if (
                    temp_new_slot_pred[sid + 1] == "O"
                    and temp_new_slot_pred[sid + 2] == "O"
                    and temp_new_slot_pred[sid + 3].endswith("DEF")
                ):
                    new_slot_pred[sid + 1] = "I-DEF"
                    new_slot_pred[sid + 2] = "I-DEF"

        new_intent_preds.append(new_intent_pred)
        new_slot_preds.append(new_slot_pred)
    return new_intent_preds, new_slot_preds

def sym_nick_filters(
    intent_preds: List[int],
    slot_preds : List[List[str]],
    raw: List[List[str]]
) -> Tuple[List[int], List[List[str]]]:
    new_intent_preds, new_slot_preds = [], []

    for intent_pred, slot_pred, raw_data in zip(intent_preds, slot_preds, raw):
        new_slot_pred = slot_pred
        new_intent_pred = intent_pred
        #1. [slot] Replace mis-labelled non SYMBOL as TERM
        temp_new_slot_pred = new_slot_pred.copy()
        for sid, sp in enumerate(temp_new_slot_pred):
            if sp.endswith("TERM") and 'SYMBOL' not in raw_data[sid]:
                new_slot_pred[sid] = 'O'

        # 2. Change TERMs in between DEFs.
        temp_new_slot_pred = new_slot_pred.copy()
        term_start, def_start = False, False
        for sid, sp in enumerate(temp_new_slot_pred):
            if sp.endswith("DEF"):
                def_start = True
            else:
                def_start = False

            if sp.endswith("TERM"):
                if def_start:
                    new_slot_pred[sid] = 'I-DEF'
                    def_start = True
                else:
                    term_start = True
            else:
                term_start = False

        # Remove intent in case a term was removed and there are none left
        pred_counter = dict(Counter(slot_pred))
        term_exist, def_exist = False, False
        for c in pred_counter:
            if c.endswith("TERM"):
                term_exist = True
            if c.endswith("DEF"):
                def_exist = True
        if not (term_exist and def_exist):
            new_slot_pred = ["O" for p in slot_pred]

        #[intent] Change intent label if no term + def detected.
        if not(term_exist and def_exist):
            new_intent_pred = 0


        new_intent_preds.append(new_intent_pred)
        new_slot_preds.append(new_slot_pred)
    return new_intent_preds, new_slot_preds

def sym_nick_query_filters(
    intent_preds: List[int],
    slot_preds : List[List[str]],
    raw: List[List[str]],
    raw_processed : List[List[str]],
    query_string: str = '||||',
    symbol_length_threshold: int = 30
) -> Tuple[List[int], List[List[str]]]:
    new_intent_preds, new_slot_preds = [], []

    for intent_pred, slot_pred, raw_data, raw_processed_data in zip(intent_preds, slot_preds, raw, raw_processed):
        new_slot_pred = slot_pred
        new_intent_pred = intent_pred
        #1. [slot] Replace mis-labelled non SYMBOL as TERM
        temp_new_slot_pred = new_slot_pred.copy()
        for sid, sp in enumerate(temp_new_slot_pred):
            if sp.endswith("TERM") and 'SYMBOL' not in raw_data[sid]:
                new_slot_pred[sid] = 'O'

        #1b. [slot] Replace SYMBOL outside the query with O
        temp_new_slot_pred = new_slot_pred.copy()
        for sid, sp in enumerate(temp_new_slot_pred):
            if sp.endswith("TERM") and 'SYMBOL' in raw_data[sid] and (sid>0 or sid<len(temp_new_slot_pred)-1):
                if raw_data[sid-1]!= query_string and raw_data[sid+1] != query_string:
                    new_slot_pred[sid] = 'O'
            elif sp.endswith("TERM") and 'SYMBOL' in raw_data[sid] and (sid==0 or sid==len(temp_new_slot_pred)-1):
                new_slot_pred[sid] = 'O'

        #1c. [slot] Replace SYMBOL with only numbers or symbols
        temp_new_slot_pred = new_slot_pred.copy()
        for sid, sp in enumerate(temp_new_slot_pred):
            if sp.endswith("TERM") and 'SYMBOL' in raw_data[sid]:
                symbol_without_num_special_chars = [x for x in raw_processed_data[sid] if x.isalpha()]
                if len(symbol_without_num_special_chars)==0:
                    new_slot_pred[sid] = 'O'

        #1d. [slot] Replace SYMBOL that's really long
        temp_new_slot_pred = new_slot_pred.copy()
        for sid, sp in enumerate(temp_new_slot_pred):
            if sp.endswith("TERM") and 'SYMBOL' in raw_data[sid]:
                if len(raw_processed_data[sid]) > symbol_length_threshold:
                    new_slot_pred[sid] = 'O'

        # 2. Change TERMs in between DEFs.
        temp_new_slot_pred = new_slot_pred.copy()
        term_start, def_start = False, False
        for sid, sp in enumerate(temp_new_slot_pred):
            if sp.endswith("DEF"):
                def_start = True
            else:
                def_start = False

            if sp.endswith("TERM"):
                if def_start:
                    new_slot_pred[sid] = 'I-DEF'
                    def_start = True
                else:
                    term_start = True
            else:
                term_start = False

        # Remove intent in case a term was removed and there are none left
        pred_counter = dict(Counter(slot_pred))
        term_exist, def_exist = False, False
        for c in pred_counter:
            if c.endswith("TERM"):
                term_exist = True
            if c.endswith("DEF"):
                def_exist = True
        if not (term_exist and def_exist):
            new_slot_pred = ["O" for p in slot_pred]

        #[intent] Change intent label if no term + def detected.
        if not(term_exist and def_exist):
            new_intent_pred = 0


        new_intent_preds.append(new_intent_pred)
        new_slot_preds.append(new_slot_pred)
    return new_intent_preds, new_slot_preds

def abbr_exp_filters(
    intent_preds: List[int],
    slot_preds : List[List[str]],
) -> Tuple[List[int], List[List[str]]]:
    return intent_preds, slot_preds

def heuristic_filters(
    intent_preds: Dict[str,List[int]],
    slot_preds: Dict[str,List[List[str]]],
    raw: List[List[str]],
    task : str,
    raw_processed: List[List[str]],
) -> Tuple[Dict[str,List[int]], Dict[str,List[List[str]]]]:
    """
    Apply various heuristic filters based on the data type [AI2020(abbr-exp), DocDef2(sym-nick), W00(term-def)]
    """

    data_types = task.split('+')
    for data_type in data_types:
        if data_type=='AI2020':
            simplified_slot_preds = []
            for slot_pred in slot_preds[data_type]:
                simplified_slot_pred = []
                for s in slot_pred:
                    simplified_slot_pred.append(s.replace('long','DEF').replace('short','TERM'))
                simplified_slot_preds.append(simplified_slot_pred)
            slot_preds[data_type] = simplified_slot_preds

        intent_preds[data_type], slot_preds[data_type] = common_filters(intent_preds[data_type], slot_preds[data_type])

        if data_type == 'AI2020':
            intent_preds[data_type], slot_preds[data_type] = abbr_exp_filters(intent_preds[data_type], slot_preds[data_type])
        elif data_type =='DocDef2':
            intent_preds[data_type], slot_preds[data_type] = sym_nick_filters(intent_preds[data_type], slot_preds[data_type], raw)
        elif data_type.startswith('DocDefQueryInplace2'):
            intent_preds[data_type], slot_preds[data_type] = sym_nick_query_filters(intent_preds[data_type], slot_preds[data_type], raw, raw_processed)
        elif data_type == 'W00':
            intent_preds[data_type], slot_preds[data_type] = term_def_filters(intent_preds[data_type], slot_preds[data_type])

    return intent_preds, slot_preds
