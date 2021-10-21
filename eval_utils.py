import os
import json
import random
import logging
from pprint import pprint
from colorama import Fore,Style
from collections import defaultdict, Counter
from typing import Any, List, Dict, Tuple, Optional, DefaultDict, Union
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from dataclasses import dataclass

def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)

    intent_preds = np.array(intent_preds)
    intent_labels = np.array(intent_labels)
    slot_preds = np.array(slot_preds)
    slot_labels = np.array(slot_labels)

    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)


    # new metrics added following Dan's suggestion
    slot_simple_result = get_slot_simple_metrics(slot_preds, slot_labels)
    partial_match_result = get_partial_match_metrics(slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)
    results.update(slot_simple_result)
    results.update(partial_match_result)

    return results



def compute_metrics_for_sentence_classification(intent_preds, intent_labels, type_=None):
    assert len(intent_preds) == len(intent_labels)
    results = {}
    intent_preds = np.array(intent_preds)
    intent_labels = np.array(intent_labels)
    intent_result = get_intent_acc(intent_preds, intent_labels, type_)
    results.update(intent_result)
    return results


def compute_metrics_for_slot_tagging(slot_preds, slot_labels, type_=None):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_preds = np.array(slot_preds)
    slot_labels = np.array(slot_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels, type_)
    slot_simple_result = get_slot_simple_metrics(slot_preds, slot_labels, type_)
    partial_match_result = get_partial_match_metrics(slot_preds, slot_labels, type_)
    results.update(slot_result)
    results.update(slot_simple_result)
    results.update(partial_match_result)
    return results

def simplify_tokens(preds, type_=None):
    simple_preds = []
    if type_ is None:
        for p in preds:
            if p.endswith('TERM'):
                simple_preds.append('TERM')
            elif p.endswith('DEF'):
                simple_preds.append('DEF')
            else:
                simple_preds.append(p)
    elif type_ in ['term', 'symbol']:
        for p in preds:
            if p.endswith('TERM'):
                simple_preds.append('TERM')
            elif p.endswith('DEF'):
                simple_preds.append('DEF')
            else:
                simple_preds.append(p)
    elif type_=='abbreviation':
        for p in preds:
            if p.endswith('short'):
                simple_preds.append('ABBR')
            elif p.endswith('long'):
                simple_preds.append('EXP')
            else:
                simple_preds.append(p)

    return simple_preds


def get_partial_match_metrics(preds, labels, type_=None):
    """
    As I understand your heuristics, once they are employed the system only predicts pairs of terms with definitions. I’d like to know the following statistics for such predicted pairs…. Suppose there are N such pairs in the gold data and the system predicts M such pairs. Say a ‘partial match’ happens when the system predicts a pair <term,defn> and there is some overlap (at least one token) between the predicted and gold term spans AND there is some overlap between the predicted and gold definition spans. Let X be the number of partial matches. What are
    Partial match precision = P/M
    Partial match recall = P/N
    """
    assert len(preds) == len(labels)

    both_in_preds, both_in_labels = [], []
    partial_matches, exact_matches = [], []
    for pred_sent, label_sent in zip(preds, labels):
        simple_pred_sent = simplify_tokens(pred_sent, type_)
        simple_label_sent = simplify_tokens(label_sent, type_)

        if type_ is None or type_ in ["term", "symbol"] :
            # check whether term/def exist together
            both_in_pred = True if ('TERM' in simple_pred_sent and 'DEF' in simple_pred_sent) else False
            both_in_label = True if ('TERM' in simple_label_sent and 'DEF' in simple_label_sent) else False
        elif type_=='abbreviation':
            # check whether term/def exist together
            both_in_pred = True if ('ABBR' in simple_pred_sent and 'EXP' in simple_pred_sent) else False
            both_in_label = True if ('ABBR' in simple_label_sent and 'EXP' in simple_label_sent) else False
        else:
            print("Wrong type:", type_)
            sys.exit(1)

        both_in_preds.append(both_in_pred)
        both_in_labels.append(both_in_label)

#         print(both_in_pred, both_in_label)
        # print(simple_pred_sent)
        # print(simple_label_sent)

        partial_match = False
        exact_match = False
        match = []
        if both_in_pred and both_in_label:
            for p,l in zip(simple_pred_sent, simple_label_sent):
                if p==l:
                    match.append(p)
                else:
                    match.append(False)
            if type_ is None or type_ in ['term', 'symbol']:
                if 'TERM' in match and 'DEF' in match:
                    partial_match = True
            elif type_=='abbreviation':
                if 'ABBR' in match and 'EXP' in match:
                    partial_match = True
            else:
                print("Wrong type:", type_)
                sys.exit(1)


            if False not in match:
                exact_match = True

        partial_matches.append(partial_match)
        exact_matches.append(exact_match)
        # print('Matched:', match)
        # print('Is partially matched:',partial_match)
        # print('Is exactly matched:',exact_match)


    count_both_in_preds = sum(both_in_preds) # N
    count_both_in_labels = sum(both_in_labels) # M
    count_partial_matches = sum(partial_matches) # P
    count_exact_matches = sum(exact_matches) # E


    # print(count_both_in_preds, both_in_preds)
    # print(count_both_in_labels, both_in_labels)
    # print(count_partial_matches, partial_matches)
    # print(count_exact_matches, exact_matches)

    if count_both_in_preds == 0 or count_both_in_labels == 0 or count_partial_matches ==0:
        partial_precision = 0.0
        partial_recall = 0.0
        partial_fscore = 0.0
    else:
        partial_precision = count_partial_matches/count_both_in_preds
        partial_recall = count_partial_matches/count_both_in_labels
        partial_fscore = 2*partial_precision*partial_recall / (partial_precision+partial_recall)



    if count_both_in_preds == 0 or count_both_in_labels == 0 or count_exact_matches ==0 :
        exact_precision = 0.0
        exact_recall = 0.0
        exact_fscore = 0.0
    else:
        exact_precision =  count_exact_matches/count_both_in_preds
        exact_recall = count_exact_matches/count_both_in_labels
        exact_fscore = 2*exact_precision*exact_recall / (exact_precision+exact_recall)


    # print('Partial match precision: %.2f'%(partial_precision))
    # print('Partial match recall: %.2f'%(partial_recall))
    # print('Partial match fscore: %.2f'%(partial_fscore))

    # print('Exact match precision: %.2f'%(exact_precision))
    # print('Exact match recall: %.2f'%(exact_recall))
    # print('Exact match fscore: %.2f'%(exact_fscore))


    # if type_ is None:
    res =  {
        "partial_match_precision": partial_precision,
        "partial_match_recall": partial_recall,
        "partial_match_f1": partial_fscore,
        # "exact_match_precision": exact_precision,
        # "excat_match_recall": exact_recall,
        # "excat_match_f1": exact_fscore,
    }
    # elif type_ in ['term', 'symbol', 'abbreviation']:
        # res =  {
            # "{}_partial_match_precision".format(type_): partial_precision,
            # "{}_partial_match_recall".format(type_): partial_recall,
            # "{}_partial_match_f1".format(type_): partial_fscore,
            # # "{}_exact_match_precision".format(type_): exact_precision,
            # # "{}_excat_match_recall".format(type_): exact_recall,
            # # "{}_excat_match_f1".format(type_): exact_fscore,
        # }
    # else:
        # print("Wrong type:", type_)
    #     sys.exit(1)


    return res








def get_slot_simple_metrics(preds, labels, type_=None):
    """
    Conceptually, define the following new types of ‘virtual tags’
    TERM = B-term OR I-Term (ie the union of those two tags)
    DEF = B-Def OR I-Def
    Now, what are the P,R & F1 numbers for TERM and DEF?  (I think these matter because users may just care about accuracy of term and defn matching and the macro averaged scores conflate other things like recall on these metrics and precision on O. Likewise the current macro average treats missing the first word in a definition differently from skipping the last word.
    """
    assert len(preds) == len(labels)

    # flatten
    preds = [p for ps in preds for p in ps]
    labels = [l for ls in labels for l in ls]

    if type_ is None or type_ in ['term','symbol']:
        # simplify by replacing {B,I}-TERM to TERM and {B,I}-DEF to DEF
        simple_preds = simplify_tokens(preds, type_)
        simple_labels = simplify_tokens(labels, type_)
        assert len(simple_preds) == len(simple_labels)
        label_names = ['O', 'TERM','DEF']
    elif type_ == 'abbreviation':
        # simplify by replacing {B,I}-short to ABBR and {B,I}-long to EXP
        simple_preds = simplify_tokens(preds, type_)
        simple_labels = simplify_tokens(labels, type_)
        assert len(simple_preds) == len(simple_labels)
        label_names = ['O', 'ABBR','EXP']
    else:
        print("Wrong type:", type_)
        sys.exit(1)

    p,r,f,s = score(simple_labels, simple_preds, average=None, labels=label_names)
    s = [int(si) for si in s]
    p = [round(float(pi),3) for pi in p]
    r = [round(float(pi),3) for pi in r]
    f = [round(float(pi),3) for pi in f]
    per_class = {'p':list(p),'r':list(r), 'f':list(f), 's':list(s)}
    # pprint(per_class)

    res =  {
        "slot_merged_TERM_precision": per_class['p'][1],
        "slot_merged_TERM_recall": per_class['r'][1],
        "slot_merged_TERM_f1": per_class['f'][1],
        "slot_merged_DEFINITION_precision": per_class['p'][2],
        "slot_merged_DEFINITION_recall": per_class['r'][2],
        "slot_merged_DEFINITION_f1": per_class['f'][2],
        "slot_merged_TERM_DEFINITION_f1_mean" : (per_class['f'][1] + per_class['f'][2])/2,

    }

    # if type_ is None or type_ in ["term", "symbol"]:
        # res =  {
            # "slot_merged_TERM_precision": per_class['p'][1],
            # "slot_merged_TERM_recall": per_class['r'][1],
            # "slot_merged_TERM_f1": per_class['f'][1],
            # "slot_merged_DEFINITION_precision": per_class['p'][2],
            # "slot_merged_DEFINITION_recall": per_class['r'][2],
            # "slot_merged_DEFINITION_f1": per_class['f'][2],
        # }
    # elif type_=='abbreviation':
        # res =  {
            # "slot_merged_ABBREVIATION_precision": per_class['p'][1],
            # "slot_merged_ABBREVIATION_recall": per_class['r'][1],
            # "slot_merged_ABBREVIATION_f1": per_class['f'][1],
            # "slot_merged_EXPANSION_precision": per_class['p'][2],
            # "slot_merged_EXPANSION_recall": per_class['r'][2],
            # "slot_merged_EXPANSION_f1": per_class['f'][2],
        # }
    # else:
        # print("Wrong type:", type_)
    #     sys.exit(1)


    return res





def get_slot_metrics(preds, labels, type_=None):
    assert len(preds) == len(labels)

    # flatten
    preds = [p for ps in preds for p in ps]
    labels = [l for ls in labels for l in ls]

    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')
    macro_p = precision_score(labels, preds, average='macro')
    micro_p = precision_score(labels, preds, average='micro')
    macro_r = recall_score(labels, preds, average='macro')
    micro_r = recall_score(labels, preds, average='micro')

    if type_ is None or type_ in ['term','symbol']:
        label_names = ['O', 'B-TERM','I-TERM', 'B-DEF', 'I-DEF']
    elif type_=='abbreviation':
        label_names = ['O', 'B-short','B-long', 'I-short','I-long']
    p,r,f,s = score(labels, preds, average=None, labels=label_names)
    s = [int(si) for si in s]
    p = [round(float(pi),3) for pi in p]
    r = [round(float(pi),3) for pi in r]
    f = [round(float(pi),3) for pi in f]
    per_class = {'p':list(p),'r':list(r), 'f':list(f), 's':list(s)}
    # print(per_class)


    res =  {
        "slot_precision_macro": macro_p,
        "slot_recall_macro": macro_r,
        "slot_f1_macro": macro_f1,
        "slot_precision_micro": micro_p,
        "slot_recall_micro": micro_r,
        "slot_f1_micro": micro_f1,
        "slot_precision_per_label": per_class['p'],
        "slot_recall_per_label": per_class['r'],
        "slot_f1_per_label": per_class['f'],
        "slot_num_per_label": per_class['s'],
    }

    # if type_ is None:
        # res =  {
            # "slot_precision_macro": macro_p,
            # "slot_recall_macro": macro_r,
            # "slot_f1_macro": macro_f1,
            # "slot_precision_micro": micro_p,
            # "slot_recall_micro": micro_r,
            # "slot_f1_micro": micro_f1,
            # "slot_precision_per_label": per_class['p'],
            # "slot_recall_per_label": per_class['r'],
            # "slot_f1_per_label": per_class['f'],
            # "slot_num_per_label": per_class['s'],
        # }
    # else:
        # res =  {
            # "{}_slot_precision_macro".format(type_): macro_p,
            # "{}_slot_recall_macro".format(type_): macro_r,
            # "{}_slot_f1_macro".format(type_): macro_f1,
            # "{}_slot_precision_micro".format(type_): micro_p,
            # "{}_slot_recall_micro".format(type_): micro_r,
            # "{}_slot_f1_micro".format(type_): micro_f1,
            # "{}_slot_precision_per_label".format(type_): per_class['p'],
            # "{}_slot_recall_per_label".format(type_): per_class['r'],
            # "{}_slot_f1_per_label".format(type_): per_class['f'],
            # "{}_slot_num_per_label".format(type_): per_class['s'],
    #     }
    # print('\n\n\n',type_, res,'\n\n\n')
    return res

def get_intent_acc(preds, labels, type_=None):
    acc = (preds == labels).mean()
    if type_ is None:
        res = {
            "intent_acc": acc
        }
    else:
        res = {
            "{}_intent_acc".format(type_): acc
        }
    return res


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }



