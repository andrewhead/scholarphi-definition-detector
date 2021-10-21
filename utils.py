import os
import json
import random
import logging
from pprint import pprint
from colorama import Fore,Style
from collections import defaultdict, Counter
from typing import Any, List, Dict, Tuple, Optional, DefaultDict, Union

from scipy import stats
import matplotlib.pylab as plt

import torch
import numpy as np
# from seqeval.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from dataclasses import dataclass
import spacy
from scispacy.abbreviation import AbbreviationDetector
from spacy.matcher import Matcher
from spacy.util import filter_spans

# rule related functions in data_loader
def explore_rules(data, rules):
    rule_dict = defaultdict(lambda: defaultdict(list))
    data_matched = []
    for d in data:
        matched = False
        rules_matched = []
        for rule in rules:
            if match_rule_to_tokens(rule, d['tokens']):
                print("*********MATCHED***********")
                print(rule)
                print(d['tokens'])
                print(d['pos'])
                print(d['label'])
                print(d['labels'])
                print("****************************")

                rule_dict[rule]['label'].append(d['label'])

                matched = True
                rules_matched.append(rule)

        # if rule matched, append it to dataset
        if matched:
            d['rules'] = rules_matched
            data_matched.append(d)

    for r, ldict in rule_dict.items():
        for l, items in ldict.items():
            print(r, l, Counter(items))
    return data_matched

def match_rule_to_tokens(rule, word_tokens):
    rule_tokens = rule.split()

    for tid, word_token in enumerate(word_tokens):
        if tid + len(rule_tokens) > len(word_tokens):
            break
        matched = 0
        for rid, rule_token in enumerate(rule_tokens):
            if word_tokens[tid+rid] == rule_token:
                matched += 1
        if matched == len(rule_tokens):
            return True

    return False






class NLP(object):
  def __init__(self):
    self.nlp = spacy.load("en_core_sci_md")
    self.nlp.add_pipe("abbreviation_detector")

    verb_pattern = [
        {"POS": "VERB", "OP": "?"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "AUX", "OP": "*"},
        {"POS": "VERB", "OP": "+"},]
    self.verb_matcher = Matcher(self.nlp.vocab)
    self.verb_matcher.add("Verb phrase", [verb_pattern])

  def featurize(self, text: str, limit: bool = False) -> DefaultDict[Any, Any]:
        doc = self.nlp(text)

        # Extract tokens containing...
        # (1) Abbreviations
        abbrev_tokens = []
        for abrv in doc._.abbreviations:
            abbrev_tokens.append(str(abrv._.long_form).split())
        abbrev_tokens_flattened = [t for at in abbrev_tokens for t in at]

        # (2) Entities
        entities = [str(e) for e in doc.ents]
        entity_tokens = [e.split() for e in entities]
        entity_tokens_flattened = [t for et in entity_tokens for t in et]

        # (3) Noun phrases
        np_tokens = []
        for chunk in doc.noun_chunks:
            np_tokens.append(str(chunk.text).split())
        np_tokens_flattened = [t for et in np_tokens for t in et]

        # (4) Verb phrases
        verb_matches = self.verb_matcher(doc)
        spans = [doc[start:end] for _, start, end in verb_matches]
        vp_tokens = filter_spans(spans)
        vp_tokens_flattened = [str(t) for et in vp_tokens for t in et]

        # Limit the samples.
        if limit:
            doc = doc[:limit]

        # Aggregate all features together.
        features: DefaultDict[str, List[Union[int, str]]] = defaultdict(list)
        for token in doc:
            if str(token.text) == '---':
                features["tokens"].append('</s>')
            else:
                features["tokens"].append(str(token.text))
            features["pos"].append(str(token.tag_))  # previously token.pos_
            features["head"].append(str(token.head))
            # (Note: the following features are binary lists indicating the presence of a
            # feature or not per token, like "[1 0 0 1 1 1 0 0 ...]")
            features["entities"].append(
                1 if token.text in entity_tokens_flattened else 0
            )
            features["np"].append(1 if token.text in np_tokens_flattened else 0)
            features["vp"].append(1 if token.text in vp_tokens_flattened else 0)
            features["abbreviation"].append(
                1 if token.text in abbrev_tokens_flattened else 0
            )

        return features



def merge_slot_predictions(labels_list, merge_method="union",
                           term_postfix="TERM",def_postfix="DEF"):
    labels_list = np.array(labels_list)
    final_labels = []
    for labels_at_i in np.array(labels_list).transpose():
        if merge_method == "union":
            # TERM > DEF > O
            final_label = "O"
            # print(labels_at_i)
            for l,c in Counter(labels_at_i).most_common():
                if l.endswith(term_postfix):
                    final_label = l
                    break
                elif l.endswith(def_postfix):
                    final_label = l
            final_labels.append(final_label)
            # print("\t",final_label)
        elif merge_method == "overlap":
            # TERM > DEF > O
            final_label = "O"
            # print(labels_at_i)
            for l,c in Counter(labels_at_i).most_common():
                if l.endswith(term_postfix):
                    if c > 1:
                        final_label = l
                        break
                elif l.endswith(def_postfix):
                    if c > 1:
                        final_label = l
            final_labels.append(final_label)
        else:
            print("Wrong merge method",merge_method)
            sys.exit(1)

    return final_labels




class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)



def transpose_dict_to_list(input_dict):
    # all entities in the dictionaty should have the same number of instance list
    assert len(set([len(v) for k,v in input_dict.items()])) == 1

    keys = list(input_dict.keys())
    output_list = [{} for _ in range(len(input_dict[keys[0]]))]
    for k, v in input_dict.items():
        for idx in range(len(output_list)):
            output_list[idx][k] = v[idx]
    return output_list

def transpose_dict_to_list_hybrid(input_dict, without_subdicts = []):
    # all entities in the dictionaty should have the same number of instance list
    # assert len(set([len(v) for k,v in input_dict.items()])) == 1

    keys = list(input_dict.keys())
    output_list = [{} for _ in range(len(input_dict[without_subdicts[0]]))]
    for k, v in input_dict.items():
        for idx in range(len(output_list)):
            if k in without_subdicts:
                output_list[idx][k] = v[idx]
            else:
                adict = {}
                for dataset_type in input_dict[k]:
                    adict[dataset_type] = v[dataset_type][idx]
                output_list[idx][k] = adict
                # for key in input_dict[k]:
                    # newkey = key+"_"+k
                #     output_list[idx][newkey] = v[key][idx]
    return output_list


@dataclass(frozen=True)
class CharacterRange:
    start: int
    "Character position where this range begins in the TeX."

    end: int
    "Character position where this range ends in the TeX."


def get_token_character_ranges(text: str, tokens: List[str]) -> List[CharacterRange]:
    """
    Extract start and end charcter positions for each token in featurized tokens
    """
    ranges = []
    current_position = 0
    for token in tokens:
        start_index = text[current_position:].index(token)
        ranges.append(
          CharacterRange(
              current_position + start_index,
              current_position + start_index + len(token) - 1,
          )
        )
        current_position += (len(token) + start_index)
    return ranges


def replace_latex_notation_to_SYMBOL(text):
    # symbol_detection = regex.findall(r"\$[\S\n ]+\$", text)
    symbol_detection = regex.findall(r"\$.*?\$", text)
    if symbol_detection:
        # print(text)
        # print(symbol_detection)
        for sym in symbol_detection:
            text = text.replace(sym, "SYMBOL")
        # print(text)
    return text, symbol_detection


def mapping_feature_tokens_to_original(
    tokens, ranges,
    symbol_start, symbol_end,
    nickname_start, nickname_end, symbols=False
):
    new_tokens = []
    new_labels = []

    current_symbol_index = 0
    for token, token_range in zip(tokens,ranges):
        if symbols:
            if token == 'SYMBOL':
                # assign token
                new_tokens.append(symbols[current_symbol_index])
                current_symbol_index += 1
            else:
                # assign token
                new_tokens.append(token)

        # assign labels
        if symbol_start <= token_range.start and token_range.end <= symbol_end :
            new_labels.append("TERM")
        elif nickname_start <= token_range.start and token_range.end <= nickname_end :
            new_labels.append("DEF")
        else:
            new_labels.append("O")

    if symbols:
        assert len(new_tokens) == len(new_labels)
    # change TERM -> B-TERM and I-TERM
    # change DEF -> B-DEF I-DEF
    new_labels_BIO = []
    prev_l = 'O'
    for l in new_labels:
        new_l = 'O'
        if l == 'TERM':
            if prev_l == 'TERM':
                new_l = 'I-TERM'
            else:
                new_l = 'B-TERM'
        if l == 'DEF':
            if prev_l == 'DEF':
                new_l = 'I-DEF'
            else:
                new_l = 'B-DEF'
        new_labels_BIO.append(new_l)
        prev_l = l
    assert len(new_labels) == len(new_labels_BIO)

    return new_tokens, new_labels_BIO


def sanity_check_for_acronym_detection(
        acronym_text, expansion_text,
        acronym_start, acronym_end,
        expansion_start, expansion_end):
    """
    Very smooth filter
    """
    # if acronym and expansion overlaps in positions, ignore them
    if len(set(range(acronym_start, acronym_end-1)).intersection(range(expansion_start, expansion_end-1))) > 0:
        return False

    expansion_text = expansion_text.lower()
    acronym_text = acronym_text.lower()

    if "citation" in acronym_text:
        return False

    # first_letter_of_each_word_in_expansion = [w[0] for w in expansion_text.split(" ")]

    is_acronym = True
    for first_letter_in_acronym in acronym_text:
        # if not first_letter_in_acronym in first_letter_of_each_word_in_expansion:
        #     is_acronym = False
        if not first_letter_in_acronym in expansion_text:
            is_acronym = False

    return is_acronym





def match_tokenized_to_untokenized(tokens, split_tokens):
    # len(tokens) > len(split_tokens)
    # mapping[split_tokens]  = [tokens]
    mapping = defaultdict(list)
    split_tokens_index = 0
    tokens_index = 0
    while (split_tokens_index < len(split_tokens) and tokens_index < len(tokens)):
        while (tokens_index + 1 <= len(tokens) and tokens[tokens_index] in split_tokens[split_tokens_index] ):
            tokens_prefix = ''
            if split_tokens_index in mapping:
                tokens_prefix += ''.join([tokens[mt] for mt in mapping[split_tokens_index]])
            tokens_prefix_with_token = (tokens_prefix + " "+ tokens[tokens_index]).strip()
            if tokens_prefix_with_token in split_tokens[split_tokens_index]:
                print(tokens_index, split_tokens_index, tokens_prefix_with_token,split_tokens[split_tokens_index])
                mapping[split_tokens_index].append(tokens_index)
                tokens_index += 1
            else:
                print(tokens_index, split_tokens_index, tokens_prefix_with_token,split_tokens[split_tokens_index])
                break
        split_tokens_index += 1

    assert len(mapping) == len(split_tokens)
    assert sum([len(s) for s in mapping.values()]) == len(tokens)
    # mapping[split_tokens_index].append(tokens_index)

    token_to_split_token = {} #v:k for k,v in mapping.items()}
    for k,v in mapping.items():
        for vone in v:
            token_to_split_token[vone] = k
    return token_to_split_token #mapping


def colorize_term_definition(tokens, labels,
                           term_postfix="TERM",def_postfix="DEF"):
    output = []
    for t,l in zip(tokens, labels):
        t_colored = ""
        if l.endswith(term_postfix):
            t_colored = Fore.YELLOW+str(t)+Style.RESET_ALL
        elif l.endswith(def_postfix):
            t_colored = Fore.GREEN+str(t)+Style.RESET_ALL
        else:
            t_colored = t
        output.append(t_colored)
    return " ".join(output)

def colorize_labels(text, term_start, term_end, def_start, def_end):
    output = ""
    for tidx,t in enumerate(text):
        t_colored = ""
        if term_start <= tidx and tidx < term_end:
            t_colored = Fore.YELLOW+str(t)+Style.RESET_ALL
        elif def_start <= tidx and tidx < def_end:
            t_colored = Fore.GREEN+str(t)+Style.RESET_ALL
        else:
            t_colored = t
        output += t_colored
    return output



def highlight(input):
    input = str(input)
    return str(Fore.RED+str(input)+Style.RESET_ALL)

def get_intent_labels(args):
    task = args.task.split('+')[0]
    task = task.split('_')[0]
    return [label.strip() for label in open(os.path.join(args.data_dir, task, args.intent_label_file), 'r', encoding='utf-8')]

def get_slot_labels(args):
    task = args.task.split('+')[0]
    task = task.split('_')[0]
    return [label.strip() for label in open(os.path.join(args.data_dir, task, args.slot_label_file), 'r', encoding='utf-8')]

def get_pos_labels(args):
    task = args.task.split('+')[0]
    task = task.split('_')[0]
    return [label.strip() for label in open(os.path.join(args.data_dir, task, args.pos_label_file), 'r', encoding='utf-8')]


def get_intent_labels_hybrid(args):
    intent_hybrid = {}
    for task in args.task.split('+'):
        task = task.split("_")[0]
        intent_hybrid[task] = [label.strip() for label in open(os.path.join(args.data_dir, task, args.intent_label_file), 'r', encoding='utf-8')]
    return intent_hybrid

def get_slot_labels_hybrid(args):
    slot_hybrid = {}
    for task in args.task.split('+'):
        task = task.split("_")[0]
        slot_hybrid[task] = [label.strip() for label in open(os.path.join(args.data_dir, task, args.slot_label_file), 'r', encoding='utf-8')]
    return slot_hybrid

def get_pos_labels_hybrid(args):
    pos_hybrid = {}
    for task in args.task.split('+'):
        task = task.split("_")[0]
        pos_hybrid[task] = [label.strip() for label in open(os.path.join(args.data_dir, task, args.pos_label_file), 'r', encoding='utf-8')]
    return pos_hybrid

def get_joint_labels(args, key):
    with open(
            os.path.join(args.output_dir, args.dataconfig_file),
            "r",
            encoding="utf-8",
        ) as f:
        data_config = json.load(f)
    return data_config[key]


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def info(logger, training_args):
    logger.info(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            highlight(training_args.local_rank),
            highlight(training_args.device),
            highlight(training_args.n_gpu),
            highlight(bool(training_args.local_rank != -1)),
            highlight(training_args.fp16))
    logger.info("Training/evaluation parameters %s", highlight(training_args))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)



# visualization functions
def plot_posterior(eval_label, eval_prob):
    num_labels = max(eval_label) + 1
    # eval_label = np.eye(num_labels)[eval_label]

    #Collecting All probability
    all_probs = []
    all_labels = []

    eval_label = np.array(eval_label)
    out_freqs, out_emp, out_true = [],[],[]
    y_axis = []
    # for idx in range(num_labels):
        # if idx == 1:
            # all_probs += np.array(eval_prob)[:,idx].tolist()
    #         all_labels += eval_label[:,idx].tolist()

    all_labels = np.array(eval_label) #all_labels)
    all_probs = np.array(eval_prob) #all_probs)
    #Bin
    bin_size = 10
    bins = np.linspace(0,1,bin_size + 1)
    # if not adaptive:
    bins_fin = bins
    # else:
    # bins_fin = stats.mstats.mquantiles(all_probs,bins)

    lower, upper = bins_fin[:-1], bins_fin[1:]
    for idx, bound in enumerate(zip(lower,upper)):
        y_axis.append(np.mean(bound))
        in_bin = np.greater_equal(all_probs,bound[0]) * np.less(all_probs,bound[1])
        if idx == len(lower) - 1:
             in_bin = np.greater_equal(all_probs,bound[0]) * np.less_equal(all_probs,bound[1])

        out_emp.append(all_probs[in_bin].mean())
        out_true.append(all_labels[in_bin].mean())
        out_freqs.append(np.sum(in_bin))

    b_emp = out_emp
    b_true = out_true
    b_freq = out_freqs
    y_axsis = y_axis

    graph_dir = ''
    task_name = 'def'
    plt.switch_backend('agg')


    f, ax = plt.subplots()#(figsize=(10, 10))
    width = 0.1

    # Get log scale y
    adaptive=False
    hist_y_log=True
    if hist_y_log:
        b_freq = np.log(b_freq)
        path = os.path.join(graph_dir,'hist_%s_%s_%s_log.png'
                %(task_name,str(bin_size),adaptive))

    else:
        path = os.path.join(graph_dir,'hist_%s_%s_%s_log.png'
                %(task_name,str(bin_size),adaptive))

    plt.bar(y_axis,b_freq,width,color='blue',edgecolor='black'
            ,alpha=0.2, label='MLE', linewidth=3)
    plt.xlim((0,1))
    #plt.ylim((0,1))
    x = np.linspace(*ax.get_xlim())
    ax.legend(loc='upper center', fontsize=18)
    plt.tick_params(labelsize=18)
    plt.savefig(path, dpi=300)
    plt.close()


    f, ax = plt.subplots()
    width = 0.1
    plt.bar(y_axis,b_true,width,color='blue',edgecolor='black'
            ,alpha=0.2, label='MLE', linewidth=3)
    plt.xlim((0,1))
    plt.ylim((0,1))
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, color='black', linestyle='dashed', linewidth=4)
    ax.legend(loc='best', fontsize=18)
    path=os.path.join(graph_dir,'calgraph_%s_%s_%s.png' %(task_name,str(bin_size),adaptive))
    plt.tick_params(labelsize=18)
    plt.savefig(path, dpi=1000)
    plt.close()



