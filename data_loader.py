import os
import copy
import numpy as np
import json
import pickle as pkl
import logging
from collections import defaultdict, Counter
import torch
from torch.utils.data import TensorDataset
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import logging

from utils import get_intent_labels, get_slot_labels, get_pos_labels, get_intent_labels_hybrid, get_slot_labels_hybrid, get_pos_labels_hybrid,NLP,match_rule_to_tokens,explore_rules


from abbreviation_detector import get_abbreviation_pairs

logger = logging.get_logger()

def extract_annotated_labels_from_list(tokens, rel_list, verbose=False):
    """
    slot_labels = "O O B-TERM O B-DEF I-DEF I-DEF I-DEF I-DEF "
    label = "none"
    """
    slot_labels = []
    start_index, end_index = 0, 0
    slot_labels = []

    term_list = [r["term"] for r in rel_list]
    def_list = [r["definition"] for r in rel_list]
    #NOTE all term_list should have the same start and end positions
    for token in tokens:
        end_index = start_index + len(token)
        slot_label = "O"
        for term in term_list:
            if term["start"] <= start_index and end_index <= term["end"]:
                slot_label = "TERM"
        for definition in def_list:
            if definition["start"] <= start_index and end_index <= definition["end"]:
                slot_label = "DEF"

        # done
        slot_labels.append(slot_label)
        start_index += len(token) + 1

    assert len(slot_labels) == len(tokens)

    label = "none"
    if 'TERM' in slot_labels and 'DEF' in slot_labels:
        label = "definition"

    new_slot_labels = []
    prev_l = 'O'
    for l in slot_labels:
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
        new_slot_labels.append(new_l)
        prev_l = l
    assert len(new_slot_labels) == len(slot_labels)

    if verbose:
        print("=====================")
        print(tokens)
        print(new_slot_labels)
        print(label)
        print("=====================")
    return new_slot_labels, label



def extract_annotated_labels(tokens, term_dict, def_dict, verbose=False):
    """
    slot_labels = "O O B-TERM O B-DEF I-DEF I-DEF I-DEF I-DEF "
    label = "none"
    """
    slot_labels = []
    start_index, end_index = 0, 0
    slot_labels = []
    for token in tokens:
        end_index = start_index + len(token)
        slot_label = "O"
        if term_dict["start"] <= start_index and end_index <= term_dict["end"]:
            slot_label = "TERM"
        if def_dict["start"] <= start_index and end_index <= def_dict["end"]:
            slot_label = "DEF"

        # done
        slot_labels.append(slot_label)
        start_index += len(token) + 1

    assert len(slot_labels) == len(tokens)

    label = "none"
    if 'TERM' in slot_labels and 'DEF' in slot_labels:
        label = "definition"


    # sometimes, annotators confused labeling term and definitions in the opposite way.
    # if DEF has only SYMBOL, then flip the annotations
    term_token_list = []
    def_token_list = []
    for t,l in zip(tokens, slot_labels):
        if l == "DEF":
            def_token_list.append(t)
    if len(def_token_list) == 1 and def_token_list[0] == "SYMBOL":
        # flip
        new_slot_labels = []
        for l in slot_labels:
            if l == "TERM":
                new_slot_labels.append("DEF")
            elif l == "DEF":
                new_slot_labels.append("TERM")
            else:
                new_slot_labels.append("O")
        slot_labels = new_slot_labels

    new_slot_labels = []
    prev_l = 'O'
    for l in slot_labels:
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
        new_slot_labels.append(new_l)
        prev_l = l
    assert len(new_slot_labels) == len(slot_labels)

    if verbose:
        print("=====================")
        print(tokens)
        print(new_slot_labels)
        print(label)
        print("=====================")
    return new_slot_labels, label


# the user input with ad-hoc annotation scheme
def extract_annotated_labels_adhoc(sentence):
    """
    sentence = "we define A:TERM as a:D system:D for:D neural:D net:D
    text = "we define A as a system for neural net"
    slot_labels = "O O B-TERM O B-DEF I-DEF I-DEF I-DEF I-DEF "
    label = "none"
    """
    text = []
    slot_labels = []
    label = "none"
    for token in sentence.split():
        if token.endswith(":T"):
            text.append(token.split(":T")[0])
            slot_labels.append("TERM")
        elif token.endswith(":D"):
            text.append(token.split(":D")[0])
            slot_labels.append("DEF")
        else:
            text.append(token)
            slot_labels.append("O")

    if "TERM" in slot_labels and "DEF" in slot_labels:
        label = "definition"


    # change TERM -> B-TERM and I-TERM
    # change DEF -> B-DEF I-DEF
    new_slot_labels = []
    prev_l = 'O'
    for l in slot_labels:
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
        new_slot_labels.append(new_l)
        prev_l = l
    assert len(new_slot_labels) == len(slot_labels)

    slot_labels = new_slot_labels

    return text, slot_labels, label






class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """
    def __init__(self, guid, type, mode, words, intent_label=None, slot_labels=None, pos_labels=None, np_labels=None, vp_labels=None,
                 entity_labels=None, acronym_labels=None):
        self.guid = guid
        self.type = type
        self.mode = mode

        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels
        self.pos_labels = pos_labels
        self.np_labels = np_labels
        self.vp_labels = vp_labels
        self.entity_labels = entity_labels
        self.acronym_labels = acronym_labels


    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids,
                 pos_labels_ids, np_labels_ids, vp_labels_ids,
                 entity_labels_ids, acronym_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

        self.pos_labels_ids = pos_labels_ids
        self.np_labels_ids = np_labels_ids
        self.vp_labels_ids = vp_labels_ids
        self.entity_labels_ids = entity_labels_ids
        self.acronym_labels_ids = acronym_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class DefProcessor(object):
    def __init__(self, args, task, hybrid=False):
        self.hybrid = hybrid
        if self.hybrid:
            self.intent_labeler = get_intent_labels_hybrid
            self.slot_labeler = get_slot_labels_hybrid
        else:
            self.intent_labeler = get_intent_labels
            self.slot_labeler = get_slot_labels

        self.exampleWrapper = InputExample
        self.pos_labeler = get_pos_labels
        self.args = args
        self.intent_labels = self.intent_labeler(args)
        self.slot_labels = self.slot_labeler(args)
        self.pos_labels = self.pos_labeler(args)
        self.target_task = task

        # load config file
        config_file = os.path.join(self.args.data_dir, self.target_task, "config.json")
        config = json.load(open(config_file))
        self.name = config["name"]
        self.type = config["type"]
        self.version = config["version"]
        self.folded = bool(config["folded"])

        # make negative but silver data for joint leraning
        #TODO later, run this on data creation and nver do inference here
        self.nlp_model = NLP()


    # @classmethod
    def _read_file(self, input_file, quotechar=None):
        data = []
        if os.path.isfile(input_file):
            with open(input_file, encoding="utf-8") as infile:
                data = json.load(infile)

        # include config information to each instance
        for d in data:
            d["name"] = self.name
            d["type"] = self.type
            d["version"] = self.version
        return data

    def _create_examples(self, data, set_type): #intents, slots,
        """Creates examples for the training and dev sets."""
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s-%s-%s" % (self.name, self.type, set_type, i)
            # 1. input_text
            words = d['tokens'] #text.split()  # Some are spaced twice

            if self.hybrid:
                intent_label = {}
                slot_labels = {}
                if set_type != "unlabeled":
                    for data_type in self.args.dataset.split('+'):
                        if data_type == self.target_task:
                            # 2. intent
                            intent_label[data_type] = self.intent_labels[data_type].index(d['label']) if d['label'] in self.intent_labels[data_type] else self.intent_labels[data_type].index("none")
                            # 3. slot
                            slot_labels[data_type] = []
                            for s in d['labels']:
                                assert s in self.slot_labels[data_type]
                                slot_labels[data_type].append(self.slot_labels[data_type].index(s))
                        # negative samples
                        else:
                            # make negative samples to be more clean
                            if data_type == "AI2020":
                                abbr_pairs = get_abbreviation_pairs(words, self.nlp_model)
                                # print(words)
                                # print(abbr_pairs)
                                # otherwise, make them as negative examples
                                # 2. intent
                                # 3. slot
                                if len(abbr_pairs)  == 0:
                                    intent_label[data_type] = self.intent_labels[data_type].index("none")
                                    slot_labels[data_type] = []
                                    for s in d['labels']:
                                        slot_labels[data_type].append(self.slot_labels[data_type].index("O"))
                                else:
                                    terms, defs = [],[]
                                    for w, l in zip(words, abbr_pairs[0]):
                                        if l.endswith("TERM"):
                                            terms.append(w)
                                        if l.endswith("DEF"):
                                            defs.append(w)
                                    terms = " ".join(terms).strip().lower()
                                    defs = " ".join(defs).strip().lower()

                                    if defs in ["table", "symbol", "section", "sections", "figure", "fig", "appendix"] or ")" in terms or "(" in terms or terms == defs:
                                        intent_label[data_type] = self.intent_labels[data_type].index("none")
                                        slot_labels[data_type] = []
                                        for s in d['labels']:
                                            slot_labels[data_type].append(self.slot_labels[data_type].index("O"))
                                    else:
                                        # print(terms,"\t",defs)
                                        intent_label[data_type] = self.intent_labels[data_type].index("abbreviation")
                                        abbr_pairs_converted = []
                                        for l in abbr_pairs[0]:
                                            if l == "B-TERM":
                                                abbr_pairs_converted.append("B-long")
                                            elif l == "I-TERM":
                                                abbr_pairs_converted.append("I-long")
                                            elif l == "B-DEF":
                                                abbr_pairs_converted.append("B-short")
                                            elif l == "I-DEF":
                                                abbr_pairs_converted.append("I-short")
                                            else:
                                                abbr_pairs_converted.append(l)
                                        slot_labels[data_type] = []
                                        for s in abbr_pairs_converted:
                                            slot_labels[data_type].append(self.slot_labels[data_type].index(s))

                                # print(intent_label[data_type])
                                # print(slot_labels[data_type])

                            else:
                                # otherwise, make them as negative examples
                                # 2. intent
                                intent_label[data_type] = self.intent_labels[data_type].index("none")
                                # 3. slot
                                slot_labels[data_type] = []
                                #for exclusive combination, use UNK or O (should be empirically tested. for now UNK)
                                for s in d['labels']:
                                    slot_labels[data_type].append(self.slot_labels[data_type].index("O"))
                else:
                    for data_type in self.args.dataset.split('+'):
                        # 2. intent
                        intent_label[data_type] = d['label']
                        # 3. slot
                        slot_labels[data_type] = d['labels']

                for data_type in self.args.dataset.split('+'):
                    assert len(words) == len(slot_labels[data_type])
            else:
                if set_type != "unlabeled":
                    intent_label = self.intent_labels.index(d['label']) if d['label'] in self.intent_labels else self.intent_labels.index("UNK")
                    slot_labels = []
                    for s in d['labels']:
                        assert s in self.slot_labels
                        slot_labels.append(self.slot_labels.index(s))
                else:
                    if "label" in d and "labels" in d:
                        if type(d["label"]) == int:
                            intent_label = d['label']
                        else:
                            intent_label = self.intent_labels.index(d['label']) if d['label'] in self.intent_labels else self.intent_labels.index("UNK")
                        if type(d["labels"][0]) == int:
                            slot_labels = d['labels']
                        else:
                            slot_labels = []
                            for s in d['labels']:
                                assert s in self.slot_labels
                                slot_labels.append(self.slot_labels.index(s))
                    else:
                        intent_label = 0
                        slot_labels = [0] * len(d['tokens'])

                assert len(words) == len(slot_labels)

            pos_labels = [self.pos_labels.index(s) if s in self.pos_labels else 0 for s in d['pos']]
            np_labels = d['np']
            vp_labels = d['vp']
            entity_labels = d['entities']
            acronym_labels = d['acronym']

            examples.append(self.exampleWrapper(
                                 guid=guid,
                                 type=self.type,
                                 mode=set_type,
                                 words=words,
                                 intent_label=intent_label,
                                 slot_labels=slot_labels,
                                 pos_labels=pos_labels,
                                 np_labels=np_labels,
                                 vp_labels=vp_labels,
                                 entity_labels=entity_labels,
                                 acronym_labels=acronym_labels))
        return examples


    def get_examples(self, mode, featurizer=False, rules=False, limit=-1):
        # decide data path
        if self.folded and self.args.kfold >= 0:
            kfold_dir = str(self.args.kfold) if self.args.kfold>=0 else ''
            data_path = os.path.join(self.args.data_dir, self.target_task, kfold_dir)
        else:
            # kfold_dir = str(self.args.kfold) if self.args.kfold>=0 else ''
            data_path = os.path.join(self.args.data_dir, self.target_task)

        # decide file path
        if mode == 'unlabeled':
            file_path = os.path.join(data_path, '{}'.format(self.args.dataset) )
        elif mode in ['train','test','dev']:
            file_path = os.path.join(data_path, '{}.json'.format(mode) )
        else:
            file_path = os.path.join(data_path, '{}'.format(mode) )

        logger.info("LOOKING AT {}".format(file_path))

        data = self._read_file(file_path)

        # rule exploration/bootstrapping
        if rules:
            data = explore_rules(data,rules)

        if limit > 0:
            data = data[:limit]

        if data is not None and len(data) > 0:
            return self._create_examples(data=data,set_type=mode), data
        else:
            return [],[]


class FeatureProcessor(object):
    def __init__(self, args, task):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)
        self.pos_labels = get_pos_labels(args)
        self.target_task = task
        self.type="feature"

    def _create_examples(self, data, set_type="feature"): #intents, slots,
        examples = []

        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)

            words = d['tokens']

            if "label" in d:
                intent_label = self.intent_labels.index(d['label']) if d['label'] in self.intent_labels else self.intent_labels.index("UNK")
            else:
                intent_label = self.intent_labels.index("none")

            if "labels" in d:
                slot_labels = []
                for s in d['labels']:
                    assert s in self.slot_labels
                    slot_labels.append(self.slot_labels.index(s))
            else:
                slot_labels = []
                for _ in words:
                    slot_labels.append(self.slot_labels.index("O"))

            pos_labels = [self.pos_labels.index(s) if s in self.pos_labels else 0 for s in d['pos']]
            np_labels = d['np']
            vp_labels = d['vp']
            entity_labels = d['entities']
            acronym_labels = d['acronym']

            examples.append(InputExample(guid=guid,
                                         type=self.type,
                                         mode=set_type,
                                         words=words, intent_label=intent_label, slot_labels=slot_labels,
                                         pos_labels=pos_labels,
                                         np_labels=np_labels, vp_labels=vp_labels,
                                         entity_labels=entity_labels, acronym_labels=acronym_labels
                                         ))
        return examples

    def get_examples(self, data, rules=False, limit=-1): #sentences,
        logger.info("LOOKING AT {}".format(len(data)))

        # rule exploration/bootstrapping
        if rules:
            data = self.explore_rules(data,rules)

        if limit > 0:
            data = data[:limit]

        if data is not None and len(data) > 0:
            return self._create_examples(data=data), data
        else:
            return [],[]



class InputProcessor(object):
    """Processor for the DefMiner data set """

    def __init__(self, args, task):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)
        self.pos_labels = get_pos_labels(args)
        self.target_task = task
        self.type="input"

    def _create_examples(self, data, set_type="input"): #intents, slots,
        """Creates examples for the training and dev sets."""
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)

            words = d['tokens']
            # intent_label = 0
            # slot_labels = [0] * len(words)
            # intent_label = d['label']
            # slot_labels = d['labels']

            # 2. intent
            intent_label = self.intent_labels.index(d['label']) if d['label'] in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            slot_labels = []
            for s in d['labels']:
                assert s in self.slot_labels
                slot_labels.append(self.slot_labels.index(s))

            assert len(words) == len(slot_labels)

            pos_labels = [self.pos_labels.index(s) if s in self.pos_labels else 0 for s in d['pos']]
            np_labels = d['np']
            vp_labels = d['vp']
            entity_labels = d['entities']
            acronym_labels = d['acronym']

            examples.append(InputExample(guid=guid,
                                         type=self.type,
                                         mode=set_type,
                                         words=words, intent_label=intent_label, slot_labels=slot_labels,
                                         pos_labels=pos_labels,
                                         np_labels=np_labels, vp_labels=vp_labels,
                                         entity_labels=entity_labels, acronym_labels=acronym_labels
                                         ))
        return examples

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data = []
        with open(input_file, encoding="utf-8") as infile:
            for line in infile:
                data.append(line.strip())
        return data


    def get_examples(self, input, featurizer, rules=False, limit=-1): #sentences,
        logger.info("LOOKING AT {}".format(input))

        sentences = [s for s in sent_tokenize(input)]

        features = []
        for sentence in sentences:
            tokens, slot_labels, label = extract_annotated_labels_adhoc(sentence)
            f = featurizer(tokens, slot_labels, label)
            # put slot_labels, sentence_label to f
            features.append(f)
        data = features

        if rules:
            data = self.explore_rules(data,rules)

        if limit > 0:
            data = data[:limit]

        if data is not None and len(data) > 0:
            return self._create_examples(data=data), data
        else:
            return [],[]




class AnnotationProcessor(object):
    """Processor for the DefMiner data set """

    def __init__(self, args, task):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)
        self.pos_labels = get_pos_labels(args)
        self.target_task = task
        self.type="annotation"

    def _create_examples(self, data, set_type="annotation"): #intents, slots,
        """Creates examples for the training and dev sets."""
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)

            words = d['tokens']
            # intent_label = 0
            # slot_labels = [0] * len(words)
            # intent_label = d['label']
            # slot_labels = d['labels']

            # 2. intent
            intent_label = self.intent_labels.index(d['label']) if d['label'] in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            slot_labels = []
            for s in d['labels']:
                assert s in self.slot_labels
                slot_labels.append(self.slot_labels.index(s))

            assert len(words) == len(slot_labels)

            pos_labels = [self.pos_labels.index(s) if s in self.pos_labels else 0 for s in d['pos']]
            np_labels = d['np']
            vp_labels = d['vp']
            entity_labels = d['entities']
            acronym_labels = d['acronym']

            examples.append(InputExample(guid=guid,
                                         type=self.type,
                                         mode=set_type,
                                         words=words, intent_label=intent_label, slot_labels=slot_labels,
                                         pos_labels=pos_labels,
                                         np_labels=np_labels, vp_labels=vp_labels,
                                         entity_labels=entity_labels, acronym_labels=acronym_labels
                                         ))
        return examples

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data = []
        with open(input_file, encoding="utf-8") as infile:
            for line in infile:
                data.append(line.strip())
        return data


    def get_examples(self, input, rules=False, limit=-1): #sentences,
        if input is None:
            print("Empty annotation input")
            return

        logger.info("LOOKING AT {}".format(len(input)))

        data = []
        for id, ann in input.items():
            done = ann["data"]["raw"]
            slot_labels, intent_label = extract_annotated_labels(done["tokens"], ann["term"],ann["definition"])
            done["labels"] = slot_labels
            done["label"] = intent_label #"definition" #intent_label
            #TODO FIXME for negative samples
            data.append(done)

        # rule exploration/bootstrapping
        if rules:
            data = self.explore_rules(data,rules)

        if limit > 0:
            data = data[:limit]

        if data is not None and len(data) > 0:
            return self._create_examples(data=data), data
        else:
            return [],[]



def convert_examples_to_features_hybrid(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    truncated_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        processed_words = []
        tokens = []
        slot_labels_ids = {}
        pos_labels_ids = []
        np_labels_ids, vp_labels_ids, entity_labels_ids, acronym_labels_ids = [],[],[],[]
        for word, pos_label, np_label, vp_label, entity_label, acronym_label in zip(example.words, example.pos_labels, example.np_labels, example.vp_labels, example.entity_labels, example.acronym_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            if len(tokens) + len(word_tokens) > max_seq_len - 2:
                break
            processed_words.append(word)
            tokens.extend(word_tokens)
            pos_labels_ids.extend([int(pos_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            np_labels_ids.extend([int(np_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            vp_labels_ids.extend([int(vp_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            entity_labels_ids.extend([int(entity_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            acronym_labels_ids.extend([int(acronym_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        truncated_examples.append(processed_words)
        for data_type in example.slot_labels:
            slot_labels_ids[data_type] = []
            for word, slot_label in zip(processed_words, example.slot_labels[data_type]):
                word_tokens = tokenizer.tokenize(word)
                slot_labels_ids[data_type].extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            pos_labels_ids = pos_labels_ids[:(max_seq_len - special_tokens_count)]

            np_labels_ids = np_labels_ids[:(max_seq_len - special_tokens_count)]
            vp_labels_ids = vp_labels_ids[:(max_seq_len - special_tokens_count)]
            entity_labels_ids = entity_labels_ids[:(max_seq_len - special_tokens_count)]
            acronym_labels_ids = acronym_labels_ids[:(max_seq_len - special_tokens_count)]
            for data_type in example.slot_labels:
                slot_labels_ids[data_type] = slot_labels_ids[data_type][:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        for data_type in example.slot_labels:
            slot_labels_ids[data_type] += [pad_token_label_id]
        pos_labels_ids += [pad_token_label_id]
        np_labels_ids += [pad_token_label_id]
        vp_labels_ids += [pad_token_label_id]
        entity_labels_ids += [pad_token_label_id]
        acronym_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        for data_type in example.slot_labels:
            slot_labels_ids[data_type] = [pad_token_label_id] + slot_labels_ids[data_type]
        pos_labels_ids = [pad_token_label_id] + pos_labels_ids
        np_labels_ids = [pad_token_label_id] + np_labels_ids
        vp_labels_ids = [pad_token_label_id] + vp_labels_ids
        entity_labels_ids = [pad_token_label_id] + entity_labels_ids
        acronym_labels_ids = [pad_token_label_id] + acronym_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        for data_type in example.slot_labels:
            slot_labels_ids[data_type] = slot_labels_ids[data_type] + ([pad_token_label_id] * padding_length)

        pos_labels_ids = pos_labels_ids + ([pad_token_label_id] * padding_length)
        np_labels_ids = np_labels_ids + ([pad_token_label_id] * padding_length)
        vp_labels_ids = vp_labels_ids + ([pad_token_label_id] * padding_length)
        entity_labels_ids = entity_labels_ids + ([pad_token_label_id] * padding_length)
        acronym_labels_ids = acronym_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        for data_type in example.slot_labels:
            assert len(slot_labels_ids[data_type]) == max_seq_len, "Error with termdef slot labels length {} vs {}".format(len(slot_labels_ids[data_type]), max_seq_len)
        assert len(pos_labels_ids) == max_seq_len, "Error with pos labels length {} vs {}".format(len(pos_labels_ids), max_seq_len)
        assert len(np_labels_ids) == max_seq_len, "Error with np labels length {} vs {}".format(len(np_labels_ids), max_seq_len)
        assert len(vp_labels_ids) == max_seq_len, "Error with vp labels length {} vs {}".format(len(vp_labels_ids), max_seq_len)
        assert len(entity_labels_ids) == max_seq_len, "Error with entity labels length {} vs {}".format(len(entity_labels_ids), max_seq_len)
        assert len(acronym_labels_ids) == max_seq_len, "Error with acronym labels length {} vs {}".format(len(acronym_labels_ids), max_seq_len)

        intent_label_id = {}
        for data_type in example.intent_label:
            intent_label_id[data_type] = int(example.intent_label[data_type])

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

            for data_type in example.intent_label:
                logger.info("%s_intent_label: %s (id = %d)" % (data_type, example.intent_label[data_type], intent_label_id[data_type]))
                logger.info("%s_slot_labels: %s" % (data_type, " ".join([str(x) for x in slot_labels_ids[data_type]])))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids,
                          pos_labels_ids=pos_labels_ids,
                          np_labels_ids=np_labels_ids,
                          vp_labels_ids=vp_labels_ids,
                          entity_labels_ids=entity_labels_ids,
                          acronym_labels_ids=acronym_labels_ids))
    return features, truncated_examples





def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    truncated_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Converting example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        processed_words = []
        tokens = []
        slot_labels_ids = []
        pos_labels_ids = []
        np_labels_ids, vp_labels_ids, entity_labels_ids, acronym_labels_ids = [],[],[],[]
        for word, slot_label, pos_label, np_label, vp_label, entity_label, acronym_label in zip(example.words, example.slot_labels,  example.pos_labels, example.np_labels, example.vp_labels, example.entity_labels, example.acronym_labels):

            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            
            if len(tokens) + len(word_tokens) > max_seq_len - 2:
                break
            processed_words.append(word)
            tokens.extend(word_tokens)

            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            pos_labels_ids.extend([int(pos_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

            np_labels_ids.extend([int(np_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            vp_labels_ids.extend([int(vp_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            entity_labels_ids.extend([int(entity_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            acronym_labels_ids.extend([int(acronym_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        truncated_examples.append(processed_words)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]
            pos_labels_ids = pos_labels_ids[:(max_seq_len - special_tokens_count)]

            np_labels_ids = np_labels_ids[:(max_seq_len - special_tokens_count)]
            vp_labels_ids = vp_labels_ids[:(max_seq_len - special_tokens_count)]
            entity_labels_ids = entity_labels_ids[:(max_seq_len - special_tokens_count)]
            acronym_labels_ids = acronym_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        pos_labels_ids += [pad_token_label_id]
        np_labels_ids += [pad_token_label_id]
        vp_labels_ids += [pad_token_label_id]
        entity_labels_ids += [pad_token_label_id]
        acronym_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        pos_labels_ids = [pad_token_label_id] + pos_labels_ids
        np_labels_ids = [pad_token_label_id] + np_labels_ids
        vp_labels_ids = [pad_token_label_id] + vp_labels_ids
        entity_labels_ids = [pad_token_label_id] + entity_labels_ids
        acronym_labels_ids = [pad_token_label_id] + acronym_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)
        pos_labels_ids = pos_labels_ids + ([pad_token_label_id] * padding_length)

        np_labels_ids = np_labels_ids + ([pad_token_label_id] * padding_length)
        vp_labels_ids = vp_labels_ids + ([pad_token_label_id] * padding_length)
        entity_labels_ids = entity_labels_ids + ([pad_token_label_id] * padding_length)
        acronym_labels_ids = acronym_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)
        assert len(pos_labels_ids) == max_seq_len, "Error with pos labels length {} vs {}".format(len(pos_labels_ids), max_seq_len)
        assert len(np_labels_ids) == max_seq_len, "Error with np labels length {} vs {}".format(len(np_labels_ids), max_seq_len)
        assert len(vp_labels_ids) == max_seq_len, "Error with vp labels length {} vs {}".format(len(vp_labels_ids), max_seq_len)
        assert len(entity_labels_ids) == max_seq_len, "Error with entity labels length {} vs {}".format(len(entity_labels_ids), max_seq_len)
        assert len(acronym_labels_ids) == max_seq_len, "Error with acronym labels length {} vs {}".format(len(acronym_labels_ids), max_seq_len)


        intent_label_id = int(example.intent_label)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids,
                          pos_labels_ids=pos_labels_ids,
                          np_labels_ids=np_labels_ids,
                          vp_labels_ids=vp_labels_ids,
                          entity_labels_ids=entity_labels_ids,
                          acronym_labels_ids=acronym_labels_ids))


    return features, truncated_examples



def load_and_cache_examples(args, tokenizer, mode, model_name,
                            featurizer=None,limit=-1,
                            input=None, rules=False):
    processor_list = []
    task_list = []
    if input is not None:
        task = args.dataset # should be "input" for "feature" or "annotation"
        task_list.append(task)
        processor_list.append(processors[task](args, task))
        convertor = convert_examples_to_features
    else:
        if '+' in args.dataset:
            for task in args.dataset.split('+'):
                task_list.append(task.split('_')[0])
                processor_list.append(processors[task](args, task, hybrid=True))
            convertor = convert_examples_to_features_hybrid
        else:
            task_list.append(args.dataset.split('_')[0])
            for task in task_list:
                processor_list.append(processors[task](args, task))
            convertor = convert_examples_to_features

    logger.info("List of tasks: {}".format(str(task_list)))
    logger.info("List of processors: {}".format(str(processor_list)))

    if limit == -1:
        limit = args.data_limit if args.data_limit > 0 else -1

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            "+".join(task_list),
            list(filter(None, model_name.split("/"))).pop(),
            args.max_seq_len
        )
    )
    cached_data_file = os.path.join(
        args.data_dir,
        'cached_data_{}_{}_{}_{}.pkl'.format(
            mode,
            "+".join(task_list),
            list(filter(None, model_name.split("/"))).pop(),
            args.max_seq_len
        )
    )

    raw_data_for_tasks = None
    if os.path.exists(cached_features_file) and os.path.exists(cached_data_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        logger.info("Loading raw data from cached file %s", cached_data_file)
        raw_data_for_tasks = pkl.load(open(cached_data_file,"rb"))
    else:
        examples_for_tasks = []
        raw_data_for_tasks = []
        for processor, task in zip(processor_list, task_list):
            # Load data features from dataset file
            if task == "input":
                logger.info("Creating features from input: %s (%s)",task,str(input))
                examples, raw_data = processor.get_examples(input,featurizer=featurizer,rules=rules,limit=limit)
            elif task == "feature" or mode == "feature":
                logger.info("Creating features from feature: {} ({})".format(task,len(input)))
                examples, raw_data = processor.get_examples(input,rules=rules,limit=limit)
            elif task == "annotation" or mode == "annotation":
                logger.info("Creating features from annotations: {} ({})".format(task,len(input)))
                examples, raw_data = processor.get_examples(input,rules=rules,limit=limit)
            elif task == "s2orc":
                # Load data features from dataset file
                logger.info("Creating features from dataset file at %s", args.data_dir)
                if mode == "unlabeled":
                    examples, raw_data = processor.get_examples("unlabeled",rules=rules,limit=limit)
                else:
                    raise Exception("For mode, Only unlabeled is available for s2orc", mode)
            elif mode in ["train", "dev", "test"]:
                # Load data features from dataset file
                logger.info("Creating features from dataset file at %s for %s task",
                            args.data_dir, task)
                if mode == "train":
                    examples, raw_data = processor.get_examples("train",rules=rules,limit=limit)
                elif mode == "dev":
                    examples, raw_data = processor.get_examples("dev",rules=rules,limit=limit)
                elif mode == "test":
                    examples, raw_data = processor.get_examples("test",rules=rules,limit=limit)
                    if args.eval_data_file is not None:
                        examples, raw_data = processor.get_examples(args.eval_data_file,limit=limit)
                    else:
                        examples,raw_data = processor.get_examples("test",limit=limit)
                else:
                    raise Exception("For mode, Only train, dev, test is available")
            else:
                raise Exception("For mode, Only train, dev, test is available")

            examples_for_tasks += examples
            raw_data_for_tasks += raw_data

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features, _ = convertor(examples_for_tasks,
                             args.max_seq_len,
                             tokenizer,
                             pad_token_label_id=pad_token_label_id)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        logger.info("Saving raw_data into cached file %s", cached_data_file)
        pkl.dump(raw_data_for_tasks, open(cached_data_file,"wb"))




    # print(len(features))
    save_to_csv_for_inspection(features, raw_data_for_tasks)
    # with open("mode.csv","w",newline="") as f:  # python 2: open("output.csv","wb")
        # title = "time,SOURCE,PLACE,TEMP,LIGHT,HUMIDITY".split(",") # quick hack
    # cw = csv.DictWriter(f,title,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # cw.writeheader()
    # cw.writerows(data)





    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if '+' in args.dataset:
        all_intent_label_ids = []
        all_slot_labels_ids = []
        for data_type in args.dataset.split('+'):
            all_intent_label_ids.append([f.intent_label_id[data_type] for f in features])
            all_slot_labels_ids.append([f.slot_labels_ids[data_type] for f in features])
        all_intent_label_ids = torch.tensor(all_intent_label_ids).permute(1,0)
        all_slot_labels_ids = torch.tensor(all_slot_labels_ids).permute(1,2,0)
    else:
        all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
        all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    all_pos_labels_ids = torch.tensor([f.pos_labels_ids for f in features], dtype=torch.long)
    all_np_labels_ids = torch.tensor([f.np_labels_ids for f in features], dtype=torch.float)
    all_vp_labels_ids = torch.tensor([f.vp_labels_ids for f in features], dtype=torch.float)
    all_entity_labels_ids = torch.tensor([f.entity_labels_ids for f in features], dtype=torch.float)
    all_acronym_labels_ids = torch.tensor([f.acronym_labels_ids for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_slot_labels_ids,
                            all_pos_labels_ids,
                            all_np_labels_ids,
                            all_vp_labels_ids,
                            all_entity_labels_ids,
                            all_acronym_labels_ids,)
    return dataset, raw_data_for_tasks



def save_to_csv_for_inspection(features, raw_data):
    fout = open("output.txt","w")
    cnt = 0
    for idx, (f,r) in enumerate(zip(features, raw_data)):
        if np.sum(r["acronym"]) > 1 and r["name"]!="AI2020":
            fout.write("{}-{}-{}\t{}\n".format(idx,r["name"],r["label"], "\t".join(r["tokens"])))
            fout.write("\t{}\n".format("\t".join([str(vone) for vone in r["acronym"]])))
            for k,v in f.slot_labels_ids.items():
                fout.write("{}-{}\t{}\n".format(k, f.intent_label_id[k], "\t".join([str(vone) for vone in v])))
            fout.write("\n")
            cnt += 1

        # if cnt > 1000:
        #     break
    fout.close()




def load_and_cache_example_batch_raw(args, tokenizer, data, pos_labels, ignore_index=0):
    """
    In relation to 'load_and_cache_example_batch', this function also returns the raw
    data (along with the tensor dataset). Raw data is needed in the filtering heuristics
    stage. Some logic may be duplicated across these two functions.
    """
    examples = []
    for d in data:
        #create fake intent and slot labels for the datasets
        intent_label= {}
        slot_labels= {}  # fake slot labels
        if '+' in args.task:
            for data_type in args.task.split('+'):
                intent_label[data_type]=1  # fake intent label
                slot_labels[data_type]= [1] * len(d["tokens"])  # fake slot labels
        else:
            intent_label=1  # fake intent label
            slot_labels= [1] * len(d["tokens"])  # fake slot labels

        example = InputExample(
            guid="one",
            type = args.task,
            mode = "symnick",
            words=d["tokens"],
            intent_label=intent_label,  # fake intent label
            slot_labels= slot_labels,  # fake slot labels
            pos_labels=[
                pos_labels.index(s) if s in pos_labels else 0
                for s in d["pos"]
            ],
            np_labels=d["np"],
            vp_labels=d["vp"],
            entity_labels=d["entities"],
            acronym_labels=d["abbreviation"],
        )
        examples.append(example)
    
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    if '+' in args.task:
        features, truncated_examples = convert_examples_to_features_hybrid(
            examples, args.max_seq_len, tokenizer, pad_token_label_id=ignore_index
        )
    else:
        features, truncated_examples = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer, pad_token_label_id=ignore_index
        )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )

    # print(features)

    if '+' in args.task:
        all_intent_label_ids = []
        all_slot_labels_ids = []
        for data_type in args.task.split('+'):
            all_intent_label_ids.append([f.intent_label_id[data_type] for f in features])
            all_slot_labels_ids.append([f.slot_labels_ids[data_type] for f in features])
        all_intent_label_ids = torch.tensor(all_intent_label_ids).permute(1,0)
        all_slot_labels_ids = torch.tensor(all_slot_labels_ids).permute(1,2,0)
    else:
        all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
        all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    all_pos_labels_ids = torch.tensor(
        [f.pos_labels_ids for f in features], dtype=torch.long
    )

    all_np_labels_ids = torch.tensor(
        [f.np_labels_ids for f in features], dtype=torch.float
    )
    all_vp_labels_ids = torch.tensor(
        [f.vp_labels_ids for f in features], dtype=torch.float
    )
    all_entity_labels_ids = torch.tensor(
        [f.entity_labels_ids for f in features], dtype=torch.float
    )
    all_acronym_labels_ids = torch.tensor(
        [f.acronym_labels_ids for f in features], dtype=torch.float
    )

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_intent_label_ids,
        all_slot_labels_ids,
        all_pos_labels_ids,
        all_np_labels_ids,
        all_vp_labels_ids,
        all_entity_labels_ids,
        all_acronym_labels_ids,
    )
    return dataset, truncated_examples

processors = {
    "W00": DefProcessor,
    "DEFT": DefProcessor,
    "WFM": DefProcessor,
    "DocDef": DefProcessor,
    "DocDef2": DefProcessor,
    "DocDef2MIA": DefProcessor,
    "DocDefQueryInplace": DefProcessor,
    "DocDefQueryInplaceMIA": DefProcessor,
    "DocDefQueryInplaceFixed": DefProcessor,
    "DocDefQueryInplaceFixedMIA": DefProcessor,
    "s2orc": DefProcessor,
    "input": InputProcessor,
    "AI2020": DefProcessor,
    "feature": FeatureProcessor,
    "sampled": FeatureProcessor,
    "annotation": AnnotationProcessor
}


