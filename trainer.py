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
import json

from heuristic_filters import heuristic_filters
from eval_utils import compute_metrics, compute_metrics_for_sentence_classification, compute_metrics_for_slot_tagging
from utils import highlight, colorize_labels, colorize_term_definition, match_tokenized_to_untokenized, CharacterRange,  replace_latex_notation_to_SYMBOL, mapping_feature_tokens_to_original, sanity_check_for_acronym_detection,transpose_dict_to_list,  merge_slot_predictions
from symbol_rule_detector import get_symbol_nickname_pairs
from abbreviation_detector import get_abbreviation_pairs
logger = logging.get_logger()

class Trainer(object):
    def __init__(
        self,
        args: List[Any],
        model: Any,
        train_dataset: Optional[TensorDataset] = None,
        dev_dataset: Optional[TensorDataset] = None,
        test_dataset: Optional[TensorDataset] = None,
        slot_label_lst: Optional[Dict[Any,Any]] = None,
        tokenizer = None
    ) -> None:

        self.args, self.model_args, self.data_args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = self.args.ignore_index  # 0 #tokenizer.pad_token_id #

        self.tokenizer = tokenizer

        #NOTE later, replace list to dictionary for every cases
        self.slot_label_lst = getattr(model, "slot_label_lst", None)
        self.pos_label_lst = getattr(model, "pos_label_lst", None)
        # self.slot_label_dict = getattr(model, "slot_label_dict", None)
        # self.intent_label_dict = getattr(model, "intent_label_dict", None)
        self.pos_label_lst = model.pos_label_lst

        self.model = model

        # GPU or CPU
        self.device = (
            "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        self.model.to(self.device)

        self.nlp = spacy.load("en_core_sci_md")
        abbreviation_pipe = AbbreviationDetector(self.nlp)
        self.nlp.add_pipe(abbreviation_pipe)

    def train(self, input_dataset=False):
        if input_dataset:
            train_sampler = RandomSampler(input_dataset)
            train_dataloader = DataLoader(input_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            train_sampler = RandomSampler(self.train_dataset)
            train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = {}".format( highlight(len(input_dataset if input_dataset else self.train_dataset))))
        logger.info("  Num Epochs = {}".format( highlight(self.args.num_train_epochs)))
        logger.info("  Total train batch size = {}".format( highlight(self.args.train_batch_size)))
        logger.info("  Gradient Accumulation steps = {}".format( highlight(self.args.gradient_accumulation_steps)))
        logger.info("  Total optimization steps = {}".format( highlight(t_total)))
        logger.info("  Logging steps = {}".format( highlight(self.args.logging_steps)))
        logger.info("  Save steps = {}".format( highlight(self.args.save_steps)))

        global_step = 0
        tr_loss = 0.0
        dev_score_history, dev_step_history = [], []
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}

                if self.model.config.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]

                if self.args.use_pos:
                    inputs['pos_label_ids'] = batch[5]

                if self.args.use_np:
                    inputs['np_label_ids'] = batch[6]
                if self.args.use_vp:
                    inputs['vp_label_ids'] = batch[7]
                if self.args.use_entity:
                    inputs['entity_label_ids'] = batch[8]
                if self.args.use_acronym:
                    inputs['acronym_label_ids'] = batch[9]

                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                epoch_iterator.set_description("step {}/{} loss={:.2f}".format(
                        step, global_step, tr_loss / (global_step+1)
                    ))

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0 and not input_dataset:
                        if self.args.use_test_set_for_validation:
                            results, _ = self.evaluate("test")
                        else:
                            results, _ = self.evaluate("dev")



                        result_to_save = {'model':self.model_args.model_name_or_path,
                                       'use_pos':self.args.use_pos,
                                       'use_np':self.args.use_np,
                                       'use_vp':self.args.use_vp,
                                       'use_entity':self.args.use_entity,
                                       'use_acronym':self.args.use_acronym,
                                       'global_step':global_step }

                        for k,v in results.items():
                            result_to_save[k] = v

                        # save model
                        # FIXME
                        dev_score = result_to_save['paired_results'][0]["results"]['slot_f1_macro']

                        if global_step == self.args.logging_steps  or dev_score > max(dev_score_history):
                            self.save_model()
                            # self.copy_best_model()
                            logger.info(" ******* new best model saved at step {}: {}".format(highlight(global_step), highlight(dev_score)))

                        dev_score_history += [dev_score]
                        dev_step_history += [global_step]
                        result_to_save['best_slot_f1_macro'] = max(dev_score_history)
                        result_to_save['best_global_step'] = dev_step_history[dev_score_history.index(result_to_save['best_slot_f1_macro'])]

                        # save log
                        filename = 'logs/logs_train_{}_{}.txt'.format(self.data_args.kfold, self.model_args.model_name_or_path)
                        if not os.path.exists(os.path.dirname(filename)):
                            os.makedirs(os.path.dirname(filename))
                        with open(filename,'a') as f:
                            if self.data_args.kfold == 0:
                                f.write('{}\n'.format('\t'.join(result_to_save.keys())))
                            f.write('{}\n'.format('\t'.join([str(v) for v in result_to_save.values()])))

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step




    def neural_inference(self, dataset, raw_data=[], verbose=False, save_result=False, label_given=True, return_embedding=False, do_visualize=False):
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info("  Num examples = {}".format( highlight(len(dataset))))
        logger.info("  Batch size = {}".format( highlight(self.args.eval_batch_size)))

        eval_loss = 0.0
        nb_eval_steps = 0
        input_ids_all = None
        pos_ids_all = None
        intent_preds = None
        slot_preds = None
        slot_conf = None
        gold_intent_label_ids = None
        gold_slot_labels_ids = None
        sequence_outputs = None
        pooled_outputs = None

        self.model.eval()

        logger.info("Start model predicion")
        for batch in tqdm(eval_dataloader, desc="Neural Inference"):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                if self.model.config.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                if self.args.use_pos:
                    inputs['pos_label_ids'] = batch[5]
                if self.args.use_np:
                    inputs['np_label_ids'] = batch[6]
                if self.args.use_vp:
                    inputs['vp_label_ids'] = batch[7]
                if self.args.use_entity:
                    inputs['entity_label_ids'] = batch[8]
                if self.args.use_acronym:
                    inputs['acronym_label_ids'] = batch[9]

                outputs = self.model(**inputs)

                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                sequence_output, pooled_output = None, None
                if len(outputs) > 2:
                    sequence_output = outputs[2]
                    pooled_output = outputs[3]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Input
            if input_ids_all is None:
                input_ids_all = inputs["input_ids"].detach().cpu().numpy()
            else:
                input_ids_all = numpy.append(input_ids_all, inputs["input_ids"].detach().cpu().numpy(), axis=0)

            # Sequence ouptut
            if sequence_outputs is None:
                sequence_outputs = sequence_output.detach().cpu().numpy()
            else:
                sequence_outputs = numpy.append(sequence_outputs, sequence_output.detach().cpu().numpy(), axis=0)
            # Pooled ouptut
            if pooled_outputs is None:
                pooled_outputs = pooled_output.detach().cpu().numpy()
            else:
                pooled_outputs = numpy.append(pooled_outputs, pooled_output.detach().cpu().numpy(), axis=0)

            # POS
            if pos_ids_all is None:
                pos_ids_all = inputs["pos_label_ids"].detach().cpu().numpy()
            else:
                pos_ids_all = numpy.append(pos_ids_all, inputs["pos_label_ids"].detach().cpu().numpy(), axis=0)

            # Intent prediction
            intent_probs = torch.softmax(intent_logits, dim=1).detach().cpu().numpy()
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                gold_intent_label_ids = (
                    inputs["intent_label_ids"].detach().cpu().numpy()
                )
                intent_conf = intent_probs[:,1]
            else:
                intent_preds = numpy.append(
                    intent_preds, intent_logits.detach().cpu().numpy(), axis=0

                )
                gold_intent_label_ids = numpy.append(
                    gold_intent_label_ids,
                    inputs["intent_label_ids"].detach().cpu().numpy(),
                    axis=0,
                )
                intent_conf = numpy.append(
                    intent_conf, intent_probs[:,1], axis=0
                )

            # Slot prediction
            slot_probs = torch.softmax(slot_logits,dim=2).detach().cpu().numpy()
            if slot_preds is None:
                if self.args.use_crf:
                    decode_out = self.model.crf.decode(slot_logits)
                    slot_logits_crf = numpy.array(decode_out)
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = slot_logits_crf
                    # get confidence from softmax
                    I,J = numpy.ogrid[:slot_logits_crf.shape[0], :slot_logits_crf.shape[1]]
                    slot_conf = slot_probs[I, J, slot_logits_crf]
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                gold_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_logits_crf = numpy.array(self.model.crf.decode(slot_logits))
                    slot_preds = numpy.append(slot_preds, slot_logits_crf, axis=0)
                    # get confidence from softmax
                    I,J = numpy.ogrid[:slot_logits_crf.shape[0], :slot_logits_crf.shape[1]]
                    slot_conf = numpy.append(slot_conf, slot_probs[I, J, slot_logits_crf], axis=0)
                else:
                    slot_preds = numpy.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                gold_slot_labels_ids = numpy.append(gold_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)


        if nb_eval_steps == 0:
            return [], {}

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = numpy.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            # get confidence from softmax
            I,J = numpy.ogrid[:slot_preds.shape[0], :slot_preds.shape[1]]
            slot_conf = slot_preds[I, J, numpy.argmax(slot_preds, axis=2)]
            slot_preds = numpy.argmax(slot_preds, axis=2)

        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        slot_label_to_idx_map = {label: i for i, label in enumerate(self.slot_label_lst)}
        pos_label_map = {i: label for i, label in enumerate(self.pos_label_lst)}
        pos_label_to_idx_map = {label: i for i, label in enumerate(self.pos_label_lst)}


        gold_slot_num_batch = int(len(gold_slot_labels_ids))
        gold_slot_num_length = int(len(gold_slot_labels_ids[0]))
        slot_labels_list = [[] for _ in range(gold_slot_num_batch)]
        slot_preds_list = [[] for _ in range(gold_slot_num_batch)]
        slot_conf_list = [[] for _ in range(gold_slot_num_batch)]
        input_ids_list = [[] for _ in range(gold_slot_num_batch)]
        pos_tags_list = [[] for _ in range(gold_slot_num_batch)]

        id_mapping = defaultdict(list)
        for i in range(gold_slot_num_batch):
            prev_input_ids = []
            for j in range(gold_slot_num_length):
                if label_given:
                    # labels (ignore subword labels and special tokens)
                    if gold_slot_labels_ids[i, j] != self.pad_token_label_id:
                        slot_labels_list[i].append(slot_label_map[gold_slot_labels_ids[i][j]])
                        slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
                        slot_conf_list[i].append(slot_conf[i][j])
                        pos_tags_list[i].append(pos_label_map[pos_ids_all[i][j]])

                    # input subword concatenation
                    if gold_slot_labels_ids[i, j] != self.pad_token_label_id:
                        input_ids_list[i].append([input_ids_all[i][j]])
                    else:
                        if j > 0 and input_ids_all[i][j] > 2:
                            input_ids_list[i][-1] = input_ids_list[i][-1] + [input_ids_all[i][j]]
                else:
                    # labels (ignore subword labels and special tokens)
                    if pos_ids_all[i, j] != self.pad_token_label_id:
                        slot_labels_list[i].append(slot_label_map[gold_slot_labels_ids[i][j]])
                        slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
                        slot_conf_list[i].append(slot_conf[i][j])
                        pos_tags_list[i].append(pos_label_map[pos_ids_all[i][j]])

                    # input subword concatenation
                    if pos_ids_all[i, j] != self.pad_token_label_id:
                        input_ids_list[i].append([input_ids_all[i][j]])
                    else:
                        try:
                            if j > 0 and input_ids_all[i][j] > 2:
                                input_ids_list[i][-1] = input_ids_list[i][-1] + [input_ids_all[i][j]]
                        except Exception as e:
                            print(e)
                            print(input_ids_all)
                            continue

        # convert input ids to tokens
        input_tokens_list = []
        for input_ids in input_ids_list:
            input_tokens = []
            for tset in input_ids:
                # print(self.tokenizer.convert_ids_to_tokens(tset))
                input_tokens.append("".join(self.tokenizer.convert_ids_to_tokens(tset)))
            input_tokens_list.append(input_tokens)

        #TODO different heuristic filters for each type
        # heuristics (gold labels are in fact not used)
        if self.args.use_heuristic:
            intent_preds, slot_preds_list = heuristic_filters(
                input_tokens_list,
                intent_preds,
                gold_intent_label_ids,
                slot_preds_list,
                slot_labels_list)

        # only input information processed
        inputs_dict = {
            "input_tokens": input_tokens_list,
            "pos": pos_tags_list,
            "intent_labels": gold_intent_label_ids,
            "slot_labels": slot_labels_list,
        }
        inputs_list = transpose_dict_to_list(inputs_dict)

        # neural-based definition detector
        neural_predictions = {
            "intent_preds":intent_preds,
            "intent_conf": intent_conf,
            "slot_preds": slot_preds_list,
            "slot_conf": slot_conf_list,
            "pooled_outputs": pooled_outputs,
            "sequence_outputs": sequence_outputs
        }
        prediction_list = transpose_dict_to_list(neural_predictions)
        # we keep having the multi term-def slot predictions
        # NOTE after evaluation, we make them separated for frontend
        # prediction_list = [[p] for p in prediction_list]
        return inputs_list, prediction_list


    def acronym_inference(self, input_tokens_list):
        """Off-the-shelf acronym detector"""
        input_iterator = tqdm(input_tokens_list, desc="Acrnonym Inference", total=len(input_tokens_list))
        prediction_list = []
        for didx, input_tokens in enumerate(input_iterator):
            #  one input instance can have multiple predictions
            prediction_list_per_input = []
            for abbr_labels in get_abbreviation_pairs(input_tokens, self.nlp):
                prediction_list_per_input.append({
                    "slot_preds": abbr_labels})
            # if no positive labels are given, add empty [O O ..] labels
            if len(prediction_list_per_input) == 0:
                prediction_list_per_input.append(
                    {"slot_preds": ["O"]*len(input_tokens)})
            prediction_list.append(prediction_list_per_input)

            # Combine multi term-def pairs to merged one.
            slot_preds_list = [p["slot_preds"] for p in prediction_list_per_input]
            merged_slot_preds_list = merge_slot_predictions(slot_preds_list,
                                            merge_method="union",
                                            term_postfix="short",
                                            def_postfix="long")
            merged_prediction_list_per_input = {
                    "slot_preds":merged_slot_preds_list}
            prediction_list.append(merged_prediction_list_per_input)
        return prediction_list

    def nickname_inference(self, input_tokens_list, pos_list):
        """Rule-based symbol nickname detector"""
        assert len(pos_list) == len(input_tokens_list)
        input_iterator = tqdm(zip(input_tokens_list,pos_list), desc="Nickname Inference", total=len(input_tokens_list))
        prediction_list = []
        for didx, (input_tokens, pos) in enumerate(input_iterator):
            #  one input instance can have multiple predictions
            prediction_list_per_input = []
            for nick_labels in get_symbol_nickname_pairs(input_tokens, pos):
                prediction_list_per_input.append({
                    "slot_preds": nick_labels})
            # if no positive labels are given, add empty [O O ..] labels
            if len(prediction_list_per_input) == 0:
                prediction_list_per_input.append(
                    {"slot_preds": ["O"]*len(input_tokens)})

            # Combine multi term-def pairs to merged one.
            slot_preds_list = [p["slot_preds"] for p in prediction_list_per_input]
            merged_slot_preds_list = merge_slot_predictions(slot_preds_list,
                                            merge_method="union",
                                            term_postfix="TERM",
                                            def_postfix="DEF")
            merged_prediction_list_per_input = {
                    "slot_preds":merged_slot_preds_list}

            prediction_list.append(merged_prediction_list_per_input)
        return prediction_list


    def evaluate(self,
                 mode=None,
                 dataset=None,
                 verbose=False,
                 save_result=False,
                 label_given=True,
                 do_visualize=False):
        if mode == 'test':
            dataset,raw_data = self.test_dataset
        elif mode == 'dev':
            dataset,raw_data = self.dev_dataset
        elif model is None and dataset is not None:
            raw_data = None
            # TODO convert latex $XX$ to SYMBOL
        else:
            raise Exception("Only dev and test or input dataset available")

        logger.info("***** Running inference on %s dataset *****", mode)
        inputs_list, neural_predictions = self.neural_inference(
            dataset,raw_data,label_given=label_given)
        """the following systems below are not producing True Negative outputs though"""
        acronym_predictions = []
        nickname_predictions = []
        if self.args.use_acronym_detector:
            """off-the-shelf acronym detector (Shwartz and Hearst)"""
            acronym_predictions = self.acronym_inference(
                [i["input_tokens"] for i in inputs_list]
            )
        if self.args.use_nickname_detector:
            """nickname detector requires POS information as well as input tokens"""
            nickname_predictions = self.nickname_inference(
                [i["input_tokens"] for i in inputs_list],
                [i["pos"] for i in inputs_list]
            )

        logger.info("***** Merging predictions on %s dataset *****", mode)
        logger.info(" Use abbreviation detector? %s (merge: %s)", self.args.use_acronym_detector, self.args.merge_predictions_for_acronym)
        logger.info(" Use nickname detector? %s (merge: %s)", self.args.use_nickname_detector, self.args.merge_predictions_for_symbol)

        merged_predictions = self.merge_predictions(
            raw_data,
            inputs_list,
            {
                "neural": neural_predictions,
                "acronym": acronym_predictions if self.args.use_acronym_detector else None,
                "nickname": nickname_predictions if self.args.use_nickname_detector else None
            },
            merge_acronym = self.args.merge_predictions_for_acronym,
            merge_symbol = self.args.merge_predictions_for_symbol,
            verbose=True,
        )

        logger.info("***** Running evaluation on %s dataset *****", mode)
        merged_results = self.compute_eval_metrics(
            merged_predictions,
            do_per_instance_eval = True,
            verbose = True
        )
        pprint(merged_results)

        if do_visualize:
            logger.info("***** Visualizing posterior dist. on %s dataset *****", mode)
            eval_label = [slot_label_to_idx_map[s] for sb in slot_preds_list for s in sb ]
            eval_prob = [s for sb in slot_conf_list for s in sb ]
            plot_posterior(eval_label, eval_prob)

        return merged_results, merged_predictions


    def compute_eval_metrics(self, predictions, do_per_instance_eval=False, verbose=False):
        results = {}

        """ per-instance evaluation first"""
        slot_labels_per_type_system_dict = defaultdict(list)
        slot_preds_per_type_system_dict = defaultdict(list)
        for p in predictions:
            term_type = p["raw_data"]["type"] if "type" in p["raw_data"] else None

            if do_per_instance_eval:
                p["results"] = {}
                result_per_instance_sentence_classification = compute_metrics_for_sentence_classification(
                    [p["outputs"]["intent"]["preds"]],
                    [p["inputs"]["intent_labels"]],
                    type_ = term_type
                )
                p["results"]["intent"] =  result_per_instance_sentence_classification
                # pprint(result_per_instance_sentence_classification)

                for system, pred_list in p["outputs"]["slot"].items():
                    # print(system, pred_list)
                    # To deal with multi term-def cases, merge list of predictions per instance here

                    result_per_instance_slot_tagging_list = []
                    result_per_instance_slot_tagging = compute_metrics_for_slot_tagging(
                        [pred_list['slot_preds']],
                        [p["inputs"]['slot_labels']]
                    )
                    # pprint(result_per_instance_slot_tagging)
                    result_per_instance_slot_tagging_list.append(result_per_instance_slot_tagging)
                    p["results"]["slot"] = {}
                    p["results"]["slot"][system] = result_per_instance_slot_tagging_list

            # aggregate slot predictions from multiple systems
            #   sytem: ["neural", "abbreviation", "nickname"]
            #   term_type: ["term", "acronym", "symbol"]
            for system, pred_list in p["outputs"]["slot"].items():
                # store slot labels/preds into each pair of term type and system
                if "type" in p["raw_data"]:
                    type_system_pair = (p["raw_data"]["type"], system)
                else:
                    type_system_pair = (system)
                slot_labels_per_type_system_dict[type_system_pair].append(
                    p["inputs"]["slot_labels"])
                slot_preds_per_type_system_dict[type_system_pair].append(
                    pred_list["slot_preds"])

        """ overall evaluation"""
        # calculate total intent evaluation metrics

        total_result_sentence_classification = compute_metrics_for_sentence_classification(
            [p["outputs"]["intent"]["preds"] for p in predictions],
            [p["inputs"]["intent_labels"] for p in predictions])
        results.update(total_result_sentence_classification)
        if verbose:
            pprint(total_result_sentence_classification)

        # calculate total slot evaluation metrics per each term type and each system
        type_system_results_list = []
        for type_system_pair in slot_labels_per_type_system_dict:
            per_type_system_result_slot_tagging = compute_metrics_for_slot_tagging(
                slot_preds_per_type_system_dict[type_system_pair],
                slot_labels_per_type_system_dict[type_system_pair]
            )
            type_system_results_list.append({
                "type_system":type_system_pair,
                "results": per_type_system_result_slot_tagging })
            if verbose:
                print(type_system_pair)
                pprint(per_type_system_result_slot_tagging)
                print("")
        results.update({"paired_results":type_system_results_list})
        return results

    def merge_predictions(self, raw_data, inputs_list, prediction_dict, merge_acronym=None, merge_symbol=None, verbose=False):
        merged_prediction_list = []
        for didx, (data, inputs) in enumerate(zip(raw_data, inputs_list)):
            merged_dict = {}
            merged_dict["didx"] = didx
            merged_dict["raw_data"] = data
            merged_dict["inputs"] = inputs
            merged_dict["outputs"] = {}

            intent_dict = {}
            intent_dict["preds"] = prediction_dict["neural"][didx]["intent_preds"]
            intent_dict["conf"] = prediction_dict["neural"][didx]["intent_conf"]
            merged_dict["outputs"]["intent"] = intent_dict

            # store list of slot predictions
            slot_dict = {}
            slot_dict["neural"] = prediction_dict["neural"][didx]
            if prediction_dict["acronym"] is not None:
                slot_dict["acronym"] = prediction_dict["acronym"][didx]
            if prediction_dict["nickname"] is not None:
                slot_dict["nickname"] = prediction_dict["nickname"][didx]

            merged_dict["outputs"]["slot"] = slot_dict


            #TODO do some output integration here
            if merge_symbol is not None and len(merged_dict["outputs"]["slot"]) > 1:
                systems_list = [s for s,p in merged_dict["outputs"]["slot"].items()]
                slot_preds_list = [p["slot_preds"] for s,p in merged_dict["outputs"]["slot"].items()]
                merged_slot_preds = merge_slot_predictions(slot_preds_list,
                                                merge_method=merge_symbol,
                                                term_postfix="TERM",
                                                def_postfix="DEF")

                merged_dict["outputs"]["slot"]["merged_nickname"] = {
                    "slot_preds": merged_slot_preds,
                    "merged_systems": systems_list,
                    "merge_method": merge_symbol }

            if merge_acronym is not None and len(merged_dict["outputs"]["slot"]) > 1:
                merged_dict["outputs"]["slot"]["merged_acronym"] = None

            merged_prediction_list.append(merged_dict)


            if verbose:
                print("=========={}===========".format(didx))
                gold_type = highlight("({})".format(data["type"])) \
                    if "type" in data else ""
                print("{}{}\t{}".format(
                    highlight("Gold".upper()),
                    gold_type,
                    colorize_term_definition(
                        merged_dict["inputs"]['input_tokens'],
                        merged_dict["inputs"]["slot_labels"])))

                print("Intent predicted: {} (gold: {})".format(
                        highlight(merged_dict["outputs"]["intent"]["preds"]),
                        highlight(merged_dict["inputs"]["intent_labels"])
                    ))

                for system, pdict in merged_dict["outputs"]["slot"].items():
                    # we don't output [O O O ] O only ensemble_predictions
                    if len(list(set(pdict["slot_preds"]))) == 1:
                        continue

                    print("{}\t{}".format(
                        highlight(system.upper()),
                        colorize_term_definition(
                            merged_dict["inputs"]['input_tokens'],
                            pdict["slot_preds"])))
                print("")

            # if "SYMBOL" in merged_dict["inputs"]['input_tokens']:
                # ctr = Counter(merged_dict["inputs"]['input_tokens'])
                # if ctr["SYMBOL"] > 1:
                    # print(merged_dict["inputs"]['input_tokens'])
            #         from pdb import set_trace; set_trace()

        return merged_prediction_list




    def save_model(self, dataconfig):
        # Save model checkpoint (Overwrite)
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", highlight(output_dir))
        with open(f"{output_dir}/dataconfig.json",'w') as fob:
            json.dump(dataconfig, fob)


    def copy_best_model(self, best_dir_name='checkpoint_best'):
        output_dir = self.args.output_dir
        best_dir = os.path.join(self.args.output_dir, best_dir_name)
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.makedirs(best_dir)

        files = (file for file in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, file)))
        for file in files:
            shutil.copy(os.path.join(output_dir,file), best_dir)






