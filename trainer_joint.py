import os
import sys
import shutil
import regex
import coloredlogs #, logging
import numpy
import torch
import re
from pprint import pprint
from colorama import Fore,Style
from tqdm import tqdm, trange
from collections import Counter,defaultdict
from typing import Any, List, Dict, Tuple, Optional, DefaultDict, Union
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, logging
# from transformers.file_utils import ENV_VARS_TRUE_VALUES
from dataclasses import dataclass
import wandb

from heuristic_filters import heuristic_filters, common_filters
from eval_utils import compute_metrics, compute_metrics_for_sentence_classification, compute_metrics_for_slot_tagging
from utils import highlight, colorize_labels, colorize_term_definition, match_tokenized_to_untokenized, CharacterRange,  replace_latex_notation_to_SYMBOL, mapping_feature_tokens_to_original, sanity_check_for_acronym_detection,transpose_dict_to_list,  merge_slot_predictions, transpose_dict_to_list_hybrid, NLP
from symbol_rule_detector import get_symbol_nickname_pairs
from abbreviation_detector import get_abbreviation_pairs
from trainer import Trainer
from data_loader import load_and_cache_example_batch_raw

logger = logging.get_logger()
logger.setLevel(logging.INFO)

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

term_label_postfix_map={
    "DocDef2": "TERM",
    "AI2020": "short",
    "W00": "TERM",
}
def_label_postfix_map={
    "DocDef2": "DEF",
    "AI2020": "long",
    "W00": "DEF",
}

class JointTrainer(Trainer):
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
        self.slot_label_dict = getattr(model, "slot_label_dict", None)
        self.intent_label_dict = getattr(model, "intent_label_dict", None)
        self.pos_label_lst = model.pos_label_lst
        self.model = model

        # GPU or CPU
        self.device = (
            "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        self.model.to(self.device)

        if os.getenv("WANDB_DISABLED", "").upper() not in ENV_VARS_TRUE_VALUES:
            wandb.init(project="heddex", entity="dongyeopk")
            wandb.watch(self.model)

        self.nlp = NLP()
        self.heuristic_filters = heuristic_filters

    def load_inputs_from_batch(self, batch):
        intent_label_ids = {}
        slot_label_ids = {}

        # [batch x task]
        if len(batch[3].shape) == 1:
            intent_label_ids_tensors = torch.unsqueeze(batch[3],-1).permute(1,0)
        else:
            intent_label_ids_tensors = batch[3].permute(1,0)
        # [batch x length x task]
        if len(batch[4].shape) == 2:
            slot_label_ids_tensors = torch.unsqueeze(batch[4],-1).permute(2,0,1)
        else:
            slot_label_ids_tensors = batch[4].permute(2,0,1)
        for idx, data_type in enumerate(self.data_args.task.split('+')):
            intent_label_ids[data_type] = intent_label_ids_tensors[idx]
            slot_label_ids[data_type] = slot_label_ids_tensors[idx]

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'intent_label_ids' : intent_label_ids,
                  'slot_label_ids' : slot_label_ids
                }

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

        return inputs


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
        best_model_epoch = 0
        best_model_step = 0
        epoch_count = -1
        dev_score_history, dev_step_history = [], []
        self.model.zero_grad()

        tasks = self.data_args.task.split('+')
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_count+=1
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = self.load_inputs_from_batch(batch)

                outputs = self.model(**inputs)

                loss_dict = outputs[0]
                loss = 0.0
                for lname, lvalue in loss_dict.items():
                    loss += lvalue

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()

                wandb.log({"Train/total_loss": tr_loss / (global_step+1)})
                wandb.log({"Train/".format(ln.replace("_","/")):lv for ln, lv in loss_dict.items()})
                epoch_iterator.set_description("step {}/{} loss={:.2f}".format(
                        step,
                        global_step,
                        tr_loss / (global_step+1)))

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
                        dev_score = 0.0
                        for presult in result_to_save["paired_results"]:
                            dev_score += presult["results"]["slot_merged_TERM_DEFINITION_f1_mean"]

                        if global_step == self.args.logging_steps  or dev_score > max(dev_score_history):
                            dataconfig = {
                                "intent_label" : self.intent_label_dict,
                                "slot_label" : self.slot_label_dict,
                                "pos_label" : self.pos_label_lst
                            }

                            self.save_model(dataconfig)
                            best_model_epoch = epoch_count
                            best_model_step = global_step
                            # self.copy_best_model()
                            logger.info(" ******* new best model saved at step {}, epoch {}: {}".format(highlight(global_step), highlight(epoch_count), highlight(dev_score)))
                        else:
                            logger.info("best model still at step {}, epoch {}".format(highlight(best_model_step), highlight(best_model_epoch)))

                        dev_score_history += [dev_score]
                        dev_step_history += [global_step]
                        result_to_save['best_slot_merged_TERM_DEFINITION_f1_mean'] = max(dev_score_history)
                        result_to_save['best_global_step'] = dev_step_history[dev_score_history.index(result_to_save['best_slot_merged_TERM_DEFINITION_f1_mean'])]

                        # save log
                        filename = 'logs/logs_train_{}_{}.txt'.format(self.data_args.kfold, self.model_args.model_name_or_path)
                        if not os.path.exists(os.path.dirname(filename)):
                            os.makedirs(os.path.dirname(filename))
                        with open(filename,'a') as f:
                            if self.data_args.kfold == 0:
                                f.write('{}\n'.format('\t'.join(list(result_to_save.keys()))))
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

        # TODO make following variables as a dictionary
        eval_loss = 0.0
        nb_eval_steps = 0
        input_ids_all = None
        pos_ids_all = None
        sequence_outputs = None
        pooled_outputs = None

        intent_preds = {}
        intent_conf = {}
        gold_intent_label_dict = {}

        slot_preds = {}
        slot_conf = {}
        gold_slot_labels_ids = {}

        self.model.eval()

        logger.info("Start model predicion")
        for bidx, batch in enumerate(tqdm(eval_dataloader, desc="Neural Inference")):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self.load_inputs_from_batch(batch)

                outputs = self.model(**inputs)

                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                sequence_output, pooled_output = None, None
                if len(outputs) > 2:
                    sequence_output = outputs[2]
                    pooled_output = outputs[3]

                for lname, lvalue in tmp_eval_loss.items():
                    eval_loss += lvalue

                eval_loss = eval_loss.item()
            nb_eval_steps += 1

            # Input
            if input_ids_all is None:
                input_ids_all = inputs["input_ids"].detach().cpu().numpy()
            else:
                input_ids_all = numpy.append(input_ids_all, inputs["input_ids"].detach().cpu().numpy(), axis=0)
            # POS
            if pos_ids_all is None:
                pos_ids_all = inputs["pos_label_ids"].detach().cpu().numpy()
            else:
                pos_ids_all = numpy.append(pos_ids_all, inputs["pos_label_ids"].detach().cpu().numpy(), axis=0)

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

            intent_probs = {}
            for data_type, logits in intent_logits.items():
                intent_probs[data_type] = torch.softmax(logits, dim=1).detach().cpu().numpy()
            if len(list(intent_preds.keys())) == 0:
                for data_type, logits in intent_logits.items():
                    intent_preds[data_type] = logits.detach().cpu().numpy()
                    gold_intent_label_dict[data_type] = inputs["intent_label_ids"][data_type].detach().cpu().numpy()
                    intent_conf[data_type] = intent_probs[data_type][:,1]
            else:
                for data_type, logits in intent_logits.items():
                    intent_preds[data_type] = numpy.append(
                        intent_preds[data_type], logits.detach().cpu().numpy(), axis=0

                    )
                    gold_intent_label_dict[data_type] = numpy.append(
                        gold_intent_label_dict[data_type],
                        inputs["intent_label_ids"][data_type].detach().cpu().numpy(),
                        axis=0,
                    )
                    intent_conf[data_type] = numpy.append(
                        intent_conf[data_type], intent_probs[data_type][:,1], axis=0
                    )

            slot_probs = {}
            for data_type, logits in slot_logits.items():
                slot_probs[data_type] = torch.softmax(logits,dim=2).detach().cpu().numpy()
            if len(list(slot_preds.keys())) == 0:
                for data_type, logits in slot_logits.items():
                    if self.args.use_crf:
                        decode_out = self.model.crfs[data_type].decode(logits)
                        slot_logits_crf = numpy.array(decode_out)
                        # decode() in `torchcrf` returns list with best index directly
                        slot_preds[data_type] = slot_logits_crf
                        # get confidence from softmax
                        I,J = numpy.ogrid[:slot_logits_crf.shape[0], :slot_logits_crf.shape[1]]
                        slot_conf[data_type] = slot_probs[data_type][I, J, slot_logits_crf]
                    else:
                        slot_preds[data_type] = logits.detach().cpu().numpy()

                    gold_slot_labels_ids[data_type] = inputs["slot_label_ids"][data_type].detach().cpu().numpy()
            else:
                for data_type, logits in slot_logits.items():
                    if self.args.use_crf:
                        slot_logits_crf = numpy.array(self.model.crfs[data_type].decode(logits))
                        slot_preds[data_type] = numpy.append(slot_preds[data_type], slot_logits_crf, axis=0)
                        # get confidence from softmax
                        I,J = numpy.ogrid[:slot_logits_crf.shape[0], :slot_logits_crf.shape[1]]
                        slot_conf[data_type] = numpy.append(slot_conf[data_type], slot_probs[data_type][I, J, slot_logits_crf], axis=0)
                    else:
                        slot_preds[data_type] = numpy.append(slot_preds[data_type], logits.detach().cpu().numpy(), axis=0)

                    gold_slot_labels_ids[data_type] = numpy.append(gold_slot_labels_ids[data_type], inputs["slot_label_ids"][data_type].detach().cpu().numpy(), axis=0)

        if nb_eval_steps == 0:
            return [], {}

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        for data_type, logits in intent_preds.items():
            intent_preds[data_type] = numpy.argmax(intent_preds[data_type], axis=1)

        # Slot result
        for data_type, preds in slot_preds.items():
            if not self.args.use_crf:
                # get confidence from softmax
                I,J = numpy.ogrid[:preds.shape[0], :preds.shape[1]]
                slot_conf[data_type] = preds[I, J, numpy.argmax(preds, axis=2)]
                slot_preds[data_type] = numpy.argmax(preds, axis=2)

        slot_label_map = {}
        slot_label_to_idx_map = {}
        for data_type, slot_labels in self.slot_label_dict.items():
            slot_label_map[data_type] = {i: label for i, label in enumerate(slot_labels)}
            slot_label_to_idx_map[data_type] = {label: i for i, label in enumerate(slot_labels)}
        pos_label_map = {i: label for i, label in enumerate(self.pos_label_lst)}
        pos_label_to_idx_map = {label: i for i, label in enumerate(self.pos_label_lst)}

        gold_slot_num_batch = int(len(gold_slot_labels_ids[list(gold_slot_labels_ids.keys())[0]]))
        gold_slot_num_length = int(len(gold_slot_labels_ids[list(gold_slot_labels_ids.keys())[0]][0]))
        slot_labels_dict = {}
        slot_preds_dict = {}
        slot_conf_dict = {}
        for data_type, _ in self.slot_label_dict.items():
            slot_labels_dict[data_type] = [[] for _ in range(gold_slot_num_batch)]
            slot_preds_dict[data_type] = [[] for _ in range(gold_slot_num_batch)]
            slot_conf_dict[data_type] = [[] for _ in range(gold_slot_num_batch)]
        input_ids_list = [[] for _ in range(gold_slot_num_batch)]
        pos_tags_list = [[] for _ in range(gold_slot_num_batch)]

        id_mapping = defaultdict(list)
        for i in range(gold_slot_num_batch):
            prev_input_ids = []
            for j in range(gold_slot_num_length):
                if label_given:
                    # labels (ignore subword labels and special tokens)
                    first_dataset = list(gold_slot_labels_ids.keys())[0]
                    if gold_slot_labels_ids[first_dataset][i, j] != self.pad_token_label_id:
                        pos_tags_list[i].append(pos_label_map[pos_ids_all[i][j]])
                        input_ids_list[i].append([input_ids_all[i][j]])
                    else:
                        if j > 0 and input_ids_all[i][j] > 2:
                            input_ids_list[i][-1] = input_ids_list[i][-1] + [input_ids_all[i][j]]
                    for data_type, _ in self.slot_label_dict.items():
                        if gold_slot_labels_ids[data_type][i, j] != self.pad_token_label_id:
                            slot_labels_dict[data_type][i].append(slot_label_map[data_type][gold_slot_labels_ids[data_type][i][j]])
                            slot_preds_dict[data_type][i].append(slot_label_map[data_type][slot_preds[data_type][i][j]])
                            slot_conf_dict[data_type][i].append(slot_conf[data_type][i][j])

                else:
                    # labels (ignore subword labels and special tokens)
                    # input subword concatenation
                    if pos_ids_all[i, j] != self.pad_token_label_id:
                        input_ids_list[i].append([input_ids_all[i][j]])
                        pos_tags_list[i].append(pos_label_map[pos_ids_all[i][j]])
                    else:
                        try:
                            if j > 0 and input_ids_all[i][j] > 2:
                                input_ids_list[i][-1] = input_ids_list[i][-1] + [input_ids_all[i][j]]
                        except Exception as e:
                            print(e)
                            print(input_ids_all)
                            continue

                    for data_type, _ in self.slot_label_dict.items():
                        if pos_ids_all[i, j] != self.pad_token_label_id:
                            slot_labels_dict[data_type][i].append(slot_label_map[data_type][gold_slot_labels_ids[data_type][i][j]])
                            slot_preds_dict[data_type][i].append(slot_label_map[data_type][slot_preds[data_type][i][j]])
                            slot_conf_dict[data_type][i].append(slot_conf[data_type][i][j])

        # convert input ids to tokens
        input_tokens_list = []
        for input_ids in input_ids_list:
            input_tokens = []
            for tset in input_ids:
                # print(self.tokenizer.convert_ids_to_tokens(tset))
                input_tokens.append("".join(self.tokenizer.convert_ids_to_tokens(tset)))
            input_tokens_list.append(input_tokens)

        # only input information processed
        inputs_dict = {
            "input_tokens": input_tokens_list,
            "pos": pos_tags_list,
            "intent_labels": gold_intent_label_dict,
            "slot_labels": slot_labels_dict,
        }

        # neural-based definition detector
        neural_predictions = {
            "intent_preds":intent_preds,
            "intent_conf": intent_conf,
            "slot_preds": slot_preds_dict,
            "slot_conf": slot_conf_dict,
            "pooled_outputs": pooled_outputs,
            "sequence_outputs": sequence_outputs
        }
        inputs_list = transpose_dict_to_list_hybrid(inputs_dict,
                                             without_subdicts=["input_tokens","pos"])
        prediction_list = transpose_dict_to_list_hybrid(neural_predictions,
                                                 without_subdicts=["pooled_outputs","sequence_outputs"])

        # we keep having the multi term-def slot predictions
        # NOTE after evaluation, we make them separated for frontend
        # prediction_list = [[p] for p in prediction_list]
        return inputs_list, prediction_list


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
        elif dataset is not None:
            raw_data = [{}] * len(dataset)
            # TODO convert latex $XX$ to SYMBOL
        else:
            raise Exception("Only dev and test or input dataset available")

        logger.info("***** Running inference on %s  (%s) dataset *****", mode, label_given)
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
            use_heuristic = self.args.use_heuristic,
            verbose=verbose,
        )

        if label_given:
            logger.info("***** Running evaluation on merged dataset %s *****", mode)
            merged_results = self.compute_eval_metrics(
                merged_predictions,
                do_per_instance_eval = True,
                verbose = verbose
            )
            if verbose:
                pprint(merged_results)
        else:
            logger.info("***** Not running evaluation if label not given *****")
            merged_results = {}

        if do_visualize:
            logger.info("***** Visualizing posterior dist. on %s dataset *****", mode)
            eval_label = [slot_label_to_idx_map[s] for sb in slot_preds_list for s in sb ]
            eval_prob = [s for sb in slot_conf_list for s in sb ]
            plot_posterior(eval_label, eval_prob)

        return merged_results, merged_predictions


    def merge_predictions(self, raw_data, inputs_list, prediction_dict, merge_acronym=None, merge_symbol=None, use_heuristic=False, verbose=False):
        merged_prediction_list = []

        tasks = self.data_args.task.split('+')
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
            slot_dict["neural"].pop("intent_conf")
            slot_dict["neural"].pop("intent_preds")
            slot_dict["neural"].pop("pooled_outputs")
            slot_dict["neural"].pop("sequence_outputs")
            if prediction_dict["acronym"] is not None:
                slot_dict["acronym"] = prediction_dict["acronym"][didx]
            if prediction_dict["nickname"] is not None:
                slot_dict["nickname"] = prediction_dict["nickname"][didx]
            merged_dict["outputs"]["slot"] = slot_dict

            # heuristics (gold labels are in fact not used)
            # use different heuristics for each type?
            if use_heuristic:
                #TODO later, type-specific filters
                for task in tasks:
                    # print(didx,data["tokens"])
                    # print([merged_dict["outputs"]["intent"]["preds"][task]])
                    # print(merged_dict["outputs"]["slot"]["neural"]["slot_preds"][task])
                    if task == "W00":
                        filtered_intent_preds, filtered_slot_preds = common_filters(
                            [merged_dict["outputs"]["intent"]["preds"][task]],
                            [merged_dict["outputs"]["slot"]["neural"]["slot_preds"][task]])
                        merged_dict["outputs"]["intent"]["preds"][task] = filtered_intent_preds[0]
                        merged_dict["outputs"]["slot"]["neural"]["slot_preds"][task] = filtered_slot_preds[0]
                        # print([merged_dict["outputs"]["intent"]["preds"][task]])
                        # print(merged_dict["outputs"]["slot"]["neural"]["slot_preds"][task])

            # if data["name"] == "AI2020":
                # print(didx)
                # print(data)
                # print(prediction_dict["neural"][didx])
            #     from pdb import set_trace; set_trace()



            # for neural+symbolic combined model
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

                for task in tasks:
                    print("Task: {}".format(highlight(task)))
                    print("{} {}\t{}".format(
                        highlight("Gold".upper()),
                        gold_type,
                        colorize_term_definition(
                            merged_dict["inputs"]['input_tokens'],
                            merged_dict["inputs"]["slot_labels"][task],
                            term_postfix=term_label_postfix_map[task],
                            def_postfix=def_label_postfix_map[task])
                    ))

                    print("Intent predicted: {} (gold: {})".format(
                            highlight(merged_dict["outputs"]["intent"]["preds"][task]),
                            highlight(merged_dict["inputs"]["intent_labels"][task])
                        ))

                    for system, pdict in merged_dict["outputs"]["slot"].items():
                        # we don't output [O O O ] O only ensemble_predictions
                        slot_preds = pdict["slot_preds"][task] if task in pdict["slot_preds"] else pdict["slot_preds"]

                        if len(list(set(slot_preds))) == 1:
                            continue

                        print("{}\t{}".format(
                            highlight(system.upper()),
                            colorize_term_definition(
                                merged_dict["inputs"]['input_tokens'],
                                slot_preds,
                                term_postfix=term_label_postfix_map[task],
                                def_postfix=def_label_postfix_map[task])
                            ))
                print("")
        return merged_prediction_list



    def compute_eval_metrics(self, predictions, do_per_instance_eval=False, verbose=False):
        results = {}
        tasks = self.data_args.task.split('+')

        """ per-instance evaluation first"""
        slot_labels_per_type_system_dict = defaultdict(list)
        slot_preds_per_type_system_dict = defaultdict(list)
        for pid, p in enumerate(predictions):
            term_type = p["raw_data"]["type"] if "type" in p["raw_data"] else None

            if do_per_instance_eval:
                p["results"] = {"intent":{}, "slot":{}}
                for task in tasks:
                    result_per_instance_sentence_classification = compute_metrics_for_sentence_classification(
                        [p["outputs"]["intent"]["preds"][task]],
                        [p["inputs"]["intent_labels"][task]],
                        type_ = term_type
                    )
                    p["results"]["intent"][task] = result_per_instance_sentence_classification

                    for system, pdict in p["outputs"]["slot"].items():
                        # To deal with multi term-def cases, merge list of predictions per instance here
                        if task == "AI2020":
                            type_ = "abbreviation"
                        elif task.startswith("DocDef"):
                            type_ = "symbol"
                        elif task == "term":
                            type_ = "term"
                        else:
                            type_ = None

                        #FIXME TODO rule-nickname detector and off-the-shelf acronym detector doesn't support dictionary format for tasks yet. TOBEFIXED
                        result_per_instance_slot_tagging_list = []

                        slot_preds = pdict["slot_preds"][task] if task in pdict["slot_preds"] else pdict["slot_preds"]

                        result_per_instance_slot_tagging = compute_metrics_for_slot_tagging(
                            [slot_preds],
                            [p["inputs"]['slot_labels'][task]],
                            type_
                        )

                        # pprint(result_per_instance_slot_tagging)
                        result_per_instance_slot_tagging_list.append(result_per_instance_slot_tagging)
                        if system not in p["results"]["slot"]:
                            p["results"]["slot"][system] = {}
                        p["results"]["slot"][system][task] = result_per_instance_slot_tagging_list
                # if verbose:
                #     pprint(p["results"])

            # aggregate slot predictions from multiple systems
            #   sytem: ["neural", "abbreviation", "nickname"]
            #   term_type: ["term", "acronym", "symbol"]
            for task in tasks:
                for system, pdict in p["outputs"]["slot"].items():

                    slot_preds = pdict["slot_preds"][task] if task in pdict["slot_preds"] else pdict["slot_preds"]
                    # NOTE we don't evaluate with negative combined test set(e.g., DocDef2 for Acronym)
                    if "name" in p["raw_data"] and task != p["raw_data"]["name"]:
                        continue

                    type_system_pair = (task, system)

                    slot_labels_per_type_system_dict[type_system_pair].append(
                        p["inputs"]["slot_labels"][task])
                    slot_preds_per_type_system_dict[type_system_pair].append(
                        slot_preds)


        """ overall evaluation"""
        # calculate total intent evaluation metrics
        task_intent_results_dict = {}
        for task in tasks:
            predictions_with_task = []
            for p in predictions:
                if "name" in p["raw_data"]:
                    if task == p["raw_data"]["name"]:
                        predictions_with_task.append(p)
                else:
                    predictions_with_task.append(p)

            total_result_sentence_classification = compute_metrics_for_sentence_classification(
                [p["outputs"]["intent"]["preds"][task] for p in predictions_with_task],
                [p["inputs"]["intent_labels"][task] for p in predictions_with_task])
            task_intent_results_dict[task] = total_result_sentence_classification
        results.update({"intent":task_intent_results_dict})

        # calculate total slot evaluation metrics per each term type and each system
        type_system_results_list = []
        for type_system_pair in slot_labels_per_type_system_dict:
            if type_system_pair[0] == "AI2020":
                type_ = "abbreviation"
            elif type_system_pair[0].startswith("DocDef"):
                type_ = "symbol"
            elif type_system_pair[0] == "term":
                type_ = "term"
            else:
                type_ = None

            per_type_system_result_slot_tagging = compute_metrics_for_slot_tagging(
                slot_preds_per_type_system_dict[type_system_pair],
                slot_labels_per_type_system_dict[type_system_pair],
                type_
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



#     def featurize(self, text: str, limit: bool = False) -> DefaultDict[Any, Any]:
        # doc = self.nlp(text)

        # # Extract tokens containing...
        # # (1) Abbreviations
        # abbrev_tokens = []
        # for abrv in doc._.abbreviations:
            # abbrev_tokens.append(str(abrv._.long_form).split())
        # abbrev_tokens_flattened = [t for at in abbrev_tokens for t in at]

        # # (2) Entities
        # entities = [str(e) for e in doc.ents]
        # entity_tokens = [e.split() for e in entities]
        # entity_tokens_flattened = [t for et in entity_tokens for t in et]

        # # (3) Noun phrases
        # np_tokens = []
        # for chunk in doc.noun_chunks:
            # np_tokens.append(str(chunk.text).split())
        # np_tokens_flattened = [t for et in np_tokens for t in et]

        # # (4) Verb phrases
        # verb_matches = self.verb_matcher(doc)
        # spans = [doc[start:end] for _, start, end in verb_matches]
        # vp_tokens = filter_spans(spans)
        # vp_tokens_flattened = [str(t) for et in vp_tokens for t in et]

        # # Limit the samples.
        # if limit:
            # doc = doc[:limit]

        # # Aggregate all features together.
        # features: DefaultDict[str, List[Union[int, str]]] = defaultdict(list)
        # for token in doc:
            # if str(token.text) == '---':
                # features["tokens"].append('</s>')
            # else:
                # features["tokens"].append(str(token.text))
            # features["pos"].append(str(token.tag_))  # previously token.pos_
            # features["head"].append(str(token.head))
            # # (Note: the following features are binary lists indicating the presence of a
            # # feature or not per token, like "[1 0 0 1 1 1 0 0 ...]")
            # features["entities"].append(
                # 1 if token.text in entity_tokens_flattened else 0
            # )
            # features["np"].append(1 if token.text in np_tokens_flattened else 0)
            # features["vp"].append(1 if token.text in vp_tokens_flattened else 0)
            # features["abbreviation"].append(
                # 1 if token.text in abbrev_tokens_flattened else 0
            # )

 #        return features

    def extract_symbols(self, sentence):
        symbols = re.findall(r"\[\[.*?\]\]", sentence)
        if len(symbols)>0:
            for symbol in symbols:
                sentence = sentence.replace(symbol,'SYMBOL')
            return sentence, symbols
        else:
            return sentence, []

    def insert_symbols(self, raw_tokens, symbols_list):
        raw_processed = []
        for symbols, raw in zip(symbols_list,raw_tokens):
            if len(symbols)>0:
                sym_idx = 0
                tokens_processed = []
                for token in raw:
                    if token=='SYMBOL':
                        tokens_processed.append(symbols[sym_idx])
                        sym_idx+=1
                    else:
                        tokens_processed.append(token)
                raw_processed.append(tokens_processed)
            else:
                raw_processed.append(raw)
        return raw_processed


    def predict_batch(self, data: List[str]) -> Tuple[Dict[Any, Any], Dict[str, List[List[str]]], Dict[Any, Any]]:
        #Featurize
        features = []
        symbols_list = []
        for sentence in tqdm(data,desc="Featurize Sentences"):
            sentence, symbols = self.extract_symbols(sentence)
            symbols_list.append(symbols)
            featurized_text = self.nlp.featurize(sentence)
            features.append(featurized_text)

        # Load data.
        test_dataset, raw = load_and_cache_example_batch_raw(
            self.data_args, self.tokenizer, features, self.pos_label_lst
        )

        # Perform inference.
        intent_pred, slot_preds, slot_pred_confs = self.evaluate_from_input(test_dataset, raw, symbols_list)

        # Process predictions.
        simplified_slot_preds_dict: Dict[str, List[List[str]]] = {}
        for prediction_type, slot_pred_list in slot_preds.items():
            simplified_slot_preds = []
            for slot_pred in slot_pred_list:
                simplified_slot_pred = []
                for s in slot_pred:
                    if s.endswith("TERM"):
                        simplified_slot_pred.append("TERM")
                    elif s.endswith("DEF"):
                        simplified_slot_pred.append("DEF")
                    else:
                        simplified_slot_pred.append("O")
                simplified_slot_preds.append(simplified_slot_pred)
            simplified_slot_preds_dict[prediction_type] = simplified_slot_preds

        raw_processed = self.insert_symbols(raw, symbols_list)

        return intent_pred, simplified_slot_preds_dict, slot_pred_confs, raw_processed

    def evaluate_from_input(self, dataset: TensorDataset, raw: List[List[str]], symbols_list: List[List[str]]) -> Tuple[Dict[Any, Any], Dict[str, List[List[str]]], Dict[Any, Any]]:
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

        # Run evaluation.
        eval_loss = 0.0
        nb_eval_steps = 0
        input_ids_all = None
        pos_ids_all = None

        intent_preds = {}
        intent_conf = {}
        slot_preds = {}
        slot_conf = {}
        gold_intent_label_dict = {}
        gold_slot_labels_ids = {}

        self.model.eval()

        for batch in tqdm(eval_dataloader,desc="Run Prediction"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self.load_inputs_from_batch(batch)
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
                # eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if input_ids_all is None and pos_ids_all is None:
                input_ids_all = inputs["input_ids"].detach().cpu().numpy()
                pos_ids_all = inputs["pos_label_ids"].detach().cpu().numpy()
            else:
                input_ids_all = numpy.concatenate((input_ids_all,inputs["input_ids"].detach().cpu().numpy()))
                pos_ids_all = numpy.concatenate((pos_ids_all,inputs["pos_label_ids"].detach().cpu().numpy()))
            # POS

            # Predict intent
            intent_probs = {}
            for data_type, logits in intent_logits.items():
                intent_probs[data_type] = torch.softmax(logits, dim=1).detach().cpu().numpy()
                if data_type not in gold_intent_label_dict and data_type not in intent_preds and data_type not in intent_conf:
                    gold_intent_label_dict[data_type] = inputs["intent_label_ids"][data_type].detach().cpu().numpy()
                    intent_preds[data_type] = logits.detach().cpu().numpy()
                    intent_conf[data_type] = intent_probs[data_type][:,1]
                else:
                    gold_intent_label_dict[data_type] = numpy.concatenate((gold_intent_label_dict[data_type], inputs["intent_label_ids"][data_type].detach().cpu().numpy()))
                    intent_preds[data_type] = numpy.concatenate((intent_preds[data_type], logits.detach().cpu().numpy()))
                    intent_conf[data_type] = numpy.concatenate((intent_conf[data_type], intent_probs[data_type][:,1]))
            # Predict slots.
            slot_probs = {}
            for data_type, logits in slot_logits.items():
                slot_probs[data_type] = torch.softmax(logits,dim=2).detach().cpu().numpy()
                if self.args.use_crf:
                    decode_out = self.model.crfs[data_type].decode(logits)
                    slot_logits_crf = numpy.array(decode_out)
                    # decode() in `torchcrf` returns list with best index directly
                    if data_type not in slot_preds:
                        slot_preds[data_type] = slot_logits_crf
                    else:
                        slot_preds[data_type] = numpy.concatenate((slot_preds[data_type], slot_logits_crf))
                    # get confidence from softmax
                    I,J = numpy.ogrid[:slot_logits_crf.shape[0], :slot_logits_crf.shape[1]]
                    if data_type not in slot_conf:
                        slot_conf[data_type] = slot_probs[data_type][I, J, slot_logits_crf]
                    else:
                        slot_conf[data_type] = numpy.concatenate((slot_conf[data_type], slot_probs[data_type][I, J, slot_logits_crf]))
                else:
                    if data_type not in slot_preds:
                        slot_preds[data_type] = logits.detach().cpu().numpy()
                    else:
                        slot_preds[data_type] = numpy.concatenate((slot_preds[data_type], logits.detach().cpu().numpy()))

                if data_type not in gold_slot_labels_ids:
                    gold_slot_labels_ids[data_type] = inputs["slot_label_ids"][data_type].detach().cpu().numpy()
                else:
                    gold_slot_labels_ids[data_type] = numpy.concatenate((gold_slot_labels_ids[data_type], inputs["slot_label_ids"][data_type].detach().cpu().numpy()))

        if nb_eval_steps == 0:
            return [], {}

        #Intent Result
        for data_type, logits in intent_preds.items():
            intent_preds[data_type] = numpy.argmax(intent_preds[data_type], axis=1)

        # Slot result
        for data_type, preds in slot_preds.items():
            if not self.args.use_crf:
                # get confidence from softmax
                I,J = numpy.ogrid[:preds.shape[0], :preds.shape[1]]
                slot_conf[data_type] = preds[I, J, numpy.argmax(preds, axis=2)]
                slot_preds[data_type] = numpy.argmax(preds, axis=2)

        slot_label_map = {}
        slot_label_to_idx_map = {}
        for data_type, slot_labels in self.slot_label_dict.items():
            slot_label_map[data_type] = {i: label for i, label in enumerate(slot_labels)}
            slot_label_to_idx_map[data_type] = {label: i for i, label in enumerate(slot_labels)}
        pos_label_map = {i: label for i, label in enumerate(self.pos_label_lst)}
        pos_label_to_idx_map = {label: i for i, label in enumerate(self.pos_label_lst)}

        gold_slot_num_batch = int(len(gold_slot_labels_ids[list(gold_slot_labels_ids.keys())[0]]))
        gold_slot_num_length = int(len(gold_slot_labels_ids[list(gold_slot_labels_ids.keys())[0]][0]))
        slot_labels_dict = {}
        slot_preds_dict = {}
        slot_conf_dict = {}
        for data_type, _ in self.slot_label_dict.items():
            slot_labels_dict[data_type] = [[] for _ in range(gold_slot_num_batch)]
            slot_preds_dict[data_type] = [[] for _ in range(gold_slot_num_batch)]
            slot_conf_dict[data_type] = [[] for _ in range(gold_slot_num_batch)]
        input_ids_list = [[] for _ in range(gold_slot_num_batch)]
        pos_tags_list = [[] for _ in range(gold_slot_num_batch)]

        id_mapping = defaultdict(list)
        for i in range(gold_slot_num_batch):
            prev_input_ids = []
            for j in range(gold_slot_num_length):
                # labels (ignore subword labels and special tokens)
                first_dataset = list(gold_slot_labels_ids.keys())[0]
                if gold_slot_labels_ids[first_dataset][i, j] != self.pad_token_label_id:
                    pos_tags_list[i].append(pos_label_map[pos_ids_all[i][j]])
                    input_ids_list[i].append([input_ids_all[i][j]])
                else:
                    if j > 0 and input_ids_all[i][j] > 2:
                        input_ids_list[i][-1] = input_ids_list[i][-1] + [input_ids_all[i][j]]
                for data_type, _ in self.slot_label_dict.items():
                    if gold_slot_labels_ids[data_type][i, j] != self.pad_token_label_id:
                        slot_labels_dict[data_type][i].append(slot_label_map[data_type][gold_slot_labels_ids[data_type][i][j]])
                        slot_preds_dict[data_type][i].append(slot_label_map[data_type][slot_preds[data_type][i][j]])
                        slot_conf_dict[data_type][i].append(slot_conf[data_type][i][j])

        # Apply heuristic filters.
        raw_processed = self.insert_symbols(raw, symbols_list)

        if self.args.use_heuristic:
            intent_preds, slot_preds_dict = self.heuristic_filters(
                intent_preds,
                slot_preds_dict,
                raw,
                self.data_args.task,
                raw_processed
            )
        return intent_preds, slot_preds_dict, slot_conf_dict
