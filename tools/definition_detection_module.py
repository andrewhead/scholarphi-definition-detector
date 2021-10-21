import sys
import os
import json
import argparse
import tempfile

sys.path.append('../')
from model import JointRoberta
from transformers import HfArgumentParser, AutoTokenizer
from configuration import DataTrainingArguments,ModelArguments,TrainingArguments
from utils import get_joint_labels
from trainer_joint import JointTrainer


class DefinitionModule(object):
    def __init__(self, models_dir_path, task):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses([
            "--model_name_or_path",
            "roberta-large",
            "--data_dir", models_dir_path,
            "--task", task,
            "--output_dir",
            f"{models_dir_path}/roberta-large",
            "--use_crf", "--use_heuristic",
            "--use_pos", "--use_np", "--use_vp", "--use_entity", "--use_acronym",
            "--do_eval",
            "--per_device_train_batch_size","4",
            "--per_device_eval_batch_size","4",
            "--max_seq_len","100"])

        training_args.output_dir = "{}{}{}{}{}{}".format(training_args.output_dir,
            "_pos={}".format(training_args.use_pos) if training_args.use_pos else "","_np={}".format(training_args.use_np) if training_args.use_np else "",
            "_vp={}".format(training_args.use_vp) if training_args.use_vp else "",
            "_entity={}".format(training_args.use_entity) if training_args.use_entity else "",
            "_acronym={}".format(training_args.use_acronym) if training_args.use_acronym else "",
        )
        os.environ["WANDB_DISABLED"] = "1"

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        intent_label_dict = get_joint_labels(training_args, "intent_label")
        slot_label_dict = get_joint_labels(training_args, "slot_label")
        pos_label_lst = get_joint_labels(training_args, "pos_label")

        model = JointRoberta.from_pretrained(
            training_args.output_dir,
            args=training_args,
            intent_label_dict=intent_label_dict,
            slot_label_dict=slot_label_dict,
            pos_label_lst=pos_label_lst,
            tasks=task.split('+'),
        )

        self.trainer = JointTrainer([training_args, model_args, data_args,],model,tokenizer=tokenizer)
        self.tasks = task.split("+")


    def predict(self,text):
        intent_pred, simplified_slot_preds_dict, slot_pred_confs, raw_processed = self.trainer.predict_batch(text)
        results = []
        for s,sentence in enumerate(raw_processed):
            result = {
                "tokens" : sentence,
                "intent_prediction" : {},
                "slot_prediction" : {},
            }
            for t in self.tasks:
                result['intent_prediction'][t] = int(intent_pred[t][s])
                result['slot_prediction'][t] = simplified_slot_preds_dict[t][s]
            results.append(result)
        return results

