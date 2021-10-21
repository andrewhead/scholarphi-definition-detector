import os
import sys
import json
from pprint import pprint, pformat
import argparse
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import coloredlogs
from transformers import *


from model import *
from trainer import Trainer
from trainer_joint import JointTrainer
from utils import init_logger, set_seed, get_intent_labels, get_slot_labels, get_pos_labels, highlight, info, NpEncoder, get_intent_labels_hybrid, get_slot_labels_hybrid
from eval_utils import compute_metrics,read_prediction_text
from data_loader import load_and_cache_examples
from configuration import ModelArguments, DataTrainingArguments, TrainingArguments

logger = logging.get_logger()
logging.set_verbosity_info()
coloredlogs.install(fmt='%(asctime)s %(name)s %(levelname)s %(message)s',level='INFO',datefmt='%m/%d%H:%M:%S',logger=logger)


MODEL_CLASSES = {
    'bert': BERT,
    'distilbert': DistilBERT,
    'albert': Albert,
    'roberta': Roberta,
    'joint_roberta': JointRoberta,
    'allenai/longformer': Longformer,
    'allenai/scibert': BERT,
    'allenai/cs': Roberta
}


MODEL_MAP_LIST_IN_HG = ["albert-xxlarge-v2", "bert-base-uncased", "bert-large-uncased", "roberta-large", "roberta-base", "allenai/scibert_scivocab_uncased", "albert-base-v2", "albert-large-v2", "allenai/longformer-base-4096", "allenai/cs_roberta_base"]


# majority voting
def ensemble_inference(predictions):
    slot_preds_major_voted = []
    for nid in range(len(predictions[0]['slot_preds'] )):
        preds = []
        for pid, prediction in enumerate(predictions):
            preds.append(predictions[pid]['slot_preds'][nid])

        preds_major_voted = []
        for row in np.array(preds).transpose():
            unique,pos = np.unique(row,return_inverse=True)
            maxpos = np.bincount(pos).argmax()
            # print(unique[maxpos],counts[maxpos])
            preds_major_voted.append(unique[maxpos])
        slot_preds_major_voted.append(preds_major_voted)

    intent_preds_major_voted = []
    for nid in range(len(predictions[0]['intent_preds'] )):
        preds = []
        for pid, prediction in enumerate(predictions):
            preds.append(predictions[pid]['intent_preds'][nid])

        unique,pos = np.unique(preds,return_inverse=True)
        maxpos = np.bincount(pos).argmax()
        # print(unique[maxpos],counts[maxpos])
        intent_preds_major_voted.append(unique[maxpos])
    return intent_preds_major_voted, slot_preds_major_voted


def renaming_output_dir(training_args):
    output_dir = \
        '{}{}{}{}{}{}'.format(
            training_args.output_dir,
            '_pos={}'.format(training_args.use_pos) if training_args.use_pos else '',
            '_np={}'.format(training_args.use_np) if training_args.use_np else '',
            '_vp={}'.format(training_args.use_vp) if training_args.use_vp else '',
            '_entity={}'.format(training_args.use_entity) if training_args.use_entity else '',
            '_acronym={}'.format(training_args.use_acronym) if training_args.use_acronym else '')
    return output_dir

def main(model_args, data_args, training_args):
    # init_logger()
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )


    # turn this off for now, to switch our training to joint learning at default
    # if ('+' in data_args.task and not training_args.joint_learning) or \
            # ('+' not in data_args.task and training_args.joint_learning):
        # raise ValueError(
            # f"When multiple datasets ({{data_args.task}}) are given, joint learning option {{training_args.joint_learning}}, or vice versa should be turned on"
    #     )


    set_seed(training_args)
    info(logger, training_args)

    if model_args.model_name_or_path not in MODEL_MAP_LIST_IN_HG:
        model_args.model_name_or_path = os.path.join(data_args.data_dir, model_args.model_name_or_path)
        model_args.model_type = os.path.basename(model_args.model_name_or_path).split('-')[0].split('_')[0] #
    else:
        model_args.model_type = model_args.model_name_or_path.split('-')[0].split('_')[0] #
    logger.info("Model type:{}".format(model_args.model_type))


    # Load config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # Load tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it, and load it from here, using --tokenizer_name"
        )

    # get dataset (NOTE that we use raw_data for dev and test set)
    data_args.ignore_index = training_args.ignore_index

    if '+' in data_args.task:
        train_dataset,_ = load_and_cache_examples(data_args, tokenizer, mode="train", model_name=model_args.model_name_or_path)
        dev_dataset = load_and_cache_examples(data_args, tokenizer, mode="dev", model_name=model_args.model_name_or_path)
        test_dataset = load_and_cache_examples(data_args, tokenizer, mode="test", model_name=model_args.model_name_or_path)
    else:
        train_dataset,_ = load_and_cache_examples(data_args, tokenizer, mode="train", model_name=model_args.model_name_or_path)
        dev_dataset = load_and_cache_examples(data_args, tokenizer, mode="dev", model_name=model_args.model_name_or_path)
        test_dataset = load_and_cache_examples(data_args, tokenizer, mode="test", model_name=model_args.model_name_or_path)

    # Load model class
    model_class = MODEL_CLASSES["{}{}".format(
        "joint_" if training_args.joint_learning else "", model_args.model_type)]
    logger.info(model_class)

    # Load trainer class
    if training_args.joint_learning:
        #TODO later, this should be merged to Trainer at default
        trainer_class = JointTrainer
    else:
        trainer_class = Trainer

    # Start training or inference
    if training_args.do_ensemble and not training_args.do_train and training_args.do_eval:
        predictions_ensemble = []
        output_dir = training_args.output_dir
        for kfold in range(data_args.num_fold):
            # replace kfold directory with kfold in ensemble inference
            training_args.output_dir = os.path.join(os.path.split(os.path.dirname(output_dir))[0], str(kfold), os.path.basename(output_dir))

            # renaming output_dir with conditions
            training_args.output_dir = renaming_output_dir(training_args)
            logger.info(highlight(" Output_dir {}".format(training_args.output_dir)))

            logger.info(highlight(" ***** START Ensemble **** {}".format(kfold)))
            if os.path.exists(training_args.output_dir) and not training_args.overwrite_output_dir:
                model = model_class.from_pretrained(training_args.output_dir,
                                                      args=training_args,
                                                      intent_label_lst=get_intent_labels(data_args),
                                                      slot_label_lst=get_slot_labels(data_args),
                                                      pos_label_lst=get_pos_labels(data_args))
                logger.info(highlight(" ***** Model loaded **** {}".format(training_args.output_dir)))
            else:
                logger.info(highlight("Training new model from scratch"))
                sys.exit(1)

            model.resize_token_embeddings(len(tokenizer))

            # Initialize our Trainer
            trainer = trainer_class(
                [training_args,model_args, data_args],
                model,
                train_dataset,dev_dataset,test_dataset)

            # only for decoding
            if training_args.do_eval:
                results, predictions = trainer.evaluate("test", verbose=True, save_result=False)
                predictions_ensemble.append(predictions)

        # ensemble predictions
        intent_preds_ensemble, slot_preds_ensemble = ensemble_inference(
            predictions_ensemble,
            method=training_args.ensemble_method)

        # calculate evaluation metrics
        results = compute_metrics(
            intent_preds_ensemble,
            predictions_ensemble[0]['intent_labels'],
            slot_preds_ensemble,
            predictions_ensemble[0]['slot_labels'])


        # save_predictions(data_args, predictions)
        save_results(training_args, data_args, results)

    else:
        # renaming output_dir with conditions
        training_args.output_dir = renaming_output_dir(training_args)
        logger.info(highlight(" Output_dir {}".format(training_args.output_dir)))

        if os.path.exists(training_args.output_dir) and not training_args.overwrite_output_dir:
            if training_args.joint_learning:

                model = model_class.from_pretrained(training_args.output_dir,
                                                args=training_args,
                                                intent_label_dict=get_intent_labels_hybrid(data_args),
                                                slot_label_dict=get_slot_labels_hybrid(data_args),
                                                pos_label_lst=get_pos_labels(data_args),
                                                tasks=data_args.task.split('+'))
            else:
                model = model_class.from_pretrained(training_args.output_dir,
                                                args=training_args,
                                                intent_label_lst=get_intent_labels(data_args),
                                                slot_label_lst=get_slot_labels(data_args),
                                                pos_label_lst=get_pos_labels(data_args))
            logger.info(highlight(" ***** Model loaded **** {}".format(training_args.output_dir)))
        else:
            logger.info(highlight("Training new model from scratch"))
            if training_args.joint_learning:
                model = model_class.from_pretrained(model_args.model_name_or_path,
                                                     config=config,
                                                    args=training_args,
                                                    intent_label_dict=get_intent_labels_hybrid(data_args),
                                                    slot_label_dict=get_slot_labels_hybrid(data_args),
                                                    pos_label_lst=get_pos_labels(data_args),
                                                    tasks=data_args.task.split('+'))
            else:
                model = model_class.from_pretrained(model_args.model_name_or_path,
                                                config=config,
                                                args=training_args,
                                                intent_label_lst=get_intent_labels(data_args),
                                                slot_label_lst=get_slot_labels(data_args),
                                                pos_label_lst=get_pos_labels(data_args))

        # below causes NotImplmentedError, is it okay to comment it out?
        # model.resize_token_embeddings(len(tokenizer))

        # Initialize our Trainer
        trainer = trainer_class(
            [training_args,model_args, data_args],
            model,
            train_dataset,
            dev_dataset,
            test_dataset,
            tokenizer=tokenizer)

        if training_args.do_train:
            trainer.train()

        if training_args.do_eval:
            if data_args.eval_data_file is not None:
                results, predictions = trainer.evaluate("test", verbose=False, save_result=False, label_given=False)
            else:
                results, predictions = trainer.evaluate("test", verbose=False, save_result=False)

            save_results_for_v3(training_args, data_args, results, verbose=True)
            save_predictions(data_args, predictions)


def save_predictions(data_args, predictions):
    logger.info("***** Saving predictions %d *****",len(predictions))
    if data_args.eval_data_file is not None:
        prediction_filename = os.path.join(data_args.prediction_dir,
                                            data_args.eval_data_file)
    else:
        prediction_filename = os.path.join(data_args.prediction_dir,
                                            "test.json")
    if not os.path.exists(os.path.dirname(prediction_filename)):
        os.makedirs(os.path.dirname(prediction_filename))

    json.dump(predictions, open(prediction_filename, "w"), indent=2, cls=NpEncoder)
    logger.info("Predictions written {}".format(highlight(prediction_filename)))


# deprecated after v3
def save_results(training_args, data_args, results, verbose=True):
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(highlight(pformat(results[key]))))

    result_to_save = {'model':
                      os.path.basename(training_args.output_dir) + \
                      ('_heuristic=True' if training_args.use_heuristic else '') + \
                      ('_ensemble=True' if training_args.do_ensemble else ''),
                      'ensemble': training_args.do_ensemble,
                       'task': data_args.task,
                       'kfold':data_args.kfold }
    result_to_save['loss'] = -1

    for k,v in results.items():
        if type(v) == list:
            for lidx, vone in enumerate(v):
                result_to_save['{}_{}'.format(k,lidx)] = vone
        else:
            result_to_save[k] = v

    if data_args.eval_data_file is not None:
        result_filename = os.path.join(data_args.result_dir,
                                            data_args.eval_data_file)
    else:
        result_filename = os.path.join(data_args.result_dir,
                                            "test.json")

    if not os.path.exists(os.path.dirname(result_filename)):
        os.makedirs(os.path.dirname(result_filename))
    with open(result_filename,'a') as f:
        if data_args.kfold == 0:
            f.write('{}\n'.format('\t'.join(result_to_save.keys())))
        f.write('{}\n'.format('\t'.join([str(v) for v in result_to_save.values()])))
    logger.info("Saved results {}".format(highlight(result_filename)))


def save_results_for_v3(training_args, data_args, results, verbose=True):
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(highlight(pformat(results[key]))))

    result_to_save_base = {'model':
                      os.path.basename(training_args.output_dir) + \
                      ('_heuristic=True' if training_args.use_heuristic else '') + \
                      ('_ensemble=True' if training_args.do_ensemble else ''),
                      'ensemble': training_args.do_ensemble,
                       'task': data_args.task,
                       'kfold':data_args.kfold }
    result_to_save_base['loss'] = -1

    for pdict in results["paired_results"]:
        result_to_save = result_to_save_base.copy()
        result_to_save["type_system"] = pdict["type_system"]
        task = pdict["type_system"][0]
        result_to_save["intent_acc"] = results["intent"][task]["intent_acc"]

        for k,v in pdict["results"].items():
            if type(v) == list:
                for lidx, vone in enumerate(v):
                    result_to_save['{}_{}'.format(k,lidx)] = vone
            else:
                result_to_save[k] = v


        if data_args.eval_data_file is not None:
            result_filename = os.path.join(data_args.result_dir,
                                    "{}_{}".format(
                                        "-".join(pdict["type_system"]),
                                        data_args.eval_data_file))
        else:
            result_filename = os.path.join(data_args.result_dir,
                                    "{}_{}".format(
                                        "-".join(pdict["type_system"]),
                                        "test.json"))

        if not os.path.exists(os.path.dirname(result_filename)):
            os.makedirs(os.path.dirname(result_filename))
        with open(result_filename,'a') as f:
            #TODO handle multi type_system paired cases
            if data_args.kfold == 0:
                f.write('{}\n'.format('\t'.join(result_to_save.keys())))
            f.write('{}\n'.format('\t'.join([str(v) for v in result_to_save.values()])))
        logger.info("Saved results {}".format(highlight(result_filename)))



if __name__ == '__main__':
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)

