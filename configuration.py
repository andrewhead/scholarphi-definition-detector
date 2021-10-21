import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    TrainingArguments
  )



# logger = logging.getLogger(__name__)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    ignore_index: Optional[int] = field(
        default=0,
        metadata={
            "help": "Specifies a target value that is ignored and does not contribute to the input gradient"},
    )
    slot_loss_coef: Optional[float] = field(
        default=1.0,
        metadata={"help": "Coeffcient for the slot loss"},
    )
    use_crf: bool = field(
        default=False, metadata={"help": "Wehther to use CRF"}
    )
    slot_pad_label: Optional[str] = field(
        default="PAD", metadata={"help": "Pad token for slot label pad (to be ignore when calculate loss)"}
    )

    dropout_rate: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Dropout for fully-connected layers"},
    )

    # additional linguistic features or heuristic filter
    use_pos: bool = field(
        default=False, metadata={"help": "Wehther to use POS embedding or not"})
    use_np: bool = field(
        default=False, metadata={"help": "Wehther to use NP embedding or not"})
    use_vp: bool = field(
        default=False, metadata={"help": "Wehther to use VP embedding or not"})
    use_entity: bool = field(
        default=False, metadata={"help": "Wehther to use Entity embedding or not"})
    use_acronym: bool = field(
        default=False, metadata={"help": "Wehther to use Acronym embedding or not"})
    use_heuristic: bool = field(
        default=False, metadata={"help": "Wehther to use heuristic filters or not"})


    # training options
    joint_learning: bool = field(
        default=False, metadata={"help": "Joint learning of given datasets (e.g., W00+AI2020+DocDef2"})
    do_ensemble: bool = field(
        default=False, metadata={"help": "Wehther to use model ensemble or not"})
    ensemble_method: Optional[str] = field(
        default="majority_voting", metadata={"help": "Ensemble method when do_ensemble=True (default: majority_voting)"}
    )
    use_test_set_for_validation: bool = field(
        default=False, metadata={"help": "If there is no dev set, use test set instead (don't tune your parameters with this though)"})

    # inference options
    use_nickname_detector: bool = field(
        default=False, metadata={"help": "Wehther to use heuristic-based nickname detector"})
    use_acronym_detector: bool = field(
        default=False, metadata={"help": "Wehther to use off-the-shelf based acronym detector"})
    merge_predictions_for_symbol: Optional[str] = field(
        default=None, metadata={"help": "When multiple systems are used, how to merge ouptuts for symbol types (default: none)"}
    )
    merge_predictions_for_acronym: Optional[str] = field(
        default=None, metadata={"help": "When multiple systems are used, how to merge ouptuts for acronym types (default: none)"}
    )
    dataconfig_file: Optional[str] = field(
      default="dataconfig.json", metadata={"help":
      '''
          The `dataconfig.json` file, that is bundled along with the joint model.
          It has intent_label dict, slot_label dict, and pos_label list) for all
          the datasets used in to train the model (and thus used during inference).
          Schema of the file :
              {
                  "intent_label" : {
                      <dataset_1> : <List of intent labels>,
                      <dataset_2> : <List of intent labels>,
                  },
                  "slot_label" : {
                      <dataset_1> : <List of slot labels>,
                      <dataset_2> : <List of slot labels>,
                  },
                  "pos_label" : <List of global POS Labels>,
              }

      '''},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    #       train_data_file: Optional[str] = field(
      # default=None, metadata={"help": "The input training data file (a text file)."}
    # )
    eval_data_file: Optional[str] = field(
      default=None,
      metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
      default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    task: Optional[str] = field(
      default=None,
      metadata={"help": "The name of the model's task"},
    )
    dataset: Optional[str] = field(
      default=None,
      metadata={"help": "The name of the dataset"},
    )

    kfold: Optional[int] = field(
      default=-1,
      metadata={"help": "TBW"},
    )
    num_fold: Optional[int] = field(
      default=-1,
      metadata={"help": "TBW"},
    )

    data_dir: Optional[str] = field(
      default='./data',
      metadata={"help": "The input data dir"},
    )
    # termdef_intent_label_file: Optional[str] = field(
      # default='termdef_intent_label.txt',
      # metadata={"help": "Term-Definition Intent label file"},
    # )
    # termdef_slot_label_file: Optional[str] = field(
      # default='termdef_slot_label.txt',
      # metadata={"help": "Term-Definition Slot label file"},
    # )
    # abbrexp_intent_label_file: Optional[str] = field(
      # default='abbrexp_intent_label.txt',
      # metadata={"help": "Abbreviation-Expansion Intent label file"},
    # )
    # abbrexp_slot_label_file: Optional[str] = field(
      # default='abbrexp_slot_label.txt',
      # metadata={"help": "Abbreviation-Expansion Slot label file"},
    # )
    intent_label_file: Optional[str] = field(
      default='intent_label.txt',
      metadata={"help": "Intent label file"},
    )
    slot_label_file: Optional[str] = field(
      default='slot_label.txt',
      metadata={"help": "Slot label file"},
    )
    pos_label_file: Optional[str] = field(
      default='pos_label.txt',
      metadata={"help": "POS label file"},
    )

    max_seq_len: Optional[int] = field(
      default=50,
      metadata={"help": "TBW"},
    )

    data_limit: Optional[int] = field(
      default=-1,
      metadata={"help": "TBW"},
    )


    result_dir: Optional[str] = field(
      default='./',
      metadata={"help": "The result dir"},
    )
    prediction_dir: Optional[str] = field(
      default='./',
      metadata={"help": "The prediction dir"},
    )



