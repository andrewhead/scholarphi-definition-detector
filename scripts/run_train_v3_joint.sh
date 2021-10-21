#!/bin/bash

#DATASET=DocDef2+AI2020 #+W00 #+WFM
DATASETS=(DocDef2+AI2020+W00) # DocDef2+W00 W00+AI2020 DocDef2+AI2020+W00)
#TODO shuffle?
DATA_DIR=$HOME/data/ScholarPhi
GPU=0
VERSION=v3.2
MODELS=(roberta-large) # bert-large-uncased allenai/scibert_scivocab_uncased)


WANDB_DISABLED=true

for DATASET in "${DATASETS[@]}"
do
for MODEL in "${MODELS[@]}"
do
    #NOTE for longformer: change max_seq_len from 80 to 512, batch_size from 32 to 8
    if [[ $MODEL = allenai/longformer* ]]
    then
        BATCH=8
        MAXLEN=512
    else
        BATCH=16 #8 #32
        MAXLEN=80
    fi

    K=0
    echo "============================================"
    echo "Training (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
    #-m torch.distributed.launch
    CUDA_VISIBLE_DEVICES=$GPU \
    python -W ignore main.py \
        --model_name_or_path=$MODEL \
        --data_dir ${DATA_DIR} \
        --output_dir ${DATA_DIR}/models/${VERSION}/${DATASET}/${K}/${MODEL} \
        --task ${DATASET} \
        --use_crf \
        --use_heuristic \
        --do_train --use_pos --use_np --use_vp --use_entity --use_acronym \
        --per_device_train_batch_size ${BATCH} --per_device_eval_batch_size ${BATCH} \
        --max_seq_len ${MAXLEN} --learning_rate 2e-5 \
        --num_train_epochs 30 --logging_steps 1000 --save_steps 1000 \
        --overwrite_output_dir \
        --use_test_set_for_validation \
        --joint_learning #--overwrite_cache
    echo "============================================"
done
done
