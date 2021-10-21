#!/bin/bash

#DATASET=W00+WFM
#DATASET=DocDef2 #AI2020 #W00+WFM

DATASETS=(W00+AI2020+DocDef2) #QueryInplaceFixedMIA) # DocDefQueryInplaceFixed DocDefQueryInplace DocDef2) #DocDef2MIA DocDef2)

DATA_DIR=$HOME/data/ScholarPhi/v4/data/
MODEL_DIR=$HOME/data/ScholarPhi/v5/model/
GPU=0
VERSION=v5
MODELS=(roberta-large)


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
        BATCH=12 #8 #32
        MAXLEN=100
    fi

    #KFOLD=(0 1 2 3 4 5 6 7 8 9)
    #for K in "${KFOLD[@]}"
    #do
    echo "============================================"
    echo "Training ..." $MODEL ${VERSION} ${DATASET} ${BATCH} ${MAXLEN}
    #-m torch.distributed.launch
    CUDA_VISIBLE_DEVICES=$GPU \
    python -W ignore main.py \
        --model_name_or_path=$MODEL \
        --data_dir ${DATA_DIR} \
        --output_dir ${MODEL_DIR}/${VERSION}/${DATASET}/MAXLEN=${MAXLEN}/${MODEL} \
        --task ${DATASET} --dataset ${DATASET}  \
        --use_crf \
        --use_heuristic \
        --do_train --use_pos --use_np --use_vp --use_entity --use_acronym \
        --per_device_train_batch_size ${BATCH} --per_device_eval_batch_size ${BATCH} \
        --max_seq_len ${MAXLEN} --learning_rate 2e-5 \
        --num_train_epochs 50 --logging_steps 500 --save_steps 500 \
        --overwrite_cache --overwrite_output_dir \
        --use_test_set_for_validation --joint_learning
    echo "============================================"
    #--kfold $K --num_fold 10
    #exit
    #done
done
done
