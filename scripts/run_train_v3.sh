#!/bin/bash

#DATASET=W00+WFM
#DATASET=DocDef2 #AI2020 #W00+WFM
DATA_DIR=$HOME/data/ScholarPhi/
GPU=0
VERSION=v3.2
MODELS=(roberta-large) # bert-large-uncased allenai/scibert_scivocab_uncased) #roberta-large_finetuned_s2orc10K)


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


    #DATASET=DocDef2 #AI2020 #W00+WFM
    #KFOLD=(0 1 2 3 4)
    #for K in "${KFOLD[@]}"
    #do
        #echo "============================================"
        #echo "Training (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        ##-m torch.distributed.launch
        #CUDA_VISIBLE_DEVICES=$GPU \
        #python -W ignore main.py \
            #--model_name_or_path=$MODEL \
            #--data_dir ${DATA_DIR} \
            #--output_dir ${DATA_DIR}/models/${VERSION}/${DATASET}/${K}/${MODEL} \
            #--task ${DATASET} --kfold $K --num_fold 5 \
            #--use_crf \
            #--use_heuristic \
            #--do_train --use_pos --use_np --use_vp --use_entity --use_acronym \
            #--per_device_train_batch_size ${BATCH} --per_device_eval_batch_size ${BATCH} \
            #--max_seq_len ${MAXLEN} --learning_rate 2e-5 \
            #--num_train_epochs 30 --logging_steps 800 --save_steps 800 \
            #--overwrite_cache --overwrite_output_dir \
            #--use_test_set_for_validation --joint_learning
        #echo "============================================"
        ##exit
    #done


    #DATASET=AI2020 #W00+WFM
    #echo "============================================"
    #echo "Training (POS,NP,VP,Entity,Acronym)..." $MODEL ${DATASET} ${BATCH} ${MAXLEN}
    ##-m torch.distributed.launch
    #CUDA_VISIBLE_DEVICES=$GPU \
    #python -W ignore main.py \
        #--model_name_or_path=$MODEL \
        #--data_dir ${DATA_DIR} \
        #--output_dir ${DATA_DIR}/models/${VERSION}/${DATASET}/${MODEL} \
        #--task ${DATASET}  \
        #--use_crf \
        #--use_heuristic \
        #--do_train --use_pos --use_np --use_vp --use_entity --use_acronym \
        #--per_device_train_batch_size ${BATCH} --per_device_eval_batch_size ${BATCH} \
        #--max_seq_len ${MAXLEN} --learning_rate 2e-5 \
        #--num_train_epochs 30 --logging_steps 800 --save_steps 800 \
        #--overwrite_cache --overwrite_output_dir --joint_learning
    ##--use_nickname_detector --use_acronym_detector \
    ## --joint_learning \
    #echo "============================================"

    DATASET=W00
    KFOLD=(0 1 2 3 4 5 6 7 8 9)
    for K in "${KFOLD[@]}"
    do
        echo "============================================"
        echo "Training (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        #-m torch.distributed.launch
        CUDA_VISIBLE_DEVICES=$GPU \
        python -W ignore main.py \
            --model_name_or_path=$MODEL \
            --data_dir ${DATA_DIR} \
            --output_dir ${DATA_DIR}/models/${VERSION}/${DATASET}/${K}/${MODEL} \
            --task ${DATASET} --kfold $K --num_fold 10 \
            --use_crf \
            --use_heuristic \
            --do_train --use_pos --use_np --use_vp --use_entity --use_acronym \
            --per_device_train_batch_size ${BATCH} --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN} --learning_rate 2e-5 \
            --num_train_epochs 30 --logging_steps 1000 --save_steps 1000 \
            --overwrite_cache --overwrite_output_dir \
            --use_test_set_for_validation --joint_learning
        echo "============================================"
        #exit
    done



done
