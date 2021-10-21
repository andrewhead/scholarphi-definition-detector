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
        #echo "Testing (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        ##-m torch.distributed.launch
        #CUDA_VISIBLE_DEVICES=$GPU \
        #python -W ignore main.py \
            #--model_name_or_path=$MODEL \
            #--data_dir ${DATA_DIR} \
            #--output_dir ${DATA_DIR}/models/${VERSION}/${DATASET}/${K}/${MODEL} \
            #--prediction_dir ${DATA_DIR}/predictions/${VERSION}/${DATASET}/${K}/${MODEL} \
            #--result_dir ${DATA_DIR}/results/${VERSION}/${DATASET}/${K}/${MODEL} \
            #--task ${DATASET} --kfold $K --num_fold 5 \
            #--use_crf \
            #--use_heuristic \
            #--do_eval --use_pos --use_np --use_vp --use_entity --use_acronym \
            #--per_device_eval_batch_size ${BATCH} \
            #--max_seq_len ${MAXLEN}  \
            #--overwrite_cache  \
            #--use_test_set_for_validation \
            #--data_limit -1 \
            #--joint_learning
        ##--use_nickname_detector  --merge_predictions_for_symbol union
        #echo "============================================"
        ##exit
    #done
    #exit

    #DATASET=AI2020 #W00+WFM
    #echo "============================================"
    #echo "Testing (POS,NP,VP,Entity,Acronym)..." $MODEL ${DATASET} ${BATCH} ${MAXLEN}
    ##-m torch.distributed.launch
    #CUDA_VISIBLE_DEVICES=$GPU \
    #python -W ignore main.py \
        #--model_name_or_path=$MODEL \
        #--data_dir ${DATA_DIR} \
        #--output_dir ${DATA_DIR}/models/${VERSION}/${DATASET}/${MODEL} \
        #--prediction_dir ${DATA_DIR}/predictions/${VERSION}/${DATASET}/${MODEL} \
        #--result_dir ${DATA_DIR}/results/${VERSION}/${DATASET}/${MODEL} \
        #--task ${DATASET}  \
        #--use_crf \
        #--use_heuristic \
        #--do_eval --use_pos --use_np --use_vp --use_entity --use_acronym \
        #--per_device_eval_batch_size ${BATCH} \
        #--max_seq_len ${MAXLEN}  \
        #--overwrite_cache  --joint_learning # --use_acronym_detector
    ## --joint_learning \
    #echo "============================================"

    DATASET=W00
    KFOLD=(0 1 2 3 4 5 6 7 8 9)
    for K in "${KFOLD[@]}"
    do
        echo "============================================"
        echo "Testing (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        #-m torch.distributed.launch
        CUDA_VISIBLE_DEVICES=$GPU \
        python -W ignore main.py \
            --model_name_or_path=$MODEL \
            --data_dir ${DATA_DIR} \
            --output_dir ${DATA_DIR}/models/${VERSION}/${DATASET}/${K}/${MODEL} \
            --prediction_dir ${DATA_DIR}/predictions/${VERSION}/${DATASET}/${K}/${MODEL} \
            --result_dir ${DATA_DIR}/results/${VERSION}/${DATASET}/${K}/${MODEL} \
            --task ${DATASET} --kfold $K --num_fold 10 \
            --use_crf \
            --use_heuristic \
            --do_eval --use_pos --use_np --use_vp --use_entity --use_acronym \
            --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN}  \
            --overwrite_cache \
            --joint_learning
        echo "============================================"
        #exit
    done



done
