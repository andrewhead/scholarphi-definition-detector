#!/bin/bash

DATASET=W00 #+WFM
EVALDATASET=DocDef
DATA_DIR=$HOME/data/ScholarPhi/
KFOLD=(0 1 2 3 4 5 6 7 8 9)
GPU=0

#bert-base-uncased bert-large-uncased roberta-large roberta-base) #
MODELS=(roberta-large) #bert-base-uncased) #roberta-large allenai/scibert_scivocab_uncased) # bert-large-uncased) # bert-large-uncased) #albert-large-v2) #allenai/scibert_scivocab_uncased) # albert-base-v2
# allenai/cs_roberta_base r

for MODEL in "${MODELS[@]}"
do
    #NOTE for longformer: change max_seq_len from 80 to 512, batch_size from 32 to 8
    if [[ $MODEL = allenai/longformer* ]]
    then
        BATCH=8
        MAXLEN=512
    else
        BATCH=32
        MAXLEN=80
    fi

    for K in "${KFOLD[@]}"
    do

        echo "============================================"
        echo "Testing (POS,NP,VP,Entity,Acronym) with heuristic..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        CUDA_VISIBLE_DEVICES=$GPU \
        python main.py \
            --model_name_or_path=$MODEL \
            --data_dir ${DATA_DIR} \
            --output_dir ${DATA_DIR}/joint_bert/${DATASET}/${K}/${MODEL} \
            --task ${EVALDATASET} --kfold $K \
            --use_crf \
            --do_eval \
            --use_nickname_detector --use_acronym_detector \
            --use_pos --use_np --use_vp --use_entity --use_acronym \
            --use_heuristic \
            --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN} --overwrite_cache
        echo "============================================"
        exit

        #echo "============================================"
        #echo "Testing (POS,NP,VP,Entity,Acronym) with heuristic and ensembling..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        #CUDA_VISIBLE_DEVICES=$GPU \
        #python main.py \
            #--model_name_or_path=$MODEL \
            #--data_dir ${DATA_DIR} \
            #--output_dir ${DATA_DIR}/joint_bert/${DATASET}/${K}/${MODEL} \
            #--task ${EVALDATASET} --kfold $K \
            #--use_crf \
            #--do_eval --use_pos --use_np --use_vp --use_entity --use_acronym \
            #--use_heuristic \
            #--do_ensemble \
            #--per_device_eval_batch_size ${BATCH} \
            #--max_seq_len ${MAXLEN} --overwrite_cache
        #echo "============================================"

    done

done



