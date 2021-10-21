#!/bin/bash
#export PYTHONPATH="$HOME/work/scholarphi_nlp_internal/code:$PYTHONPATH"
export PYTHONPATH="$HOME/work/scholarphi_nlp_internal/code/HEDDEx:$PYTHONPATH"
MODELNAME=DocDefQueryInplaceFixedMIA
python inference/inference_server.py --task $MODELNAME --model $HOME/data/ScholarPhi/v4/model/v5/$MODELNAME/MAXLEN=100/
