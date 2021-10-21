#!/bin/bash

DATA=./data_files/
mkdir -p $DATA

# Abbreviation-Expansion (AI2020)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/data/AI2020.zip -P $DATA/
unzip $DATA/AI2020.zip -d $DATA/

# Term-Definition (W00)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/data/W00.zip -P $DATA/
unzip $DATA/W00.zip -d $DATA/

# DocDef (DocDef2)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/data/DocDef2.zip -P $DATA/
unzip $DATA/DocDef2.zip -d $DATA/

# DocDef Query format (DocDefQueryInplace)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/data/DocDefQueryInplace.zip -P $DATA/
unzip $DATA/DocDefQueryInplace.zip -d $DATA/


