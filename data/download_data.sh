#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/recipenlg

curl -L -o $SCRIPT_DIR/recipenlg/recipenlg.zip\
  https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/recipenlg
unzip $SCRIPT_DIR/recipenlg/recipenlg.zip -d $SCRIPT_DIR/recipenlg/

mkdir -p $SCRIPT_DIR/epirecipes
curl -L -o $SCRIPT_DIR/epirecipes/epirecipes.zip\
  https://www.kaggle.com/api/v1/datasets/download/hugodarwood/epirecipes
unzip $SCRIPT_DIR/epirecipes/epirecipes.zip -d $SCRIPT_DIR/epirecipes/
