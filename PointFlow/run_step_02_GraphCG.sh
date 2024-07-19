#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphCG

echo $@
date

echo "start"
python -u step_02_GraphCG.py $@

echo "end"
date
