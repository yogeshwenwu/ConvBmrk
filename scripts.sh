#!/bin/bash

source /mnt/mydisk/yogesh/anaconda3/etc/profile.d/conda.sh

conda activate env_py_3.11_torch_2.5
echo "Activated Env------->"

python model.py

conda deactivate
echo "Deactivated the Env------->"
