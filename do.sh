#!/bin/bash
git pull
python SENet_model.py
python VGG_16_model.py
python train.py
tensorboard --logdir=runs --port=6006 --host 0.0.0.0