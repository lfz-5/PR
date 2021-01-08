#!/bin/bash
git pull
python SENet_model.py
python VGG_16_model.py
python train.py