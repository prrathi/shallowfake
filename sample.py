import pandas as pd
from process import *
import nn.tinypix2pix_mod as tinypix2pix_mod
import subprocess

# We assume that the directory is in the following format, with paired files between clean and noise directories:
# ├── clean/
#     ├── train/
#         ├── file1.wav
#         └── ...
#     ├── valid/
#         └── ...
#     └── test/
#         └── ...
# ├── noise/
#     ├── train/
#         ├── file1.wav
#         └── ...
#     ├── valid/
#         └── ...
#     └── test/
#         └── ...
# └── gen/
#     └── test/
# ├── nn/
#     ├── data/
#     ├── tinypix2pix_mod.py
# └── labels.csv

# create words length dictionary
# for this example, the data is taken from LibriSpeech dataset
labels = pd.read_csv('./labels.csv')
words_count = labels['wrd'].str.split().str.len()
words_dict = dict(zip(labels['ID'], words_count))

# preprocess the splits
preprocess(source_adv='./source/noise/', source_nat='./source/clean/', split_noise='./nn/data/noise/train/', split_nat='./nn/data/clean/train/', split_output='./nn/data/', words_dict=words_dict, split_label='train', label='train')
preprocess(source_adv='./source/noise/', source_nat='./source/clean/', split_noise='./nn/data/noise/valid/', split_nat='./nn/data/clean/valid/', split_output='./nn/data/', words_dict=words_dict, split_label='valid', label='valid')
preprocess(source_adv='./source/noise/', source_nat='./source/clean/', split_noise='./nn/data/noise/test/', split_nat='./nn/data/clean/test/', split_output='./nn/data/', words_dict=words_dict, split_label='test', label='test')

# train model
subprocess.run(['sh', 'sample.sh'])

# postprocess predictions
postprocess(output='./nn/data/output/', source_nat='./source/clean/', split_output='./nn/data/', final_write='', split_label='test', label='test')