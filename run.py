import os
import yaml

from njem.nejm.model_training.train_model import train

config = yaml.load('config.yaml')

run = config['run']

if run['train_rnn']:
    train()