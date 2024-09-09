#!/bin/bash
echo "Dependencies Install"
pip install torch
pip install nltk
pip install transformers
echo "Downloading stopwords"
python stopwords.py
echo "Setup done"
