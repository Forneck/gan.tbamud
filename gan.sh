#!/bin/bash

# Loop para repetir os comandos 3 vezes
echo "Gerando novos textos falsos:"
python fake-gen.py

echo "Iniciando treinamento"
python gan.py --num_epocas 50

echo""

