import os
import torch
import transformers
import argparse
import io
import nltk
import pickle
import collections
import json
import random
from torch.nn.utils.rnn import pad_sequence

pasta = os.path.expanduser('~/mud/gan/v1/')
types = ['.mob']
textos_falsos = {}
def carregar_vocabulario(pasta, types):
    palavra_para_numero = {}
    numero_para_palavra = {}
    textos_reais = {}

    for tipo in types:
        print(f'Carregando os arquivos {tipo[1:]}.pt')
        textos_reais[tipo] = torch.load(os.path.join(pasta, tipo[1:] + '.pt'))

        print(f'Carregando o vocabulário para o tipo {tipo}')
        # Correção na formatação do nome do arquivo JSON
        with open(os.path.join(pasta, f'vocabulario{tipo}.json'), 'r') as f:
            palavra_para_numero[tipo] = json.load(f)
            # Criando o dicionário numero_para_palavra
            numero_para_palavra[tipo] = {i: palavra for palavra, i in palavra_para_numero[tipo].items()}

    return palavra_para_numero, numero_para_palavra, textos_reais

palavra_para_numero, numero_para_palavra,textos_reais = carregar_vocabulario(pasta, types)

for tipo in types:
    textos_falsos = []
    random_list = []
    fake = 'fake_' + tipo[1:] + '.pt'
    textos_falsos = []  # Inicialize textos_falsos fora do loop
    textos_unpad = []
    for texto in textos_reais[tipo]:
             texto = texto.tolist()
             while texto[-1] == 0:
                 texto.pop()
             textos_unpad.append(texto)
    min_length = min(len(texto) for texto in textos_unpad)
    for i in range(len(textos_reais[tipo])):
        random_list = []  # Inicialize random_list dentro do loop
        porcentagem = (i/len(textos_reais[tipo]))*100
        print(f'Gerando texto falso {i} de {len(textos_reais[tipo])} ({porcentagem:.4f}%)')
        for _ in range(random.randint(min_length,max([len(t) for t in textos_reais[tipo]]))):
            random_number = random.randint(0, len(palavra_para_numero[tipo]) - 1)
            random_list.append(random_number) 
        random_list = torch.tensor(random_list, dtype=torch.float32)
        textos_falsos.append(random_list)
    
    print('Salvando progresso')
    max_length = max([len(t) for t in textos_reais[tipo]])   
    # Padronizando o tamanho dos textos falsos
    textos_falsos_pad = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t)))) for t in textos_falsos], batch_first=True)
    # Salvando os textos falsos em arquivos .pt
    torch.save(textos_falsos_pad, fake) 
