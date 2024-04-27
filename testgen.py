import os
import torch
import transformers
import argparse
import io
import nltk
import pickle
import json
import collections
import datetime
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
print('Iniciando teste do Gerador')
agora = datetime.datetime.now()
inicio = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Inicio da sessão: {inicio}')
gerador_path = 'gerador_mob.pt'
types= ['.mob']
pasta = os.path.expanduser('~/mud/gan/v1')

# Definindo o argumento para escolher entre salvar localmente ou na nuvem
parser = argparse.ArgumentParser()
parser.add_argument('--quantidade', type=int, default=1,help='Quantidade de textos gerados')
parser.add_argument('--modo', choices=['auto','manual'],default='manual', help='Modo do Prompt: auto ou manual')
parser.add_argument('--debug', choices=['on', 'off'], default='off', help='Debug Mode')
parser.add_argument('--verbose', choices=['on', 'off'], default='on', help='Mostra mais informaçōes de saida')
args = parser.parse_args()

quantidade = args.quantidade
modo = args.modo
debug = args.debug
verbose = args.verbose

if verbose == 'on':
    print('Definindo a arquitetura do modelo gerador')
class Gerador(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_size):
        super(Gerador, self).__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden

def carregar_vocabulario(pasta, types):
    palavra_para_numero = {}
    numero_para_palavra = {}
    textos_reais = {}

    for tipo in types:
        if verbose == 'on':
           print(f'Carregando os arquivos {tipo[1:]}.pt')
        textos_reais[tipo] = torch.load(os.path.join(pasta, tipo[1:] + '.pt'))

        if verbose == 'on':
            print(f'Carregando o vocabulário para o tipo {tipo}')
        # Correção na formatação do nome do arquivo JSON
        with open(os.path.join(pasta, f'vocabulario{tipo}.json'), 'r') as f:
            palavra_para_numero[tipo] = json.load(f)
            # Criando o dicionário numero_para_palavra
            numero_para_palavra[tipo] = {i: palavra for palavra, i in palavra_para_numero[tipo].items()}

    return palavra_para_numero, numero_para_palavra, textos_reais

if args.verbose == 'on':
    print('Definindo o Encoder')
def encoder(texto, tipo, palavra_para_numero):
    return [palavra_para_numero[tipo].get(palavra, 0) for palavra in nltk.word_tokenize(texto)]

if verbose == 'on':
   print('Definindo o Decoder')
def decoder(texto_codificado, tipo, numero_para_palavra):
       # Decodificar o texto usando o dicionário numero_para_palavra do tipo de arquivo correspondente
      return ' '.join([numero_para_palavra[tipo].get(numero, '<UNK>') for numero in texto_codificado])

palavra_para_numero, numero_para_palavra,textos_reais = carregar_vocabulario(pasta, types)

if verbose == 'on':
    print('Carregando o modelo gerador')
gerador = torch.load(gerador_path)

def gerar_texto_falso(gerador, quantidade, tipo):
    # Gerando textos falsos
    with torch.no_grad():
        print(f'Colocando o modelo em modo de avaliação.')
        gerador.eval()
        epoca = 1
        while epoca <= quantidade:
           noise_dim = torch.randint(40,253, (1,))
           if modo == 'manual': 
              prompt = input(f'> ')
              if len(prompt.strip())==0:
                  prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,noise_dim))
              else:
                  prompt = encoder(prompt,tipo,palavra_para_numero)
                  prompt = torch.tensor(prompt).unsqueeze(0)
                  if prompt.size(1) < noise_dim:
                      noise = torch.randint(0,len(numero_para_palavra[tipo]),(1,noise_dim - prompt.size(1)))
                      for noise_samp in noise:
                          prompt = pad_sequence([torch.cat((t, noise_samp)) for t in prompt], batch_first=True)
           elif modo == 'auto':
                #modo 'auto'
               prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,noise_dim))
               decoded = decoder(prompt[0].tolist(),tipo,numero_para_palavra)
               if args.verbose == 'on':
                   print(f'Prompt aleatorio: {decoded}')

           lprompt = prompt.to(torch.int64)
           saida,lstm = gerador(lprompt)
           print(f'{saida.shape}')
           if debug == 'on':
              print(f'Saida Bruta do Gerador: {saida}')
              print(f'Estado oculto do LSTM - Memória do Gerador: {lstm}')
           if verbose == 'on':
               texto_falso_max = torch.argmax(saida, dim=-1)
               texto_falso_max = texto_falso_max.to(torch.int64)
               saida = decoder(texto_falso_max[0].tolist(),tipo,numero_para_palavra)
               print(f'Saida do Gerador: {saida}')
           epoca = epoca + 1

for tipo in types:
     # Gerando o texto falso
     gerar_texto_falso(gerador, quantidade, tipo)

agora = datetime.datetime.now()
fim = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Início da sessão de teste do gerador em {inicio} com fim em {fim}')
