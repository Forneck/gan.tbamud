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

print('Iniciando treinamento do Gerador e Avaliador')
agora = datetime.datetime.now()
timestamp = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Inicio da sessão: {timestamp}')
#stats = f'session-quality-{timestamp}.json'
pasta = os.path.expanduser('~/gan/v1')
# Tipos de arquivos que você quer gerar
types = ['.mob']

parser = argparse.ArgumentParser()
parser.add_argument('--save_mode', choices=['local', 'nuvem'], default='local', help='Escolha onde salvar o modelo')
parser.add_argument('--save_time', choices=['epoch', 'session'], default='epoch', help='Escolha quando salvar o modelo')
parser.add_argument('--num_epocas', type=int, default=100, help='Número de épocas para treinamento')
parser.add_argument('--num_samples', type=int, default=1, help='Número de amostras para cada época')
parser.add_argument('--noise_dim', type=limit_noise_dim, default=100, help='Dimensão do ruído para o gerador')
parser.add_argument('--noise_samples', type=int,default=1, help='Número de amostras de ruído para o gerador')
parser.add_argument('--verbose', choices=['on', 'off'], default='on', help='Mais informações de saída')
parser.add_argument('--modo', choices=['auto','manual', 'real'],default='real', help='Modo do Prompt: auto, manual ou real')
parser.add_argument('--debug', choices=['on', 'off'], default='off', help='Debug Mode')
parser.add_argument('--treino', choices=['abs','rel'], default='rel', help='Treino Absoluto ou Relativo')
args = parser.parse_args()

num_classes = 2 #0 e 1
num_numeros = 18 #quantidade de campos numericos
if args.verbose == 'on':
    print('Definindo os parâmetros de treinamento')
num_epocas = args.num_epocas
taxa_aprendizado_gerador = 0.01 #inicial 0.0001
noise_dim = args.noise_dim # entre 1 e 100
noise_samples = 1
debug = args.debug
modo = args.modo
num_samples = args.num_samples #numero de amostras dentro da mesma época
treino = args.treino
textos_falsos = {}

if args.verbose == 'on':
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

if args.verbose == 'on':
    print('Definindo a arquitetura do modelo avaliador')
class Avaliador(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_numeros):
        super(Avaliador, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc_texto = nn.Linear(hidden_dim, num_classes)
        self.fc_numeros = nn.Linear(num_numeros, num_classes)
        self.fc_final = nn.Linear(num_classes * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, texto, numeros, hidden=None):
        # Processamento do texto
        embedded_texto = self.embedding(texto).transpose(1, 2)
        conv_texto = F.relu(self.conv1d(embedded_texto)).transpose(1, 2)
        lstm_out, hidden = self.lstm(conv_texto, hidden)
        
        # Atenção
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        texto_contexto = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Processamento dos números
        numeros_contexto = F.relu(self.fc_numeros(numeros))
        
        # Combinação das características de texto e números
        combined = torch.cat((texto_contexto, numeros_contexto), dim=1)
        
        # Classificação final
        output = self.fc_final(combined)
        output = self.softmax(output)
        return output, hidden

def compare_state_dicts(dict1, dict2):
    for (key1, tensor1), (key2, tensor2) in zip(dict1.items(), dict2.items()):
        if torch.equal(tensor1, tensor2):
            print(f'Os pesos para {key1} não mudaram.')
        else:
            print(f'Os pesos para {key1} mudaram.')

if args.verbose == 'on':
    print('Definindo o Encoder')
def encoder(texto, tipo, palavra_para_numero):
    return [palavra_para_numero[tipo].get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
    #return [palavra_para_numero[tipo].get(palavra, 0) for palavra in palavras]

if args.verbose == 'on':
    print('Definindo o Decoder')
def decoder(texto_codificado, tipo, numero_para_palavra):
      return ' '.join([numero_para_palavra[tipo].get(numero, '<UNK>') for numero in texto_codificado])

if args.verbose == 'on':
    print('Definindo o Vocabulario')
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

max_length = {}
min_length = {}
for tipo in types:
    if args.verbose == 'on':
        print(f"Formato dos textos reais para o tipo {tipo}: {textos_reais[tipo].shape}")
    max_length[tipo] = max([len(t) for t in textos_reais[tipo]])
    rotulos = [1]*len(textos_reais[tipo])
    textos_unpad = []
    for texto in textos_reais[tipo]:
         texto = texto.tolist()
         while texto[-1] == 0:
            texto.pop()
         textos_unpad.append(texto)
 
    min_length[tipo] = min(len(texto) for texto in textos_unpad)


