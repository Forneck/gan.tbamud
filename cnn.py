import torch.nn as nn
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

print('Treinamento e Validação da Cnn')
types = ['.mob']
pasta = os.path.expanduser('~/mud/gan/v1')
agora = datetime.datetime.now()
timestamp = agora.strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', choices=['on', 'off','cnn'], default='cnn', help='Mais informações de saída')
parser.add_argument('--num_epocas', type=int, default=1, help='Número de épocas para treinamento')
parser.add_argument('--num_samples', type=int, default=1, help='Número de amostras para cada época')
parser.add_argument('--limiar', type=float, default=0.51, help='Limiar minimo para considerar real punfalso. EM PONTO FLUTUANTE. De 0 a .9995')
args = parser.parse_args()
taxa_aprendizado_cnn = 0.0001
num_epocas = args.num_epocas 
tamanho_lote = 1 #args.tamanho_lote 
num_samples = args.num_samples #numero de amostras dentro da mesma época
limiar = args.limiar

class Cnn(nn.Module):
    def __init__(self, vocab_size, embedding_dim,output_size):
        super(Cnn, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        x = embedded.permute(0, 2, 1)  # Change shape to (batch_size, input_channels, sequence_length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(2)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class TextDataset(Dataset):
    def __init__(self, textos, rotulos):
        self.textos = textos
        self.rotulos = rotulos

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        return self.textos[idx], self.rotulos[idx]

def encoder(palavras, tipo, palavra_para_numero):
   # return [palavra_para_numero[tipo].get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
    return [palavra_para_numero[tipo].get(palavra, 0) for palavra in palavras]

if args.verbose == 'on':
    print('Definindo o Decoder')
def decoder(texto_codificado, tipo, numero_para_palavra):
      return ' '.join([numero_para_palavra[tipo].get(numero, '<UNK>') for numero in texto_codificado])

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
vocab_size = len(numero_para_palavra)

# Example usage:
cnn_output = 2  # Binary classification (prob real vs. prob fake)
criterio_cnn = torch.nn.MSELoss() #pode ser a BCELoss, mas a MSE penaliza as maiores diferenças entre a saida e os rótulos.

textos_falsos = {}

# Inicializar os DataLoaders para cada tipo
train_loaders, valid_loaders, test_loaders = {}, {}, {}
for tipo in types:
    textos_falsos[tipo] = []
    fake = 'fake_' + tipo[1:] + '.pt'
    textos_falsos[tipo] = torch.load(fake) 

    if args.verbose == 'on':
        print("Formato dos textos reais:",textos_reais[tipo].shape)
        print("Formato dos textos falsos:", textos_falsos[tipo].shape)
    if args.verbose == 'on':
        print('Padronizando o tamanho dos textos reais e falsos')
    max_length = max(max([len(t) for t in textos_reais[tipo]]), max([len(t) for t in textos_falsos[tipo]]))
# Criando uma lista vazia para os textos reais sem padding
    textos_unpad = []
    for texto in textos_reais[tipo]:
         texto = texto.tolist()
         while texto[-1] == 0:
            texto.pop()
         textos_unpad.append(texto)
 
    min_length = min(len(texto) for texto in textos_unpad)
    if args.verbose == 'on':
        print(f'Min leght for real text: {min_length}')
    textos_reais_pad = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t)))) for t in textos_reais[tipo]], batch_first=True)
    textos_falsos_pad = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t)))) for t in textos_falsos[tipo]], batch_first=True)

    if args.verbose == 'on':
        print(' Combinando os textos reais e os textos falsos')
    textos = torch.cat((textos_reais_pad, textos_falsos_pad), dim=0)

    if args.verbose == 'on':
       print('Atribuir rótulos binários para cada texto')
    rotulos = [1]*len(textos_reais[tipo]) + [0]*len(textos_falsos[tipo])

    if args.verbose == 'on':
       print('Tokenizar e codificar os textos')
    #tokenized_textos = [encoder(decoder(texto,tipo), tipo) for texto in textos]
    textos = textos.to(torch.int64)
    tokenized_textos = []
    for i in range(textos.size(0)):  # Itera sobre a primeira dimensão do tensor
        texto = textos[i].tolist()  # Converte o tensor para uma lista
        texto_decodificado = decoder(texto, tipo, numero_para_palavra)
        tokenized_textos.append(torch.tensor(encoder(nltk.word_tokenize(texto_decodificado), tipo, palavra_para_numero)))
   
    # Padronizar o tamanho dos textos
    textos_pad = pad_sequence(tokenized_textos, batch_first=True)
    
    # Criar o conjunto de dados
    dataset = TextDataset(textos_pad, rotulos)

    # Dividir o conjunto de dados
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    # Criar os DataLoaders
    train_loaders[tipo] = DataLoader(train_dataset, batch_size=tamanho_lote, shuffle=True)
    valid_loaders[tipo] = DataLoader(valid_dataset, batch_size=tamanho_lote)
    test_loaders[tipo] = DataLoader(test_dataset, batch_size=tamanho_lote)

# Criando os modelos gerador e cnn para cada tipo de texto
gerador, cnn = {}, {}
for tipo in types:
    output_size = max(max([len(t) for t in textos_reais[tipo]]), max([len(t) for t in textos_falsos[tipo]]))

    # Caminhos dos modelos
    cnn_path = os.path.expanduser('cnn_' + tipo[1:] + '.pt')

    print('Verificando se a cnn existe para o tipo: ', tipo[1:])
    if os.path.exists(cnn_path):
        print('Carregar o cnn')
        cnn[tipo] = torch.load(cnn_path)
    else:
        print('Criar novo cnn')
        cnn[tipo] = Cnn(len(numero_para_palavra[tipo]),output_size, cnn_output)

# Criando os otimizadores para cada modelo
otimizador_cnn, otimizador_gerador = {}, {}
for tipo in types:
    otimizador_cnn[tipo] = torch.optim.Adam(cnn[tipo].parameters(), lr=taxa_aprendizado_cnn)

for epoca in range(num_epocas):
    for tipo in types:
        print(f'Colocando os modelos em modo de treinamento para epoca {epoca + 1}')
        cnn[tipo].train()
        if args.verbose == 'on':
            print('Inicializando as perdas e as acurácias')
        perda_cnn = 0
        acuracia_cnn = 0
        if args.verbose == 'on':
            print('Percorrendo cada lote de dados')
        for (textos, rotulos), amostra in zip(train_loaders[tipo], range(num_samples)):
            if args.verbose == 'on':
                print('Obtendo os textos e os rótulos do lote / amostra')
                print(f'Texto treinamento: {textos} e o rótulo: {rotulos}')
            if args.verbose == 'cnn':
                print(f'Rotulo do texto de treinamento: {rotulos}')
            if args.verbose == 'on':
                print('Zerando a acurácia para a amostra')
            acuracia_cnn = 0
            # Passando o texto para a cnn
            saida_real = cnn[tipo](textos)
            
            if args.verbose == 'on' or args.verbose == 'cnn':
                print(f'Saida da cnn com textos de treinamento {saida_real}')
            rotulos_float = rotulos.float()
            rotulos_reshaped = rotulos_float.view(-1, 1).repeat(1, 2)
            #Identifica com os rotulos certos
            perda_real = criterio_cnn(saida_real, rotulos_reshaped)
            perda_cnn = perda_real
            if args.verbose == 'on':
                print('Atualizando os parâmetros do cnn')
            otimizador_cnn[tipo].zero_grad()
            perda_cnn.backward()
            otimizador_cnn[tipo].step()
            saida_real = cnn[tipo](textos)
            print(f'Saida da Cnn apos otimizacao: {saida_real}')
            if args.verbose == 'on':
                print('Calculando a acurácia da cnn')
            acuracia_cnn += ((saida_real[:,0] > limiar) == torch.ones_like(rotulos)).float().mean()
            acuracia_cnn += ((saida_real[:,1] < limiar) == torch.zeros_like(rotulos)).float().mean()
            # Imprimindo as perdas e as acurácias
            print(f'Tipo {tipo}, Epoca {epoca + 1} de {num_epocas},Limiar {limiar}, Perda cnn {perda_cnn.item():.4f}, Acuracia cnn {acuracia_cnn.item() / 2:.4f}')
            if args.verbose == 'on' or args.verbose == 'cnn':
                print('Salvando modelo')
            torch.save(cnn[tipo], os.path.expanduser('cnn_' + tipo[1:] + '.pt'))
   
#cnn = Cnn(input_channels, cnn_output)
#print(cnn)
