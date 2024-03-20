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

types = ['.mob']
pasta = os.path.expanduser('~/mud/gan/v1')
agora = datetime.datetime.now()
timestamp = agora.strftime("%Y-%m-%d_%H-%M-%S")
stats = f'cnn-stats_{timestamp}.json'

# Inicialize um dicionário para armazenar as estatísticas
estatisticas = {
    'tipo': [],
    'epoca': [],
    'num_epocas': [],
    'perda_cnn': [],
    'perda_gerador': [],
    'acuracia_cnn': [],
    'acuracia_gerador': []
}

def limit_noise_dim(value):
    ivalue = int(value)
    if ivalue > 253:
        ivalue = 253
    if ivalue < 1:
        ivalue = 1
    return ivalue

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', choices=['on', 'off','cnn'], default='cnn', help='Mais informações de saída')
parser.add_argument('--num_epocas', type=int, default=1, help='Número de épocas para treinamento')
parser.add_argument('--tamanho_lote', type=int, default=1, help='Tamanho do lote para treinamento')
parser.add_argument('--num_samples', type=int, default=1, help='Número de amostras para cada época')
parser.add_argument('--noise_dim', type=limit_noise_dim, default=100, help='Dimensão do ruído para o gerador')
parser.add_argument('--noise_samples', type=int,default=1, help='Número de amostras de ruído para o gerador') 
args = parser.parse_args()

taxa_aprendizado_cnn = 0.001
taxa_aprendizado_gerador = 0.0001 #era 0.0001 mas demorava para aprender
num_epocas = args.num_epocas 
tamanho_lote = 1 #args.tamanho_lote 
noise_dim = args.noise_dim # entre 1 e 100
noise_samples = args.noise_samples #numero de amostras de ruído
num_samples = args.num_samples #numero de amostras dentro da mesma época

class Gerador(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(Gerador, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden

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

class GeneratorOutputDataset(Dataset):
    def __init__(self, generator, noise_dim, num_samples, noise_samples, text_len, min_text_len):
        self.generator = generator
        self.noise_dim = noise_dim
        self.num_samples = num_samples
        self.noise_samples = noise_samples
        self.text_len = text_len
        self.min_text_len = min_text_len  # Tamanho mínimo do texto real

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = torch.zeros((self.noise_samples, self.text_len), dtype=torch.long)
        noise = torch.randint(0, self.noise_dim, (self.noise_samples, self.noise_dim))
        if args.verbose == 'on':
            print('Noise: ', noise)
        text_chunk, _ = self.generator(noise)

        # Calcula o tamanho aleatório do texto
        random_text_len = torch.randint(self.min_text_len, self.text_len + 1, (self.noise_samples,))

        if self.text_len <= self.noise_dim:
            # Se o tamanho do texto for menor ou igual a noise_dim, use apenas a parte necessária do texto gerado
            sample[:, :self.text_len] = torch.argmax(text_chunk, dim=-1)[:, :self.text_len]
        else:
            # Se o tamanho do texto for maior que noise_dim, use o código anterior para gerar o texto em pedaços
            for i in range(self.text_len // self.noise_dim):
                sample[:, i*self.noise_dim:(i+1)*self.noise_dim] = torch.argmax(text_chunk, dim=-1)
            if self.text_len % self.noise_dim != 0:
                noise = torch.randint(0, self.noise_dim, (self.noise_samples, self.noise_dim))
                text_chunk, _ = self.generator(noise)
                start_index = (self.text_len // self.noise_dim) * self.noise_dim
                sample[:, start_index:] = torch.argmax(text_chunk, dim=-1)[:, :self.text_len-start_index]

        # Ajusta o tamanho do texto para o tamanho aleatório
        sample = sample[:, :random_text_len.max()]

        return sample

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
criterio_gerador = torch.nn.NLLLoss()
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
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
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
    gerador_path = os.path.expanduser('gerador_' + tipo[1:] + '.pt')
    cnn_path = os.path.expanduser('cnn_' + tipo[1:] + '.pt')

    print('Verificando se o gerador existe para o tipo: ', tipo[1:])
    if os.path.exists(gerador_path):
        print('Carregar o gerador')
        gerador[tipo] = torch.load(gerador_path)
    else:
        print('Criar novo gerador')
        gerador[tipo] = Gerador(len(numero_para_palavra[tipo]), 256, 512, output_size)

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
    otimizador_gerador[tipo] = torch.optim.Adam(gerador[tipo].parameters(), lr=taxa_aprendizado_gerador)

# Criando o dataset para as saídas do gerador
dataset_gerador = GeneratorOutputDataset(gerador[tipo], noise_dim, num_samples, noise_samples,max_length,min_length)
loader_gerador = DataLoader(dataset_gerador, batch_size=tamanho_lote, shuffle=True)

for epoca in range(num_epocas):
    for tipo in types:
        print(f'Colocando os modelos em modo de treinamento para epoca {epoca + 1}')
        gerador[tipo].train()
        cnn[tipo].train()
        if args.verbose == 'on':
            print('Inicializando as perdas e as acurácias')
        perda_cnn, perda_gerador = 0, 0
        acuracia_cnn, acuracia_gerador = 0, 0
        if args.verbose == 'on':
            print('Percorrendo cada lote de dados')
        for (textos, rotulos), textos_falsos in zip(train_loaders[tipo], loader_gerador):
            if args.verbose == 'cnn':
                print('Obtendo os textos e os rótulos do lote / amostra')
                print(f'Texto treinamento: {textos} e o rótulo: {rotulos}')
            #rotulos = rotulos.view(-1,1)
            if args.verbose == 'on':
                print('Zerando a acurácia para a amostra')
            acuracia_cnn, acuracia_gerador = 0, 0
            if args.verbose == 'on':
                print('Calculando a perda do cnn, usando os textos reais e os textos falsos')
            if args.verbose == 'on':
                print(textos_falsos)
            #Obtendo o índice da palavra com a maior probabilidade
            if args.verbose == 'on':
                 print('Saida Gerador: ',textos_falsos.shape)
                 for amostra in textos_falsos:
                        for ruido in amostra:
                            falso = decoder(ruido.tolist(),tipo,numero_para_palavra)
                            print('Texto falso gerado: ', falso) 
            textos_falsos = textos_falsos.view(textos_falsos.size(0), -1)
            if args.verbose == 'on':
                print('Entrada do cnn:  ', textos_falsos.shape)
            # Verifica se o tamanho dos textos falsos é menor que max_length
            if len(textos_falsos) < max_length:
            # Preenche os textos falsos com zeros à direita para atingir o tamanho máximo
                textos_falsos = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t), dtype=torch.int64))) for t in textos_falsos], batch_first=True)
            # Passando o texto falso para o cnn
            saida_real = cnn[tipo](textos)
            saida_falso = cnn[tipo](textos_falsos)
            if args.verbose == 'cnn' or args.verbose == 'on':
                print(f'Saida da cnn com textos de treinamento {saida_real} e gerados {saida_falso}')
            rotulos_float = rotulos.float()
            rotulos_reshaped = rotulos_float.view(-1, 1).repeat(1, 2)
            perda_real = criterio_cnn(saida_real, rotulos_reshaped)
            perda_falso = criterio_cnn(saida_falso, torch.zeros_like(rotulos_reshaped))
            perda_cnn = (perda_real + perda_falso) / 2
            if args.verbose == 'on':
                print('Atualizando os parâmetros do cnn')
            otimizador_cnn[tipo].zero_grad()
            perda_cnn.backward()
            otimizador_cnn[tipo].step()
            saida_falso = cnn[tipo](textos_falsos)
            if args.verbose == 'on' or args.verbose == 'cnn':
                print(f'Saida da cnn com textos falsos apos otimização: {saida_falso}')
            if args.verbose == 'on':
                print('Calculando a perda do gerador, usando os textos falsos e os rótulos invertidos')
            rotulos_reshaped = torch.ones(saida_falso.size(0), dtype=torch.long)
            rotulos_reshaped.view(-1)
            saida_falso = torch.log_softmax(saida_falso, dim=-1)
            perda_gerador = criterio_gerador(saida_falso,rotulos_reshaped)
            if args.verbose == 'on':
                print('Atualizando os parâmetros do gerador')
            otimizador_gerador[tipo].zero_grad()
            perda_gerador.backward()
            otimizador_gerador[tipo].step()
            if args.verbose == 'on':
                print('Calculando a acurácia do cnn e do gerador')
            acuracia_cnn += ((saida_real > 0.5) == rotulos).float().mean()
            acuracia_cnn += ((saida_falso < 0.5) == torch.zeros_like(rotulos)).float().mean()
            acuracia_gerador += ((saida_falso > 0.5) == torch.ones_like(rotulos)).float().mean()
            # Imprimindo as perdas e as acurácias
            print(f'Tipo {tipo}, Epoca {epoca + 1} de {num_epocas}, Perda cnn {perda_cnn.item():.4f}, Perda Gerador {perda_gerador.item():.4f}, Acuracia cnn {acuracia_cnn.item() / 2:.4f}, Acuracia Gerador {acuracia_gerador.item():.4f}')
            tentativa = 0
            saida_cnn = saida_falso
            while acuracia_gerador == 0 and tentativa < 2:
              tentativa=tentativa+1
              print(f'Tentativa {tentativa} de repassar o texto pelo gerador')
              if tentativa == 1:
                  texto_falso = torch.argmax(textos_falsos, dim=-1).unsqueeze(-1)
              else:
                  texto_falso = torch.argmax(texto_falso, dim=-1).unsqueeze(-1)
              if len(texto_falso) < max_length:
                 texto_falso = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t), dtype=torch.int64))) for t in texto_falso], batch_first=True)
              texto_falso ,_ = gerador[tipo](texto_falso)
              texto_falso = torch.argmax(texto_falso, dim=2)
              saida_cnn = torch.exp(saida_cnn)
              saida_cnn = cnn[tipo](texto_falso)
              print(f'Saida da Cnn para esta tentativa: {saida_cnn}')
              acuracia_gerador += ((saida_cnn[:, 0] > 0.5) == torch.ones_like(rotulos)).float().mean()
              saida_cnn = torch.log_softmax(saida_cnn, dim=-1) #revertendo para log_softmax para calcular a perda     
              perda_gerador = criterio_gerador(saida_cnn,rotulos_reshaped)
              otimizador_gerador[tipo].zero_grad()
              perda_gerador.backward()
              otimizador_gerador[tipo].step()
              print(f'Texto novo analisado pela cnn: {texto_falso}')
              print(f'Tentativa: {tentativa}, Perda cnn {perda_cnn.item():.4f}, Perda Gerador {perda_gerador.item():.4f}, Acuracia cnn {acuracia_cnn.item() / 2:.4f}, Acuracia Gerador {acuracia_gerador.item():.4f}')

            # No final de cada época, adicione as estatísticas à lista
            estatisticas['tipo'].append(tipo)
            estatisticas['epoca'].append(epoca)
            estatisticas['num_epocas'].append(num_epocas)
            estatisticas['perda_cnn'].append(perda_cnn.item())
            estatisticas['perda_gerador'].append(perda_gerador.item())
            estatisticas['acuracia_cnn'].append(acuracia_cnn.item() / 2)
            estatisticas['acuracia_gerador'].append(acuracia_gerador.item())
            # Save stats info
            with open(stats,'w') as f:
                json.dump(estatisticas, f)
            if args.verbose == 'on' or args.verbose == 'cnn':
                print('Salvando modelos')
            torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
            torch.save(cnn[tipo], os.path.expanduser('cnn_' + tipo[1:] + '.pt'))
   
#cnn = Cnn(input_channels, cnn_output)
#print(cnn)
