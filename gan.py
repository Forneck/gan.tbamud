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

agora = datetime.datetime.now()
timestamp = agora.strftime("%Y-%m-%d_%H-%M-%S")
stats = f'session-stats_{timestamp}.json'

# Inicialize um dicionário para armazenar as estatísticas
estatisticas = {
    'tipo': [],
    'epoca': [],
    'num_epocas': [],
    'perda_discriminador': [],
    'perda_gerador': [],
    'acuracia_discriminador': [],
    'acuracia_gerador': []
}

token = 'HF-AUTH-TOKEN'

def limit_noise_dim(value):
    ivalue = int(value)
    if ivalue > 100:
        ivalue = 100
    if ivalue < 1:
        ivalue = 1
    return ivalue

# Definindo o argumento para escolher entre salvar localmente ou na nuvem
parser = argparse.ArgumentParser()
parser.add_argument('--save_mode', choices=['local', 'nuvem'], default='local', help='Escolha onde salvar o modelo')
parser.add_argument('--num_epocas', type=int, default=1, help='Número de épocas para treinamento')
parser.add_argument('--tamanho_lote', type=int, default=1, help='Tamanho do lote para treinamento')
parser.add_argument('--num_samples', type=int, default=1, help='Número de amostras para cada época')
parser.add_argument('--noise_dim', type=limit_noise_dim, default=50, help='Dimensão do ruído para o gerador')
parser.add_argument('--noise_samples', type=int,default=1, help='Número de amostras de ruído para o gerador') 
args = parser.parse_args()

print('Definindo a arquitetura do modelo gerador')
class Gerador(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(Gerador, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.linear(output)
        return output, hidden

print('Definindo a arquitetura do modelo discriminador')
class Discriminador(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Discriminador, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.pooling(output.transpose(1, 2)).squeeze(2)
        output = self.classifier(output)
        output = self.sigmoid(output)
        return output, hidden

class TextDataset(Dataset):
    def __init__(self, textos, rotulos):
        self.textos = textos
        self.rotulos = rotulos

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        return self.textos[idx], self.rotulos[idx]

class GeneratorOutputDataset(Dataset):
    def __init__(self, generator, noise_dim, num_samples,noise_samples):
        self.generator = generator
        self.noise_dim = noise_dim
        self.num_samples = num_samples
        self.noise_samples = noise_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.randint(0, self.noise_dim, (self.noise_samples, self.noise_dim))
        sample, _ = self.generator(noise)
        return sample

# Tipos de arquivos que você quer gerar
types = ['.mob']

print('Definindo o Encoder')
def encoder(texto, tipo):
    # Codificar o texto usando o dicionário palavra_para_numero do tipo de arquivo correspondente
    return [palavra_para_numero[tipo].get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar

print('Definindo o Decoder')
def decoder(texto_codificado, tipo):
    # Decodificar o texto usando o dicionário numero_para_palavra do tipo de arquivo correspondente
    return ' '.join([numero_para_palavra[tipo].get(numero, '<UNK>') for numero in texto_codificado])  # usando o nltk para juntar as palavras

# Mapeando cada palavra para um número único e número para palavra correspondente para cada tipo de arquivo
palavra_para_numero = {}
numero_para_palavra = {}
# Carregando os arquivos .pt que estão dentro do Colab
textos_reais = {}
for tipo in types:
  textos_reais[tipo] = []
  print(f'Carregando os arquivos {tipo[1:]}.pt')
  textos_reais[tipo] = torch.load(os.path.expanduser(tipo[1:] + '.pt'))

print('Construindo o vocabulário para cada tipo de arquivo')
vocabs = {}
for tipo in types:
    # Criar um conjunto vazio para armazenar as palavras do tipo de arquivo atual
    vocab = set()
    for texto in textos_reais[tipo]:
        for palavra in nltk.word_tokenize(str(texto)):  # usando o nltk para tokenizar
            vocab.add(palavra)
    # Adicionar o conjunto vocab ao dicionário vocabs, usando o tipo de arquivo como chave
    vocabs[tipo] = vocab

for tipo in types:
    # Obter o vocabulário do tipo de arquivo atual
    vocab = vocabs[tipo]
    # Criar um dicionário que mapeia cada palavra para um número, usando a ordem alfabética
    palavra_para_numero[tipo] = {palavra: i for i, palavra in enumerate(sorted(vocab))}
    # Criar um dicionário que mapeia cada número para uma palavra, usando o inverso do dicionário anterior
    numero_para_palavra[tipo] = {i: palavra for palavra, i in palavra_para_numero[tipo].items()}

# Agora você pode usar args.num_epocas, args.tamanho_lote, args.noise_dim e args.num_samples

print('Definindo os parâmetros de treinamento')
num_epocas = args.num_epocas 
tamanho_lote = args.tamanho_lote 
taxa_aprendizado_discriminador = 0.001
taxa_aprendizado_gerador = 0.0001
noise_dim = args.noise_dim # entre 1 e 100
noise_samples = args.noise_samples #numero de amostras de ruído
num_samples = args.num_samples #numero de amostras dentro da mesma época

textos_falsos = {}

# Inicializar os DataLoaders para cada tipo
train_loaders, valid_loaders, test_loaders = {}, {}, {}

for tipo in types:
    textos_falsos[tipo] = []
    print('Carregar os textos reais')
    real = tipo[1:] + '.pt'
    textos_reais[tipo] = torch.load(real)
    fake = 'fake.pt'
    textos_falsos[tipo] = torch.load(fake)

    print("Formato dos textos reais:",textos_reais[tipo].shape)
    print("Formato dos textos falsos:", textos_falsos[tipo].shape)
    # Padronizando o tamanho dos textos reais e falsos
    max_length = max(max([len(t) for t in textos_reais[tipo]]), max([len(t) for t in textos_falsos[tipo]]))
    textos_reais_pad = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t)))) for t in textos_reais[tipo]], batch_first=True)
    textos_falsos_pad = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t)))) for t in textos_falsos[tipo]], batch_first=True)

    # Combinando os textos reais e os textos falsos
    textos = torch.cat((textos_reais_pad, textos_falsos_pad), dim=0)

    # Atribuir rótulos binários para cada texto
    rotulos = [1]*len(textos_reais[tipo]) + [0]*len(textos_falsos[tipo])

    # Tokenizar e codificar os textos
    #tokenized_textos = [encoder(decoder(texto,tipo), tipo) for texto in textos]
    tokenized_textos = [torch.tensor(encoder(decoder(texto,tipo), tipo)) for texto in textos]

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

# Definindo o objetivo de aprendizado
criterio_discriminador = torch.nn.BCELoss()
criterio_gerador = torch.nn.MSELoss()

# Criando os modelos gerador e discriminador para cada tipo de texto
gerador, discriminador = {}, {}
for tipo in types:
    output_size = max(max([len(t) for t in textos_reais[tipo]]), max([len(t) for t in textos_falsos[tipo]]))

    # Caminhos dos modelos
    gerador_path = os.path.expanduser('gerador_' + tipo[1:] + '.pt')
    discriminador_path = os.path.expanduser('discriminador_' + tipo[1:] + '.pt')

    print('Verificando se o gerador existe para o tipo: ', tipo[1:])
    if os.path.exists(gerador_path):
        print('Carregar o gerador')
        gerador[tipo] = torch.load(gerador_path)
    else:
        print('Criar novo gerador')
        gerador[tipo] = Gerador(len(vocabs[tipo]), 256, 512, output_size)

    print('Verificando se o discriminador existe para o tipo: ', tipo[1:])
    if os.path.exists(discriminador_path):
        print('Carregar o discriminador')
        discriminador[tipo] = torch.load(discriminador_path)
    else:
        print('Criar novo discriminador')
        discriminador[tipo] = Discriminador(len(vocabs[tipo]), 256, 512)


# Criando os otimizadores para cada modelo
otimizador_discriminador, otimizador_gerador = {}, {}
for tipo in types:
    otimizador_discriminador[tipo] = torch.optim.Adam(discriminador[tipo].parameters(), lr=taxa_aprendizado_discriminador)
    otimizador_gerador[tipo] = torch.optim.Adam(gerador[tipo].parameters(), lr=taxa_aprendizado_gerador)

# Criando o dataset para as saídas do gerador
dataset_gerador = GeneratorOutputDataset(gerador[tipo], noise_dim, num_samples, noise_samples)
loader_gerador = DataLoader(dataset_gerador, batch_size=tamanho_lote, shuffle=True)

# Treinando os modelos gerador e discriminador alternadamente
for epoca in range(num_epocas):
    for tipo in types:
        print(f'Colocando os modelos em modo de treinamento para epoca {epoca + 1}')
        gerador[tipo].train()
        discriminador[tipo].train()
        print('Inicializando as perdas e as acurácias')
        perda_discriminador, perda_gerador = 0, 0
        acuracia_discriminador, acuracia_gerador = 0, 0
        print('Percorrendo cada lote de dados')
        for (textos, rotulos), textos_falsos in zip(train_loaders[tipo], loader_gerador):
            print('Obtendo os textos e os rótulos do lote / amostra')
            rotulos = rotulos.view(-1,1)
            print('Zerando a acurácia para a amostra')
            acuracia_discriminador, acuracia_gerador = 0, 0
            print('Calculando a perda do discriminador, usando os textos reais e os textos falsos')
            textos_falsos = textos_falsos.long()
            textos_falsos = textos_falsos.view(textos_falsos.size(0), -1)
            saida_real, _ = discriminador[tipo](textos)
            saida_falso, _ = discriminador[tipo](textos_falsos)
            rotulos = rotulos.float()
            perda_real = criterio_discriminador(saida_real, rotulos)
            perda_falso = criterio_discriminador(saida_falso, torch.zeros_like(rotulos))
            perda_discriminador = (perda_real + perda_falso) / 2
            print('Atualizando os parâmetros do discriminador')
            otimizador_discriminador[tipo].zero_grad()
            perda_discriminador.backward()
            otimizador_discriminador[tipo].step()
            saida_falso, _ = discriminador[tipo](textos_falsos)
            print('Calculando a perda do gerador, usando os textos falsos e os rótulos invertidos')
            perda_gerador = criterio_discriminador(saida_falso, torch.ones_like(rotulos))
            print('Atualizando os parâmetros do gerador')
            otimizador_gerador[tipo].zero_grad()
            perda_gerador.backward()
            otimizador_gerador[tipo].step()
            print('Calculando a acurácia do discriminador e do gerador')
            acuracia_discriminador += ((saida_real > 0.5) == rotulos).float().mean()
            acuracia_discriminador += ((saida_falso < 0.5) == torch.zeros_like(rotulos)).float().mean()
            acuracia_gerador += ((saida_falso > 0.5) == torch.ones_like(rotulos)).float().mean()
            # Imprimindo as perdas e as acurácias
            print(f'Tipo {tipo}, Epoca {epoca + 1} de {num_epocas}, Perda Discriminador {perda_discriminador.item():.4f}, Perda Gerador {perda_gerador.item():.4f}, Acuracia Discriminador {acuracia_discriminador.item() / 2:.4f}, Acuracia Gerador {acuracia_gerador.item():.4f}')
            # No final de cada época, adicione as estatísticas à lista
            estatisticas['tipo'].append(tipo)
            estatisticas['epoca'].append(epoca)
            estatisticas['num_epocas'].append(num_epocas)
            estatisticas['perda_discriminador'].append(perda_discriminador.item())
            estatisticas['perda_gerador'].append(perda_gerador.item())
            estatisticas['acuracia_discriminador'].append(acuracia_discriminador.item() / 2)
            estatisticas['acuracia_gerador'].append(acuracia_gerador.item())
            # Save stats info
            with open(stats,'w') as f:
                json.dump(estatisticas, f)
            print('Salvando modelos')
            if args.save_mode == 'local':
                torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
                torch.save(discriminador[tipo], os.path.expanduser('discriminador_' + tipo[1:] + '.pt'))
            elif args.save_mode == 'nuvem':
                gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)
                discriminador[tipo].save_pretrained('https://huggingface.co/' + 'discriminador_' + tipo[1:], use_auth_token=token)
