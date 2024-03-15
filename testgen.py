import torch
from torch.utils.data import Dataset, DataLoader
import nltk
import os
import json


print('Definindo a arquitetura do modelo gerador')
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
        noise = torch.randint(0, self.noise_dim, (self.noise_samples , self.noise_dim))
        #print('Noise:' ,noise)
        sample, _ = self.generator(noise)
        return sample

types= ['.mob']
pasta = os.path.expanduser('~/mud/gan/v1')

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

print('Definindo o Decoder')
def decoder(texto_codificado, tipo, numero_para_palavra):
       # Decodificar o texto usando o dicionário numero_para_palavra do tipo de arquivo correspondente
      return ' '.join([numero_para_palavra[tipo].get(numero, '<UNK>') for numero in texto_codificado])

palavra_para_numero, numero_para_palavra,textos_reais = carregar_vocabulario(pasta, types)

def gerar_texto_falso(gerador_path, noise_dim, num_samples,noise_samples, tipo):
    # Carregando o modelo gerador
    gerador = torch.load(gerador_path)
    gerador.eval()

    # Criando o dataset para as saídas do gerador
    dataset_gerador = GeneratorOutputDataset(gerador, noise_dim, num_samples, noise_samples)
    loader_gerador = DataLoader(dataset_gerador, batch_size=1, shuffle=True)

    # Gerando textos falsos
    with torch.no_grad():
        for batch in loader_gerador:
            texto_falso = batch
            #print('Formato de texto_falso:',texto_falso.shape)
            # Obtendo o índice da palavra com a maior probabilidade
            texto_falso_max = torch.argmax(texto_falso, dim=-1)
            texto_falso_lista = texto_falso_max.tolist()
            #print('Formato depois do argmax:', texto_falso_max.shape)
            #print(texto_falso_max)
            #print('Formato da lista',texto_falso_lista)
            print('Saida gerador: ', decoder(texto_falso_lista[0][0],tipo,numero_para_palavra))

# Definindo os parâmetros
gerador_path = 'gerador_mob.pt'
noise_dim = 100
noise_samples = 1
num_samples = 10
tipo = '.mob'

# Gerando o texto falso
gerar_texto_falso(gerador_path, noise_dim, num_samples, noise_samples, tipo)
