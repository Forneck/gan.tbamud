import torch
from torch.utils.data import Dataset, DataLoader
import nltk
import os
import json

# Definindo os parâmetros
gerador_path = 'gerador_mob.pt'
noise_dim = 100
noise_samples = 1
num_samples = 1
tipo = '.mob'
max_length = 253
types= ['.mob']
pasta = os.path.expanduser('~/mud/gan/v1')

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
    def __init__(self, generator, noise_dim, num_samples, noise_samples, text_len):
        self.generator = generator
        self.noise_dim = noise_dim
        self.num_samples = num_samples
        self.noise_samples = noise_samples
        self.text_len = text_len  # Adicione o tamanho do texto real aqui

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = torch.zeros((self.noise_samples, self.text_len), dtype=torch.long)
        noise = torch.randint(0, self.noise_dim, (self.noise_samples, self.noise_dim))
        text_chunk, _ = self.generator(noise)
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
        return sample

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

# Carregando o modelo gerador
gerador = torch.load(gerador_path)
# Criando o dataset para as saídas do gerador
dataset_gerador = GeneratorOutputDataset(gerador, noise_dim, num_samples, noise_samples,max_length)
loader_gerador = DataLoader(dataset_gerador, batch_size=1, shuffle=True)

def gerar_texto_falso(gerador, noise_dim, num_samples,noise_samples, tipo):
 
    # Gerando textos falsos
    with torch.no_grad():
        print(f'Colocando o modelo em modo de avaliação.')
        gerador.eval()
        for textos_falsos in loader_gerador:
            print('Saida Gerador: ',textos_falsos.shape)
            for amostra in textos_falsos:
                for ruido in amostra:
                    falso = decoder(ruido.tolist(),tipo,numero_para_palavra)
                    print('Texto falso gerado: ', falso)

# Gerando o texto falso
gerar_texto_falso(gerador, noise_dim, num_samples, noise_samples, tipo)
