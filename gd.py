import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import nltk
import os
import json

print('Definindo os parâmetros')
gerador_path = 'gerador_mob.pt'
discriminador_path = 'discriminador_mob.pt'
noise_dim = 100
noise_samples = 1
num_samples = 1
tamanho_lote = 1
tipo = '.mob'

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
        if args.verbose == 'on':
            print('Noise: ', noise)
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

def encoder(palavras, tipo, palavra_para_numero):
   # return [palavra_para_numero[tipo].get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
    return [palavra_para_numero[tipo].get(palavra, 0) for palavra in palavras]

print('Definindo o Decoder')
def decoder(texto_codificado, tipo, numero_para_palavra):
       # Decodificar o texto usando o dicionário numero_para_palavra do tipo de arquivo correspondente
      return ' '.join([numero_para_palavra[tipo].get(numero, '<UNK>') for numero in texto_codificado])

palavra_para_numero, numero_para_palavra,textos_reais = carregar_vocabulario(pasta, types)

# Carregando o modelo gerador
gerador = torch.load(gerador_path)
gerador.eval()
discriminador = torch.load(discriminador_path)
discriminador.eval()
textos_reais = torch.load('mob.pt')
textos_falsos = torch.load('fake.pt') 

# Padronizando o tamanho dos textos reais e falsos
max_length = max(max([len(t) for t in textos_reais]), max([len(t) for t in textos_falsos]))
textos_reais_pad = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t)))) for t in textos_reais], batch_first=True)
textos_falsos_pad = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t)))) for t in textos_falsos], batch_first=True)

# Combinando os textos reais e os textos falsos
textos = torch.cat((textos_reais_pad, textos_falsos_pad), dim=0)

# Atribuir rótulos binários para cada texto
rotulos = [1]*len(textos_reais) + [0]*len(textos_falsos)

# Tokenizar e codificar os textos
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
train_loaders = DataLoader(train_dataset, batch_size=tamanho_lote, shuffle=True)
valid_loaders = DataLoader(valid_dataset, batch_size=tamanho_lote)
test_loaders  = DataLoader(test_dataset, batch_size=tamanho_lote)

# Criando o dataset para as saídas do gerador
dataset_gerador = GeneratorOutputDataset(gerador, noise_dim, num_samples, noise_samples,max_length)
loader_gerador = DataLoader(dataset_gerador, batch_size=tamanho_lote, shuffle=True)

acuracia_discriminador, acuracia_gerador = 0, 0
i = 1
print('Gerando textos falsos')
with torch.no_grad():
    while acuracia_gerador == 0:
            for (textos, rotulos), textos_falsos in zip(train_loaders, loader_gerador):
                print(f'Tentativa {i}')
                rotulos = rotulos.view(-1,1)
                acuracia_discriminador, acuracia_gerador = 0, 0
                # Obtendo o índice da palavra com a maior probabilidade
                textos_falsos= torch.argmax(textos_falsos,dim=-1)
                textos_falsos = textos_falsos.view(textos_falsos.size(0), -1)
                # Passando o texto falso para o discriminador
                saida_real, _ = discriminador(textos)
                saida_falso, _ = discriminador(textos_falsos)
                rotulos = rotulos.float()
                acuracia_discriminador += ((saida_real > 0.5) == rotulos).float().mean()
                acuracia_discriminador += ((saida_falso < 0.5) == torch.zeros_like(rotulos)).float().mean()
                acuracia_gerador += ((saida_falso > 0.5) == torch.ones_like(rotulos)).float().mean()
                print(f'Discriminador: {acuracia_discriminador} e Gerador: {acuracia_gerador}')
                i = i + 1

    textos_falsos_lista = textos_falsos.tolist()
    print(textos_falsos)
    print('Formato da lista: ',textos_falsos_lista)
    print('Saida gerador: ', decoder(textos_falsos_lista,tipo,numero_para_palavra))
