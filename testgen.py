import torch
from torch.utils.data import Dataset, DataLoader
import nltk


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
    def __init__(self, generator, noise_dim, num_samples):
        self.generator = generator
        self.noise_dim = noise_dim
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.randint(0, self.noise_dim, (1, self.noise_dim))
        sample, _ = self.generator(noise)
        return sample

types= ['.mob']

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
  textos_reais[tipo] = torch.load(tipo[1:]+'.pt')

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

def gerar_texto_falso(gerador_path, noise_dim, num_samples, tipo):
    # Carregando o modelo gerador
    gerador = torch.load(gerador_path)

    # Criando o dataset para as saídas do gerador
    dataset_gerador = GeneratorOutputDataset(gerador, noise_dim, num_samples)
    loader_gerador = DataLoader(dataset_gerador, batch_size=1, shuffle=True)

    # Gerando um texto falso
    for texto_falso in loader_gerador:
        texto_decodificado=decoder(texto_falso, tipo)
        print(texto_decodificado)

# Definindo os parâmetros
gerador_path = 'gerador_mob.pt'
noise_dim = 50
num_samples = 1
tipo = '.mob'

# Gerando o texto falso
gerar_texto_falso(gerador_path, noise_dim, num_samples, tipo)
