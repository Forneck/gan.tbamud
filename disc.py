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

print('Treinamento e Validação do Discriminador')
agora = datetime.datetime.now()
timestamp = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Inicio da sessão: {timestamp}')
pasta = os.path.expanduser('~/mud/gan/v1')
# Tipos de arquivos que você quer gerar
types = ['.mob']

token = 'HF-AUTH-TOKEN'

def limit_noise_dim(value):
    ivalue = int(value)
    if ivalue > 253:
        ivalue = 253
    if ivalue < 1:
        ivalue = 1
    return ivalue

def limit_treshold(value):
    ivalue = int(value)
    if ivalue > 100:
        ivalue = 100
        print('O valor máximo é de 100')
    if ivalue < 50:
        ivalue = 50
        print('O valor mínimo é de 50')
    return ivalue

# Definindo o argumento para escolher entre salvar localmente ou na nuvem
parser = argparse.ArgumentParser()
parser.add_argument('--save_mode', choices=['local', 'nuvem'], default='local', help='Escolha onde salvar o modelo')
parser.add_argument('--save_time', choices=['epoch', 'session'], default='epoch', help='Escolha quando salvar o modelo')
parser.add_argument('--num_epocas', type=int, default=1, help='Número de épocas para treinamento')
parser.add_argument('--num_samples', type=int, default=1, help='Número de amostras para cada época')
parser.add_argument('--verbose', choices=['on', 'off', 'cnn'], default='off', help='Mais informações de saída')
parser.add_argument('--limiar', type=limit_treshold, default=51, help='Limiar para considerar texto verdadeiro ou falso. Valor entre 50 e 100 - em %')
parser.add_argument('--modo', choices=['all','train','val'], default='all', help='Modo da gan')
args = parser.parse_args()

if args.verbose == 'on':
    print('Definindo a arquitetura do modelo discriminador')
class Discriminador(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Discriminador, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.pooling(output.transpose(1, 2)).squeeze(2)
        output = self.classifier(output)
        output = self.log_softmax(output)
        return output, hidden

class TextDataset(Dataset):
    def __init__(self, textos, rotulos):
        self.textos = textos
        self.rotulos = rotulos

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        return self.textos[idx], self.rotulos[idx]

if args.verbose == 'on':
    print('Definindo o Encoder')
def encoder(palavras, tipo, palavra_para_numero):
   # return [palavra_para_numero[tipo].get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
    return [palavra_para_numero[tipo].get(palavra, 0) for palavra in palavras]

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

if args.verbose == 'on':
    print('Definindo os parâmetros de treinamento')
num_epocas = args.num_epocas 
tamanho_lote = 1 
taxa_aprendizado_discriminador = 0.0001 #era 0.001 mas aprendia muito rapido
num_samples = args.num_samples #numero de amostras dentro da mesma época
limiar = args.limiar/100
textos_falsos = {}

# Inicializar os DataLoaders para cada tipo
train_loaders, valid_loaders, test_loaders = {}, {}, {}

palavra_para_numero, numero_para_palavra,textos_reais = carregar_vocabulario(pasta, types)
vocab_size = len(numero_para_palavra)

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
        print(f'Min length for real text: {min_length}')
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
    textos = textos.to(torch.int64)
    tokenized_textos = []
    for i in range(textos.size(0)):  # Itera sobre a primeira dimensão do tensor
        texto = textos[i].tolist()  # Converte o tensor para uma lista
        texto_decodificado = decoder(texto, tipo, numero_para_palavra)
        tokenized_textos.append(torch.tensor(encoder(nltk.word_tokenize(texto_decodificado), tipo, palavra_para_numero)))
   
    if args.verbose == 'on':
        print(' Padronizando o tamanho dos textos')
    textos_pad = pad_sequence(tokenized_textos, batch_first=True)
    
    if args.verbose == 'on':
        print('Criar o conjunto de dados')
    dataset = TextDataset(textos_pad, rotulos)

    if args.verbose == 'on':
        print('Dividindo o conjunto de dados')
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    if args.verbose == 'on':
        print('Criando os DataLoaders')
    train_loaders[tipo] = DataLoader(train_dataset, batch_size=tamanho_lote, shuffle=True)
    valid_loaders[tipo] = DataLoader(valid_dataset, batch_size=tamanho_lote)
    test_loaders[tipo] = DataLoader(test_dataset, batch_size=tamanho_lote)


if args.verbose == 'on':
    print('Definindo o objetivo de aprendizado')
criterio_discriminador = torch.nn.BCELoss()

# Criando os modelos gerador,cnn e discriminador para cada tipo de texto
discriminador = {}
for tipo in types:
    output_size = max(max([len(t) for t in textos_reais[tipo]]), max([len(t) for t in textos_falsos[tipo]]))

    discriminador_path = os.path.expanduser('discriminador_' + tipo[1:] + '.pt')

    print('Verificando se o discriminador existe para o tipo: ', tipo[1:])
    if os.path.exists(discriminador_path):
        print('Carregar o discriminador')
        discriminador[tipo] = torch.load(discriminador_path)
    else:
        print('Criar novo discriminador')
        discriminador[tipo] = Discriminador(len(numero_para_palavra[tipo]), 256, 512)
    
# Criando os otimizadores para cada modelo
otimizador_discriminador = {}
for tipo in types:
    otimizador_discriminador[tipo] = torch.optim.Adam(discriminador[tipo].parameters(), lr=taxa_aprendizado_discriminador)

print('Iniciando o treinamento')
for epoca in range(num_epocas):
   if args.modo == 'all' or args.modo == 'train':
    for tipo in types:
        print(f'Epoca {epoca} - Tipo {tipo} - Treinamento')
        discriminador[tipo].train()
        perda_discriminador = 0
        acuracia_discriminador = 0
        for (textos, rotulos), amostra in zip(train_loaders[tipo], range(num_samples)):
           if args.verbose == 'on':
              print('Obtendo os textos e os rótulos do lote / amostra')
              print(f'Texto codificado de treinamento: {textos} e o rótulo: {rotulos}')
           print(f'Rotulo do texto de treinamento: {rotulos}')
           if args.verbose == 'on':
              print('Zerando a acurácia para a amostra')
           acuracia_discriminador = 0
           rotulos_float = rotulos.float()
           rotulos_reshaped = rotulos_float.view(-1, 1).repeat(1, 2)
           saida_real, _ = discriminador[tipo](textos)
           saida_real = torch.exp(saida_real)
           print(f'Saida do discriminador para texto de treinamento: {saida_real}')
           rotulos_float = rotulos.float()
           rotulos_reshaped = rotulos_float.view(-1, 1).repeat(1, 2)
           perda_real = criterio_discriminador(saida_real, rotulos_reshaped)
           perda_discriminador = perda_real
           if args.verbose == 'on':
              print('Atualizando os parâmetros do discriminador')
           otimizador_discriminador[tipo].zero_grad()
           perda_discriminador.backward()
           otimizador_discriminador[tipo].step()
           if args.verbose == 'on':
              print('Calculando a acurácia do discriminador')
           acuracia_discriminador += ((saida_real[:,0] > limiar) == torch.ones_like(rotulos)).float().mean()
           acuracia_discriminador += ((saida_real[:,1]  > limiar) == torch.zeros_like(rotulos)).float().mean()
           #Imprimindo as perdas e as acurácias
           print(f'Tipo {tipo}, Epoca {epoca + 1} de {num_epocas}, Perda Discriminador {perda_discriminador.item():.4f}, Acuracia Discriminador {acuracia_discriminador.item() / 2:.4f}')
           if args.save_time == 'samples':
               if args.verbose == 'on' or args.verbose == 'cnn':
                  print('Salvando modelos')
               if arg.save_mode == 'local':
                  torch.save(discriminador[tipo], os.path.expanduser('discriminador_' + tipo[1:] + '.pt'))
               elif args.save_mode == 'nuvem':
                  discriminador[tipo].save_pretrained('https://huggingface.co/' + 'discriminador_' + tipo[1:], use_auth_token=token)
        #Fim da epoca para o tipo atual
        if args.save_time == 'epoch':
           if args.verbose == 'on':
               print('Salvando modelos')
           if args.save_mode == 'local':
               torch.save(discriminador[tipo], os.path.expanduser('discriminador_' + tipo[1:] + '.pt'))
           elif args.save_mode == 'nuvem':
               discriminador[tipo].save_pretrained('https://huggingface.co/' + 'discriminador_' + tipo[1:], use_auth_token=token)
    #Fim dos tipo para treinamento. Incluir validação:
   if args.modo =='all' or args.modo == 'val':
    for tipo in types:
        # Fase de validação
        print(f'Epoca {epoca} - Tipo {tipo} - Validação')
        discriminador[tipo].eval()
        with torch.no_grad():
            for (textos, rotulos), amostra in zip(valid_loaders[tipo],range(num_samples)):
               perda_discriminador = 0
               acuracia_discriminador = 0 
               if args.verbose == 'on':
                  print('Obtendo os textos e os rótulos do lote / amostra')
                  print(f'Texto codificado de validação: {textos} e o rótulo: {rotulos}')
               
               print(f'Rotulo do texto de validação: {rotulos}')
               if args.verbose == 'on':
                  print('Zerando a acurácia para a amostra')
               acuracia_discriminador = 0
               saida_real, _ = discriminador[tipo](textos)
               saida_real = torch.exp(saida_real)
               print(f'Saida do discriminador para texto de validação: {saida_real}')
               rotulos_float = rotulos.float()
               rotulos_reshaped = rotulos_float.view(-1, 1).repeat(1, 2)
               perda_real = criterio_discriminador(saida_real, rotulos_reshaped)
               perda_discriminador = perda_real
               if args.verbose == 'on':
                  print('Calculando a acurácia do discriminador')
               acuracia_discriminador += ((saida_real[:,0] > limiar) == torch.ones_like(rotulos)).float().mean()
               acuracia_discriminador += ((saida_real[:,1]  > limiar) == torch.zeros_like(rotulos)).float().mean()
               print(f'Validação: Tipo {tipo}, Epoca {epoca + 1} de {num_epocas}, Perda Discriminador {perda_discriminador.item():.4f}, Acuracia Discriminador {acuracia_discriminador.item() / 2:.4f}')

#Fim da sessão. Incluir teste:
if args.save_time == 'session':
    if args.verbose == 'on':
        print('Salvando modelos')
    if args.save_mode == 'local':
        torch.save(discriminador[tipo], os.path.expanduser('discriminador_' + tipo[1:] + '.pt'))
    elif args.save_mode == 'nuvem':
        discriminador[tipo].save_pretrained('https://huggingface.co/' + 'discriminador_' + tipo[1:], use_auth_token=token)

agora = datetime.datetime.now()
fim = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Início da sessão em {timestamp} com fim em {fim}')
