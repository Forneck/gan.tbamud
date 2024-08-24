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
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
# Baixar as stopwords
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
print('Iniciando treinamento de qualidade do Gerador')
agora = datetime.datetime.now()
timestamp = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Inicio da sessão: {timestamp}')
#stats = f'session-quality-{timestamp}.json'
pasta = os.path.expanduser('~/gan/v1/')
# Tipos de arquivos que você quer gerar
types = ['.mob']
# Inicialize um dicionário para armazenar as estatísticas
estatisticas = {
    'tipo': [],
    'perda_gerador': [],
    }
token = 'HF-AUTH-TOKEN'

config = {
    'seq',
    'rep'
}

def limit_noise_dim(value):
    ivalue = int(value)
    if ivalue > 150:
        ivalue = 150
    if ivalue < 3:
        ivalue = 3
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
parser.add_argument('--save_time', choices=['epoch', 'session'], default='session', help='Escolha quando salvar o modelo')
parser.add_argument('--num_epocas', type=int, default=63, help='Número de épocas para treinamento')
parser.add_argument('--num_samples', type=int, default=1, help='Número de amostras para cada época')
parser.add_argument('--rep', type=int, default=1, help='Quantidade de repetições')
parser.add_argument('--verbose', choices=['on', 'off'], default='on', help='Mais informações de saída')
parser.add_argument('--debug', choices=['on', 'off'], default='off', help='Debug Mode')
parser.add_argument('--smax', choices=['on','off'], default = 'on', help='Softmax direto no gerador?')
parser.add_argument('--limiar', type=limit_treshold, default=75, help='Limiar para considerar texto verdadeiro ou falso. Valor entre 50 e 100')
parser.add_argument('--noise_dim', type=limit_noise_dim, default=100, help='Tamanho do ruido. Valor entre 3 e 150')
args = parser.parse_args()

smax = args.smax
if args.verbose == 'on':
    print('Definindo a arquitetura do modelo gerador')

class Gerador(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_size, smax='off'):
        super(Gerador, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Camada de Atenção
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_combine = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        
        # Camada Linear para saída final
        self.linear = nn.Linear(hidden_dim * 2, output_size)
        
        self.smax = smax
        if self.smax == 'on':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        
        # Cálculo da Atenção
        attention_weights = F.softmax(self.attention(output), dim=-1)
        
        # Multiplica atenção com o output para obter o vetor de contexto
        context_vector = torch.sum(attention_weights * output, dim=1)
        
        # Expande o vetor de contexto e combina com o output original
        context_vector = context_vector.unsqueeze(1).repeat(1, output.size(1), 1)
        combined = torch.cat((context_vector, output), dim=-1)
        
        # Passa pela camada para combinar a atenção
        combined = self.attention_combine(combined)
        
        # Passa pela camada linear para previsão
        output = self.linear(combined)
        
        if self.smax == 'on':
            output = self.softmax(output)
        
        return output, hidden

if args.verbose == 'on':
    print('Definindo a arquitetura do modelo discriminador')
class Discriminador(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminador, self).__init__()
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden=None):
        output, hidden = self.lstm(input, hidden)
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
def encoder(texto, tipo, palavra_para_numero):
    return [palavra_para_numero[tipo].get(palavra, 501) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
    #return [palavra_para_numero[tipo].get(palavra, 0) for palavra in palavras]

def compare_state_dicts(dict1, dict2):
    for (key1, tensor1), (key2, tensor2) in zip(dict1.items(), dict2.items()):
        if torch.equal(tensor1, tensor2):
            print(f'Os pesos para {key1} não mudaram.')
        else:
            print(f'Os pesos para {key1} mudaram.')
            if args.debug == 'on':
              print(f'Pesos iniciais para {key1}: {tensor1}')
              print(f'Pesos finais para {key2}: {tensor2}')

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
rep = args.rep
if rep > num_epocas:
    rep = num_epocas
elif rep <= 0:
    rep = 1
debug = args.debug
taxa_aprendizado_gerador = 0.1 # > 0.01 gerador da output 0
taxa_aprendizado_discriminador = 0.001
num_samples = args.num_samples #numero de amostras dentro da mesma época
limiar = args.limiar / 100
noise_dim = args.noise_dim
textos_falsos = {}
palavra_para_numero, numero_para_palavra,textos_reais = carregar_vocabulario(pasta, types)

# Inicializar os DataLoaders para cada tipo
train_loaders, valid_loaders, test_loaders = {}, {}, {}

aprendeu = 0

max_length = {}
min_length = {}
for tipo in types:
    textos_falsos[tipo] = []
    fake = 'fake_' + tipo[1:] + '.pt'
    textos_falsos[tipo] = torch.load(fake)

    print("Formato dos textos reais:",textos_reais[tipo].shape)
    print("Formato dos textos falsos:", textos_falsos[tipo].shape)
    print('Padronizando o tamanho dos textos reais e falsos')
    max_length[tipo] = max(max([len(t) for t in textos_reais[tipo]]), max([len(t) for t in textos_falsos[tipo]]))
    min_length[tipo] = 25

    if args.verbose == 'on':
        print(' Combinando os textos reais e os textos falsos')
    textos = torch.cat((textos_reais[tipo], textos_falsos[tipo]), dim=0)

    if args.verbose == 'on':
       print('Atribuir rótulos binários para cada texto')
    rotulos = [1]*len(textos_reais[tipo]) + [0]*len(textos_falsos[tipo])

    textos = textos.to(torch.int64)

    if args.verbose == 'on':
        print('Criar o conjunto de dados')
    dataset = TextDataset(textos, rotulos)

    if args.verbose == 'on':
        print('Dividindo o conjunto de dados')
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    if args.verbose == 'on':
        print('Criando os DataLoaders')
    train_loaders[tipo] = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loaders[tipo] = DataLoader(valid_dataset, batch_size=1)
    test_loaders[tipo] = DataLoader(test_dataset, batch_size=1)

if args.verbose == 'on':
    print('Definindo o objetivo de aprendizado')
criterio_gerador = torch.nn.MSELoss()
criterio_discriminador = torch.nn.BCELoss()
#criterio_discriminador = torch.nn.MSELoss()

# Criando os modelos gerador,cnn e discriminador para cada tipo de texto
gerador = {}
discriminador = {}
for tipo in types:
    # Caminhos dos modelos
    gerador_path = os.path.expanduser('gerador_' + tipo[1:] + '.pt')

    print('Verificando se o gerador existe para o tipo: ', tipo[1:])
    if os.path.exists(gerador_path):
        print('Carregar o gerador')
        gerador[tipo] = torch.load(gerador_path)
    else:
        print('Criar novo gerador')
        gerador[tipo] = Gerador(len(numero_para_palavra[tipo]),256, 512,len(numero_para_palavra[tipo]))
        #embbeding 256, hidden 512
    
    # Caminhos dos modelos
    discriminador_path = os.path.expanduser('discriminador_' + tipo[1:] + '.pt')

    print('Verificando se o discriminador existe para o tipo: ', tipo[1:])
    if os.path.exists(discriminador_path):
        print('Carregar o discriminador')
        discriminador[tipo] = torch.load(discriminador_path)
    else:
        print('Criar novo discriminador')
        discriminador[tipo] = Discriminador(512)

# Criando os otimizadores para cada modelo
otimizador_gerador = {}
scheduler_gerador = {}
otimizador_discriminador = {}
scheduler_discriminador = {}
peso_inicio = {}
peso_inicio_discriminador = {}

for tipo in types:
    otimizador_gerador[tipo] = torch.optim.Adam(gerador[tipo].parameters(), lr=taxa_aprendizado_gerador)
    scheduler_gerador[tipo] = torch.optim.lr_scheduler.ExponentialLR(otimizador_gerador[tipo], gamma=0.99)
    #scheduler_gerador[tipo] = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador_gerador[tipo], mode='min', factor=0.1, patience=10)
    otimizador_discriminador[tipo] = torch.optim.Adam(discriminador[tipo].parameters(), lr=taxa_aprendizado_discriminador)
    #scheduler_discriminador[tipo] = torch.optim.lr_scheduler.ExponentialLR(otimizador_discriminador[tipo], gamma=0.99)
    scheduler_discriminador[tipo] = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador_discriminador[tipo], mode='min', factor=0.1, patience=10)
    peso_inicio[tipo] = {key: value.clone() for key, value in gerador[tipo].state_dict().items()}
    peso_inicio_discriminador[tipo] = {key: value.clone() for key, value in discriminador[tipo].state_dict().items()}
    if debug == 'on':
        print(f'Pesos antes do treinamento:\n Gerador: {peso_inicio[tipo]} \n Discriminador: {peso_inicio_discriminador[tipo]}')

torch.autograd.set_detect_anomaly(True)
print('Iniciando o treinamento')
for tipo in types:
        if args.verbose == 'on':
            print(f'Tipo {tipo} - Treinamento')      
        epoca = 1
        while epoca <= num_epocas:
           if (epoca % 100 == 0):
               print(f'Epoca multiplo de 100: salvando') 
               torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
               torch.save(discriminador[tipo], os.path.expanduser('discriminador_' + tipo[1:] + '.pt'))
           gerador[tipo].train()
           discriminador[tipo].train()

           if args.verbose == 'on':
             print(f'\n\n\nEpoca {epoca}/{num_epocas}')
           textos_reais[tipo].requires_grad_()
           #prompt_length = torch.randint(min_length[tipo], max_length[tipo] + 1, (1,)).item()
           prompt_length = torch.randint(min_length[tipo], max_length[tipo] + 1, (1,)).item()
           prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,prompt_length))
           decoded = decoder(prompt[0].tolist(),tipo,numero_para_palavra)
           print(f"O prompt usado foi: {decoded}")
           if prompt.size(1) < max_length[tipo]:
               prompt = pad_sequence([torch.cat((t,torch.zeros(max_length[tipo] - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)

           texto_falso,_ = gerador[tipo](prompt)

           texto_falso.requires_grad_()
           texto_falso.retain_grad()
           
           #Cria camada de ajuste saida do gerador para discriminador
           ajustador_dim = torch.nn.Linear(len(numero_para_palavra[tipo]),512)
           saida_ajustada = ajustador_dim(texto_falso)

           for(real, rotulos), texto in zip(train_loaders[tipo], texto_falso):
              if args.verbose == 'on':
                  print('Zerando a acurácia para a amostra')
              acuracia_disc = 0
              acuracia_ger = 0
              perda = 0
              perda_falso = 0
              perda_gerador = 0

              embedding_layer = torch.nn.Embedding(len(numero_para_palavra[tipo]), 512)
              texto_real = embedding_layer(real)
              saida_real,_ = discriminador[tipo](texto_real)
              saida_disc_real = torch.exp(saida_real)
              print(f'Saida do discriminador para texto de treinamento: {saida_disc_real} com rotulo de treinamento: {rotulos}')
              #rotulo 0 texto de treinamento falso
              if rotulos == 0:
                  rotulos_reshaped = [[0,1]]
              #rotulo 1 texto de treinamento verdadeiro
              elif rotulos == 1:
                  rotulos_reshaped = [[1,0]]
            
              rotulos_reshaped = torch.tensor(rotulos_reshaped, dtype=torch.float32)
              print(f'Rotulo reshaped: {rotulos_reshaped}')
              
              perda_real = criterio_discriminador(saida_disc_real, rotulos_reshaped)
              perda = perda + perda_real
              print(f'Perda do Discriminador para texto de treinamento: {perda_real}')

              if args.verbose == 'on':
                  print('Atualizando os parâmetros do discriminador')
              perda_real.backward()
              otimizador_discriminador[tipo].step()
              otimizador_discriminador[tipo].zero_grad()
              if args.verbose == 'on':
                 print('Calculando a perda do discriminador para texto gerado')
              
              saida_falsa,_ = discriminador[tipo](saida_ajustada)
              saida_disc_falsa = torch.exp(saida_falsa)
              print(f'Saida do Discriminador para texto gerado: {saida_disc_falsa}')
              #Invertendo rotulos, texto falso do gerador no discriminador
              rotulos_reshaped = [[0,1]]
              rotulos_reshaped = torch.tensor(rotulos_reshaped, dtype=torch.float32)
              perda_falso = criterio_discriminador(saida_disc_falsa,rotulos_reshaped)
              perda = perda + perda_falso
              print(f'Perda do Discriminador para texto gerado: {perda_falso}')
              if args.verbose == 'on':
                  print('Atualizando os parâmetros do Discriminador para texto gerado')

              otimizador_discriminador[tipo].zero_grad()
              perda_falso.backward()
              otimizador_discriminador[tipo].step()
              scheduler_discriminador[tipo].step(perda_falso)

              print('Invertendo os rotulos e calculando a perda do gerador')
              saida_falsa,_ = discriminador[tipo](saida_ajustada.detach())
              saida_nova = torch.exp(saida_falsa)
              print(f'Saida do discriminador apos treinamento: {saida_nova}')
              rotulos_invertidos = [[1,0]]
              rotulos_invertidos = torch.tensor(rotulos_invertidos, dtype=torch.float32)
              perda_gerador = criterio_gerador(saida_nova,rotulos_invertidos)
              perda_gerador.backward()
              otimizador_gerador[tipo].step()
              otimizador_gerador[tipo].zero_grad()

           print(f'Tipo {tipo}, Epoca {epoca} de {num_epocas}, Perda Discriminador {perda / 2}, Perda Gerador {perda_gerador}')

           texto_falso_max = torch.argmax(texto_falso, dim=-1)
           texto_falso_max = texto_falso_max.to(torch.int64) 
           saida = decoder(texto_falso_max[0].tolist(),tipo,numero_para_palavra)
            
           if args.verbose == 'on':
              print(f'\nSaida do Gerador: {saida} \n')
           else:
              print(f'\n{saida} \n  ({epoca}/{num_epocas}) \n')
           
           epoca = epoca + 1
           
           #estatisticas['tipo'].append(tipo)
           #estatisticas['perda_gerador'].append(perda_gerador.item())
           #Save stats info
           #with open(stats,'w') as f:
                #json.dump(estatisticas, f)
        
           #Fim da epoca para o tipo atual 
        #Fim do tipo atual

#Fim da sessão. Incluir teste:
peso_fim = {}
peso_fim_disc = {}
for tipo in types:
        peso_fim[tipo] = {key: value.clone() for key, value in gerador[tipo].state_dict().items()}
        print(f'Comparando os pesos do gerador para o tipo {tipo}:')
        compare_state_dicts(peso_inicio[tipo], peso_fim[tipo])
        peso_fim_disc[tipo] = {key: value.clone() for key, value in discriminador[tipo].state_dict().items()}
        print(f'Comparando os pesos do discriminador para o tipo {tipo}:')
        compare_state_dicts(peso_inicio_discriminador[tipo], peso_fim_disc[tipo])

if args.save_time == 'session':
    #if args.verbose == 'on':
    print('Salvando modelos')
    if args.save_mode == 'local':
        torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
        torch.save(discriminador[tipo], os.path.expanduser('discriminador_' + tipo[1:] + '.pt'))
    elif args.save_mode == 'nuvem':
        gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)
        discriminador[tipo].save_pretrained('https://huggingface.co/' + 'discriminador_' + tipo[1:], use_auth_token=token)

agora = datetime.datetime.now()
fim = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Início da sessão de treinamento do gerador em {timestamp} com fim em {fim}')

# Função para contar o número total de parâmetros
def contar_parametros(model):
    return sum(p.numel() for p in model.parameters())

for tipo in types:
    # Exibindo o número total de parâmetros
    total_param_gen = contar_parametros(gerador[tipo])
    total_param_disc = contar_parametros(discriminador[tipo])
    print(f"O modelo gerador para o {tipo}  possui {total_param_gen} parâmetros.\nE o modelo discriminador possui {total_param_disc}")
