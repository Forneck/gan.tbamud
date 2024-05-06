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

def solicitar_pontuacoes():
    criterios = {
        'Qualidade': 'Avalie se o texto parece real',
    }
    pontuacoes = {}
    for criterio, descricao in criterios.items():
        while True:
            print(f'{criterio}: {descricao}')
            pontuacao = input(f'Insira a pontuação para {criterio} (0-1 ou 9 para sair): ')
            try:
                pontuacao = int(pontuacao)
                if 0 <= pontuacao <= 1 or pontuacao == 9:
                    pontuacoes[criterio] = pontuacao
                    break
                else:
                    print('Valor fora do intervalo. Por favor, insira um valor entre 0 e 1.')
                    
                    if pontuacao > 1:
                        pontuacoes[criterio] = 1
                    elif pontuacao < 0:
                        pontuacoes[criterio] = 0
                    break
            except ValueError:
                print('Entrada inválida. Por favor, insira um número inteiro entre 0 e 1')
    return pontuacoes

if args.verbose == 'on':
    print('Definindo o objetivo de aprendizado')
    criterio_gerador = torch.nn.MSELoss()
    criterio_avaliador = torch.nn.MSELoss()

#Criando os modelos gerador e avaliador para cada tipo
gerador = {}
avaliador = {}
for tipo in types:
    # Caminhos dos modelos
    output_size = max_length[tipo]
    gerador_path = os.path.expanduser('gerador_' + tipo[1:] + '.pt')
    avaliador_path = os.path.expanderuser('avaliador_' + tipo[1:] + '.pt')
    
    print('Verificando se o gerador existe para o tipo: ', tipo[1:])
    if os.path.exists(gerador_path):
        print('Carregar o gerador')
        gerador[tipo] = torch.load(gerador_path)
    else:
        print('Criar novo gerador')
        gerador[tipo] = Gerador(len(numero_para_palavra[tipo]),256, 512,len(numero_para_palavra[tipo]))
        #embbeding 256, hidden 512

    print('Verificando se o avaliador existe para o tipo: ', tipo[1:])
    if os.path.exists(avaliador_path):
        print('Carregar o avaliador')
        avaliador[tipo] = torch.load(avaliador_path)
    else:
        print('Criar novo avaliador')
        avaliador[tipo] = Avaliador(len(numero_para_palavra[tipo]),256, 512,num_classes, num_numeros)

# Criando os otimizadores para cada modelo
otimizador_gerador = {}
scheduler_gerador = {}
otimizador_avaliador = {}
scheduler_avaliador = {}
for tipo in types:
    otimizador_gerador[tipo] = torch.optim.Adam(gerador[tipo].parameters(), lr=taxa_aprendizado_gerador)
    scheduler_gerador[tipo] = torch.optim.lr_scheduler.ExponentialLR(otimizador_gerador[tipo], gamma=0.1)
    otimizador_avaliador[tipo] = torch.optim.Adam(avaliador[tipo].parameters(), lr=taxa_aprendizado_gerador)
    scheduler_avaliador[tipo] = torch.optim.lr_scheduler.ExponentialLR(otimizador_avaliador[tipo], gamma=0.1)

#print(f'Pesos antes do treinamento: {gerador[tipo].state_dict()}')
peso_inicio = {}
print('Iniciando o treinamento')
for tipo in types:
        peso_inicio[tipo] = gerador[tipo].state_dict()
        print(f'Tipo {tipo} - Treinamento')
        perda_gerador  = 0
        epoca = 1
        while epoca <= num_epocas:
           gerador[tipo].train()
           print(f'Epoca {epoca}/{num_epocas}')
           textos_reais[tipo].requires_grad_()
           rand = torch.randint(0,len(textos_reais[tipo]),(1,))
           prompt = textos_reais[tipo][rand]
           lprompt = prompt.to(torch.int64)
           for texto in lprompt:
                  upad = []
                  texto = texto.tolist()
                  while texto[-1] == 0:
                      texto.pop()
                  if debug == 'on':
                     print(texto)
                  uprompt = torch.tensor(texto)
           prompt_unpad = uprompt.to(torch.int64)
           esp_size = len(uprompt)
           iprompt = uprompt.to(torch.int64)
           decoded = decoder(iprompt.tolist(), tipo, numero_para_palavra)
           if args.verbose == 'on':
              print(f'Esperado: {decoded}')
           # Gere um novo texto
           print(f'Gerando um novo texto no modo: {modo} de tamanho {esp_size}')
           if modo == 'manual':
              prompt = input(f'> ')
              if len(prompt.strip())==0:
                  prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,esp_size))
                  print('Usando prompt aleatorio')
                  #if noise_dim < esp_size:
                     #prompt = pad_sequence([torch.cat((t, torch.ones(esp_size - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
                  #prompt = pad_sequence([torch.cat((t,torch.zeros(max_length[tipo] - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
              else:
                  prompt = encoder(prompt,tipo,palavra_para_numero)
                  prompt = torch.tensor(prompt).unsqueeze(0)
                  prompt = pad_sequence([torch.cat((t, torch.ones(esp_size - len(t), dtype = torch.int64))) for t in prompt], batch_first=True)
                  #prompt = pad_sequence([torch.cat((t,torch.zeros(max_length[tipo] - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
           elif modo == 'auto':
               prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,esp_size))
               #prompt = pad_sequence([torch.cat((t, torch.zeros(max_length[tipo] - len(t), dtype = torch.int64))) for t in prompt], batch_first = True)
               decoded = decoder(prompt[0].tolist(),tipo,numero_para_palavra)
               if args.verbose == 'on':
                   print(f'Prompt aleatorio: {decoded}')
           else:
               prompt = uprompt
           
           lprompt = prompt.to(torch.int64)
           texto_falso,_ = gerador[tipo](lprompt)

           prompt_hot = F.one_hot(prompt_unpad,len(numero_para_palavra[tipo])).float()
           if treino == 'rel':
              texto_falso = torch.log(texto_falso) 
           prompt_hot.requires_grad_()
           prompt_hot.retain_grad()
           texto_falso.requires_grad_()
           texto_falso.retain_grad()
           texto_falso = texto_falso.squeeze(0)
           print(f'Forma do texto falso {texto_falso.shape} e do Prompt 1-hot: {prompt_hot.shape}')
           perda_gerador = criterio_gerador(texto_falso, prompt_hot)
           perda_gerador.backward()
           # Atualize os parâmetros do gerador
           if debug == 'on':
               print(f'Gradiente do gerador: {texto_falso.grad}\n Gradiente do esperado: {prompt_hot.grad}')
               #Imprimir os gradientes
               for name, param in gerador[tipo].named_parameters():
                   if param.requires_grad:
                      print(name, param.grad)
           
           otimizador_gerador[tipo].step()
           #scheduler_gerador[tipo].step()
           otimizador_gerador[tipo].zero_grad()
           if args.verbose == 'on':
               texto_falso_max = torch.argmax(texto_falso, dim=-1)
               texto_falso_max = texto_falso_max.to(torch.int64)
               saida = decoder(texto_falso_max.tolist(),tipo,numero_para_palavra)
               print(f'\nSaida do Gerador: {saida}')

           epoca = epoca + 1
