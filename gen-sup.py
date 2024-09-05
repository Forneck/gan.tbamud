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
UNK = 17854
print('Iniciando treinamento de qualidade do Gerador')
agora = datetime.datetime.now()
timestamp = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Inicio da sessão: {timestamp}')
#stats = f'session-quality-{timestamp}.json'
pasta = os.path.expanduser('~/gan/v1')
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
    if ivalue > 253:
        ivalue = 253
    if ivalue < 1:
        ivalue = 1
    return ivalue

# Definindo o argumento para escolher entre salvar localmente ou na nuvem
parser = argparse.ArgumentParser()
parser.add_argument('--save_mode', choices=['local', 'nuvem'], default='local', help='Escolha onde salvar o modelo')
parser.add_argument('--save_time', choices=['epoch', 'session'], default='session', help='Escolha quando salvar o modelo')
parser.add_argument('--num_epocas', type=int, default=150, help='Número de épocas para treinamento')
parser.add_argument('--num_samples', type=int, default=150, help='Número de amostras para cada época')
parser.add_argument('--rep', type=int, default=1, help='Quantidade de repetições')
parser.add_argument('--verbose', choices=['on', 'off'], default='on', help='Mais informações de saída')
parser.add_argument('--modo', choices=['auto','manual', 'curto', 'longo'],default='longo', help='Modo do Prompt: auto, manual ou real')
parser.add_argument('--debug', choices=['on', 'off'], default='off', help='Debug Mode')
parser.add_argument('--treino', choices=['abs','rel'], default='abs', help='Treino Absoluto ou Relativo')
parser.add_argument('--valor', choices=['auto','seq', 'cont'], default='seq', help='Valor é automatico ou sequencial')
parser.add_argument('--output', choices=['same','next'], default='same', help='Atual ou proximo texto')
parser.add_argument('--smax', choices=['on','off'], default = 'on', help='Softmax direto no gerador?')
args = parser.parse_args()

smax = args.smax
output = args.output
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
    print('Definindo o Encoder')
def encoder(texto, tipo, palavra_para_numero):
    return [palavra_para_numero[tipo].get(palavra, UNK) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
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

def unpad(texto_entrada):
       unpad = []
       for texto in texto_entrada:
              texto = texto.tolist()
              while texto[-1] == 0:
                    texto.pop()
              unpad.append(texto)
              texto_saida = torch.tensor(unpad)
       return texto_saida

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
noise_dim = 100  # entre 1 e 100
noise_samples = 1
debug = args.debug
modo = args.modo
taxa_aprendizado_gerador = 0.1 # > 0.01 gerador da output 0
num_samples = args.num_samples #numero de amostras dentro da mesma época
treino = args.treino
textos_falsos = {}
valor = args.valor
palavra_para_numero, numero_para_palavra,textos_reais = carregar_vocabulario(pasta, types)

aprendeu = 0

max_length = {}
min_length = {}
for tipo in types:
    #if args.verbose == 'on':
    print(f"Formato dos textos reais para o tipo {tipo}: {textos_reais[tipo].shape}")
    max_length[tipo] = max([len(t) for t in textos_reais[tipo]])
    rotulos = [1]*len(textos_reais[tipo]) 
    min_length[tipo] = 1

# Supondo que os dicionários palavra_para_numero e numero_para_palavra já estejam definidos
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
if treino == 'rel':
   criterio_gerador = torch.nn.KLDivLoss()
else:
   criterio_gerador = torch.nn.MSELoss()

# Criando os modelos gerador,cnn e discriminador para cada tipo de texto
gerador = {}
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
                                                
# Criando os otimizadores para cada modelo
otimizador_gerador = {}
scheduler_gerador = {}
peso_inicio = {}
for tipo in types:
    otimizador_gerador[tipo] = torch.optim.Adam(gerador[tipo].parameters(), lr=taxa_aprendizado_gerador)
    scheduler_gerador[tipo] = torch.optim.lr_scheduler.ExponentialLR(otimizador_gerador[tipo], gamma=0.99)
    #scheduler_gerador[tipo] = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador_gerador[tipo], mode='min', factor=0.1, patience=10)
    peso_inicio[tipo] = {key: value.clone() for key, value in gerador[tipo].state_dict().items()}
    if debug == 'on':
       print(f'Pesos antes do treinamento: {peso_inicio[tipo]}')

print('Iniciando o treinamento')
torch.autograd.set_detect_anomaly(True)
for tipo in types:
        if args.verbose == 'on':
           print(f'Tipo {tipo} - Treinamento')
        perda_gerador  = 0
        epoca = 1
        while epoca <= num_epocas:
           if (epoca % 100 == 0):
               print(f'Epoca multiplo de 100: salvando') 
               torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
           gerador[tipo].train()
           if args.verbose == 'on':
             print(f'\n\n\nEpoca {epoca}/{num_epocas}')
           textos_reais[tipo].requires_grad_()
           max_len = textos_reais[tipo].size(0) - 1
           #max_len = 38
           if valor == 'auto':
             if 'rand' not in globals() or (epoca % rep == 0):
                rand = torch.randint(0,len(textos_reais[tipo]),(1,))
           elif valor == 'seq':
              # Carregar o valor de seq do arquivo
              try:
                 with open('seq.json', 'r') as f:
                      seq = json.load(f)
              except FileNotFoundError:
                 seq = 0

              if seq > max_len:
                #seq = seq % max_len
                seq = 0
                torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
                print(f'\n\nREINICIANDO DO INICIO DOS TEXTOS!!!\n')
                rep = rep + 1
                print(f'\nAUMENTANDO AS REPETIÇÕES\n')
                #if modo == 'curto':
                #    modo = 'longo'
                #    output = 'next'
                #    print(f'Mudanndo para modo {modo}\n')
                #elif modo == 'longo':
                #    modo = 'curto'
                #    output = 'same'
                #    print(f'Mudando para modo {modo}\n')

              rand = torch.tensor([seq])
              if ((epoca) % rep == 0):
                 seq = seq + 1
                 if rep > 1:
                    print(f'\n ULTIMA REPETICAO PARA ESTE TEXTO \n')
              # Salvar o valor de seq no arquivo
              with open('seq.json', 'w') as f:
                 json.dump(seq, f)

           else:
               if 'rand' not in globals():
                  val = input(f'Valor: ')
                  val = int(val)
                  if val<0 or val > max_len:
                    val = 0 
                    #val = torch.randint(0,len(textos_reais[tipo])-2,(1,))
               if ((epoca - 1) % rep == 0):
                   val = val + 1
               rand = torch.tensor([val])
           
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
               if modo == 'longo':
                   var = prompt.to(torch.int64)
                   decoded = decoder(var[0].tolist(), tipo, numero_para_palavra)
               if  modo != 'manual':
                   if output == 'next':
                      print(f'Prompt: {decoded}')
                   else:
                      print(f'Esperado: {decoded}')
           # Gere um novo texto
           #print(f'Gerando um novo texto no modo: {modo} de tamanho {esp_size}')
           if modo == 'manual':
              prompt = input(f'> ')
              if len(prompt.strip())==0:
                  prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,esp_size))
                  print('Usando prompt aleatorio')
                  #if noise_dim < esp_size:
                     #prompt = pad_sequence([torch.cat((t, torch.ones(esp_size - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
                  #prompt = pad_sequence([torch.cat((t,torch.zeros(max_length[tipo] - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
              else:
                  #prompt = ' INICIO ' + prompt + ' FIM '
                  prompt = encoder(prompt,tipo,palavra_para_numero)
                  prompt = torch.tensor(prompt).unsqueeze(0)
                  print(f'Confirmando: {prompt}')
                  
                  #fill_size = esp_size - prompt.size(1)
                  #filler = torch.randint(0,len(numero_para_palavra[tipo]),(1,fill_size))
                  #prompt = prompt.squeeze(0)
                  #filler = filler.squeeze(0)
                  #prompt = torch.cat((prompt, filler), dim=0)
                  #prompt = prompt.unsqueeze(0)
                  prompt = pad_sequence([torch.cat((t,torch.zeros(max_length[tipo] - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
           elif modo == 'auto':
               prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,esp_size))
               #prompt = pad_sequence([torch.cat((t, torch.zeros(max_length[tipo] - len(t), dtype = torch.int64))) for t in prompt], batch_first = True)
               decoded = decoder(prompt[0].tolist(),tipo,numero_para_palavra)
               if args.verbose == 'on':
                   print(f'Prompt aleatorio: {decoded}')
           elif modo == 'curto':
               decoded = re.sub(r'\b\d+\b', '', decoded)
               words = nltk.word_tokenize(decoded)
               # Remover stopwords
               stop_words = set(stopwords.words('english'))
               words = [word for word in words if word.casefold() not in stop_words and word != '~']
               # Manter apenas substantivos e nomes próprios (NN, NNP)
               tagged = pos_tag(words)
               #words = [word for word, pos in tagged if pos in ['NN', 'NNP']]
               words = [word for word, pos in tagged if pos in ['NN', 'NNP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']] #nomes, verbos e adjetivos
               #Remover palavras repetidas
               seen = set()
               words = [word for word in words if not (word in seen or seen.add(word))]
               # Juntar as palavras de volta em uma string
               decoded = ' '.join(words)
               prompt = encoder(decoded, tipo, palavra_para_numero)
               if args.verbose == 'on':
                  print(f'Prompt simplificado: {decoded}')
               prompt = torch.tensor(prompt).unsqueeze(0)
               #fill_size = esp_size - prompt.size(1)
               #filler = torch.randint(0,len(numero_para_palavra[tipo]),(1,fill_size))
               #prompt = prompt.squeeze(0)
               #filler = filler.squeeze(0)
               #prompt = torch.cat((prompt, filler), dim=0)
               #prompt = prompt.unsqueeze(0)
               #Preencher com 0 
               #prompt = pad_sequence([torch.cat((t,torch.zeros(max_length[tipo] - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
           else:
               prompt = textos_reais[tipo][rand]
               lprompt = prompt.tolist()
               lista = []
               for texto in lprompt:
                   while texto[-1] == 0:
                       texto.pop()
                   lista.append(texto)
               prompt = torch.tensor(lista)
           lprompt = prompt.to(torch.int64)
           entrada = lprompt

           if output == 'next':
               prox = rand + 1
               if prox > max_len:
                   prox = 0
               prox = torch.tensor([prox])
               prompt_unpad = textos_reais[tipo][prox]
               prompt_unpad = prompt_unpad.to(torch.int64)
               novo = prompt_unpad.tolist()
               lista = []
               for texto in novo:
                   while texto[-1] == 0:
                       texto.pop()
                   lista.append(texto)
               #rep = len(texto) # * 10 no futuro
               new = decoder(prompt_unpad[0].tolist(),tipo,numero_para_palavra)
               prompt_unpad = torch.tensor(lista)
               if args.verbose == 'on' and modo != 'manual':
                  print(f'\nEsperado: {new}')

           if modo=='manual':
             confirm = 0
             while confirm != 1:
               esperado = input(f'Desejado >')
               #esperado = ' INICIO ' + esperado + ' FIM '
               esperado = encoder(esperado,tipo,palavra_para_numero)
               esperado = torch.tensor(esperado).unsqueeze(0)
               prompt_unpad = esperado
               #prompt_unpad = pad_sequence([torch.cat((t,torch.zeros(max_length[tipo] - len(t), dtype=torch.int64))) for t in esperado], batch_first=True)
               #print(f'Confirmando: {esperado}')
               #resposta = input('0 para errado e 1 para certo: ')
               #confirm = int(resposta)
               confirm = 1
           if output == 'same':
               prompt_unpad = prompt_unpad.unsqueeze(0)
           
           entrada = F.pad(entrada, ( 0, max_length[tipo] - entrada.size(1),0,0))
           #entrada = pad_sequence([torch.cat((t, torch.full((fill_size,), 503, dtype=torch.int64))) for t in entrada], batch_first=True)
           print(f'{entrada.shape} e {prompt_unpad.shape}')
           if entrada.size(1) < prompt_unpad.size(1):
               print('\nentrada do gerador menor que saida esperada.')
               fill_size = prompt_unpad.size(1) - entrada.size(1)
               entrada = pad_sequence([torch.cat((t, torch.full((fill_size,), UNK, dtype=torch.int64))) for t in entrada], batch_first=True)
           elif prompt_unpad.size(1) < entrada.size(1):
               print('\nsaida esperada precisa de padding para ficar igual entrada')
               fill_size = entrada.size(1) - prompt_unpad.size(1)
               prompt_unpad = pad_sequence([torch.cat((t, torch.full((fill_size,), UNK , dtype=torch.int64))) for t in prompt_unpad], batch_first=True)

           texto_falso,_ = gerador[tipo](entrada)
           print(f'Formato saida gerador {texto_falso.shape}')
           if smax == 'off':
               texto_falso = torch.softmax(texto_falso,dim=1)
               #print(f'Saida pos-max {texto_falso}')
           prompt_unpad = prompt_unpad.to(torch.int64)

           print('1-hot do esperado')
           prompt_hot = F.one_hot(prompt_unpad,len(numero_para_palavra[tipo])).float()
           #prompt_hot = prompt_hot.squeeze(0)
           
           if treino == 'rel':
              print(f'Log na saida do gerador para {texto_falso.shape}')
              saida_original = texto_falso
              texto_falso = torch.log(texto_falso)
              print(f'shape apos log: {texto_falso.shape}')
              print(f'Convertendo saida esperada em probabilidades com forma {prompt_hot.shape}')
              epsilon = 0.1
              smoothed = (1 - epsilon) * prompt_hot + (epsilon / len(numero_para_palavra[tipo]))
              print(f'Forma do esperado depois da suavização {smoothed.shape} e valor: {smoothed}')
              prompt_hot = smoothed

           prompt_hot = prompt_hot.squeeze(0)

           prompt_hot.requires_grad_()
           prompt_hot.retain_grad()
           texto_falso.requires_grad_()
           texto_falso.retain_grad()
           texto_falso = texto_falso.squeeze(0)
           if treino == 'rel':
              saida_original = saida_original.squeeze(0)
              texto_falso_max = torch.argmax(saida_original, dim= -1)
           else:
              texto_falso_max = torch.argmax(texto_falso, dim=-1)
           texto_falso_max = texto_falso_max.to(torch.int64)

           saida = decoder(texto_falso_max.tolist(),tipo,numero_para_palavra)
           if args.verbose == 'on':
              print(f'\nSaida do Gerador: {saida} \n')
           else:
              print(f'\n{saida} \n  ({epoca}/{num_epocas}) \n')

           if args.verbose == 'on':
              print(f'Forma do texto falso {texto_falso.shape} e do Prompt 1-hot: {prompt_hot.shape}')
           perda_gerador = criterio_gerador(texto_falso, prompt_hot)
           if perda_gerador == 0:
               seq = seq + 1
               aprendeu = 1

           perda_gerador.backward()
           # Atualize os parâmetros do gerador
           if debug == 'on':
               print(f'Gradiente do gerador: {texto_falso.grad}\n Gradiente do esperado: {prompt_hot.grad}')
               #Imprimir os gradientes
               for name, param in gerador[tipo].named_parameters():
                   if param.requires_grad:
                      print(name, param.grad)
           
           otimizador_gerador[tipo].step()
           
           if aprendeu == 1:
              print(f'\nReduzindo a taxa de aprendizagem em 1% \n\n')
              scheduler_gerador[tipo].step()
              aprendeu = 0

           otimizador_gerador[tipo].zero_grad()
           
           epoca = epoca + 1
           
           if args.verbose == 'on':
              print(f'Tipo {tipo}, Epoca {epoca-1}/{num_epocas} - Perda Gerador {perda_gerador} Modo - {modo} Indice {rand}')
           else:
              print(f'Perda: {perda_gerador}')
           
           #estatisticas['tipo'].append(tipo)
           #estatisticas['perda_gerador'].append(perda_gerador.item())
           #Save stats info
           #with open(stats,'w') as f:
                #json.dump(estatisticas, f)
        
           #Fim da epoca para o tipo atual 
        #Fim do tipo atual

#Fim da sessão. Incluir teste:
peso_fim = {}
for tipo in types:
        peso_fim[tipo] = {key: value.clone() for key, value in gerador[tipo].state_dict().items()}
        print(f'Comparando os pesos para o tipo {tipo}:')
        compare_state_dicts(peso_inicio[tipo], peso_fim[tipo])

if args.save_time == 'session':
    #if args.verbose == 'on':
    print('Salvando modelos')
    if args.save_mode == 'local':
        torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
    elif args.save_mode == 'nuvem':
        gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)

agora = datetime.datetime.now()
fim = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Início da sessão de treinamento do gerador em {timestamp} com fim em {fim}')

# Função para contar o número total de parâmetros
def contar_parametros(model):
    return sum(p.numel() for p in model.parameters())

for tipo in types:
    # Exibindo o número total de parâmetros
    total_parametros = contar_parametros(gerador[tipo])
    print(f"O modelo para o {tipo}  possui {total_parametros} parâmetros.")
