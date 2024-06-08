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
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
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
pasta = os.path.expanduser('~/gan/v1')
# Tipos de arquivos que você quer gerar
types = ['.mob']
# Inicialize um dicionário para armazenar as estatísticas
estatisticas = {
    'tipo': [],
    'perda_gerador': [],
    }
token = 'HF-AUTH-TOKEN'

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
parser.add_argument('--save_time', choices=['epoch', 'session'], default='epoch', help='Escolha quando salvar o modelo')
parser.add_argument('--num_epocas', type=int, default=100, help='Número de épocas para treinamento')
parser.add_argument('--num_samples', type=int, default=1, help='Número de amostras para cada época')
parser.add_argument('--noise_dim', type=limit_noise_dim, default=100, help='Dimensão do ruído para o gerador')
parser.add_argument('--noise_samples', type=int,default=1, help='Número de amostras de ruído para o gerador')
parser.add_argument('--verbose', choices=['on', 'off'], default='on', help='Mais informações de saída')
parser.add_argument('--modo', choices=['auto','manual', 'real'],default='auto', help='Modo do Prompt: auto, manual ou real')
parser.add_argument('--debug', choices=['on', 'off'], default='off', help='Debug Mode')
parser.add_argument('--treino', choices=['abs','rel'], default='abs', help='Treino Absoluto ou Relativo')
parser.add_argument('--valor', choices=['auto','seq', 'cont'], default='seq', help='Valor é automatico ou sequencial')
args = parser.parse_args()

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
        sample = torch.zeros((self.noise_samples, self.text_len), dtype=torch.float)
        sample.requires_grad_()
        noise = torch.randint(0, self.noise_dim, (self.noise_samples, self.noise_dim))
        text_chunk, _ = self.generator(noise)  # A saída aqui já é softmax

        # Calcula o tamanho aleatório do texto
        random_text_len = torch.randint(self.min_text_len, self.text_len + 1, (self.noise_samples,))

        # Divide text_chunk em partes
        text_chunk_parts = torch.split(text_chunk, self.noise_dim, dim=1)

        # Se o tamanho do texto for menor ou igual a noise_dim, use apenas a parte necessária do texto gerado
        if self.text_len <= self.noise_dim:
            sample[:, :self.text_len] = text_chunk_parts[0]
        else:
            # Se o tamanho do texto for maior que noise_dim, use o código anterior para gerar o texto em pedaços
            for i, part in enumerate(text_chunk_parts):
                if i*self.noise_dim + part.size(1) > self.text_len:
                    break
                sample[:, i*self.noise_dim:i*self.noise_dim + part.size(1)] = part

        # Ajusta o tamanho do texto para o tamanho aleatório
        sample = sample[:, :random_text_len.max()]
        return sample


if args.verbose == 'on':
    print('Definindo o Encoder')
def encoder(texto, tipo, palavra_para_numero):
    return [palavra_para_numero[tipo].get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
    #return [palavra_para_numero[tipo].get(palavra, 0) for palavra in palavras]

def compare_state_dicts(dict1, dict2):
    for (key1, tensor1), (key2, tensor2) in zip(dict1.items(), dict2.items()):
        if torch.equal(tensor1, tensor2):
            print(f'Os pesos para {key1} não mudaram.')
        else:
            print(f'Os pesos para {key1} mudaram.')

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
noise_dim = args.noise_dim # entre 1 e 100
noise_samples = 1
debug = args.debug
modo = args.modo
taxa_aprendizado_gerador = 0.01 # > 0.01 gerador da output 0
num_samples = args.num_samples #numero de amostras dentro da mesma época
treino = args.treino
textos_falsos = {}
valor = args.valor
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
    output_size = max_length[tipo]
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
for tipo in types:
    otimizador_gerador[tipo] = torch.optim.Adam(gerador[tipo].parameters(), lr=taxa_aprendizado_gerador)
    scheduler_gerador[tipo] = torch.optim.lr_scheduler.ExponentialLR(otimizador_gerador[tipo], gamma=0.1)

# Criando o dataset para as saídas do gerador
dataset_gerador = GeneratorOutputDataset(gerador[tipo], noise_dim, 1, noise_samples,max_length[tipo],min_length)
loader_gerador = DataLoader(dataset_gerador, batch_size=1, shuffle=True)


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
           if valor == 'auto':
             if 'rand' not in globals():
                rand = torch.randint(0,len(textos_reais[tipo]),(1,))
           elif valor == 'seq':
              max_len = textos_reais[tipo].size(0) - 1

              # Carregar o valor de seq do arquivo
              try:
                 with open('seq.json', 'r') as f:
                      seq = json.load(f)
              except FileNotFoundError:
                 seq = 1

              if seq > max_len:
                seq = seq % max_len

              rand = torch.tensor([seq])
              seq = seq + 1
              # Salvar o valor de seq no arquivo
              with open('seq.json', 'w') as f:
                 json.dump(seq, f)

           else:
               rand = torch.tensor([1])
               #print(f'{rand.shape} e {rand}')
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
                  fill_size = esp_size - prompt.size(1)
                  filler = torch.randint(0,len(numero_para_palavra[tipo]),(1,fill_size))
                  prompt = prompt.squeeze(0)
                  filler = filler.squeeze(0)
                  prompt = torch.cat((prompt, filler), dim=0)
                  prompt = prompt.unsqueeze(0)
                  #prompt = pad_sequence([torch.cat((t,torch.zeros(max_length[tipo] - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
           elif modo == 'auto':
               prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,esp_size))
               #prompt = pad_sequence([torch.cat((t, torch.zeros(max_length[tipo] - len(t), dtype = torch.int64))) for t in prompt], batch_first = True)
               decoded = decoder(prompt[0].tolist(),tipo,numero_para_palavra)
               if args.verbose == 'on':
                   print(f'Prompt aleatorio: {decoded}')
           else:
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
               print(f'Prompt simplificado: {decoded}')
               prompt = torch.tensor(prompt).unsqueeze(0)
               fill_size = esp_size - prompt.size(1)
               filler = torch.randint(0,len(numero_para_palavra[tipo]),(1,fill_size))
               prompt = prompt.squeeze(0)
               filler = filler.squeeze(0)
               prompt = torch.cat((prompt, filler), dim=0)
               prompt = prompt.unsqueeze(0)
           
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
           
           print(f'Tipo {tipo}, Epoca {epoca-1}/{num_epocas} - Perda Gerador {perda_gerador}')
           #estatisticas['tipo'].append(tipo)
           #estatisticas['perda_gerador'].append(perda_gerador.item())
           #Save stats info
           #with open(stats,'w') as f:
                #json.dump(estatisticas, f)
        
           #Fim da epoca para o tipo atual
           if args.save_time == 'epoch':
             if args.verbose == 'on':
               print('Salvando modelos')
             if args.save_mode == 'local':
               torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
             elif args.save_mode == 'nuvem':
               gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)
    
        #Fim do tipo atual

#Fim da sessão. Incluir teste:
if debug == 'on':
    peso_fim = {}
    for tipo in types:
        peso_fim[tipo] = gerador[tipo].state_dict()
        print(f'Comparando os pesos para o tipo {tipo}:')
        compare_state_dicts(peso_inicio[tipo], peso_fim[tipo])

if args.save_time == 'session':
    if args.verbose == 'on':
        print('Salvando modelos')
    if args.save_mode == 'local':
        torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
    elif args.save_mode == 'nuvem':
        gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)

agora = datetime.datetime.now()
fim = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Início da sessão de treinamento do gerador em {timestamp} com fim em {fim}')
