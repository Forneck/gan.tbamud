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

print('Iniciando treinamento exclusivo do Gerador')
agora = datetime.datetime.now()
timestamp = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Inicio da sessão: {timestamp}')
stats = f'session-gen-{timestamp}.json'
pasta = os.path.expanduser('~/mud/gan/v1')
# Tipos de arquivos que você quer gerar
types = ['.mob']
# Inicialize um dicionário para armazenar as estatísticas
estatisticas = {
    'tipo': [],
    'epoca': [],
    'num_epocas': [],
    'perda_gerador': [],
    'acuracia_gerador': []
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
parser.add_argument('--num_epocas', type=int, default=1, help='Número de épocas para treinamento')
parser.add_argument('--num_samples', type=int, default=1, help='Número de amostras para cada época')
parser.add_argument('--noise_dim', type=limit_noise_dim, default=100, help='Dimensão do ruído para o gerador')
parser.add_argument('--noise_samples', type=int,default=1, help='Número de amostras de ruído para o gerador')
parser.add_argument('--verbose', choices=['on', 'off'], default='off', help='Mais informações de saída')
parser.add_argument('--max_tentativas', type=int,default=3, help='Número maximo de passagens de repasse pelo gerador')
parser.add_argument('--modo', choices=['all','train','val'], default='all', help='Modo da gan')
parser.add_argument('--tamanho_lote', type=int,default=1, help='Tamanho do lote do loader do gerador')
args = parser.parse_args()

if args.verbose == 'on':
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
        sample = torch.zeros((self.noise_samples, self.text_len), dtype=torch.long)
        noise = torch.randint(0, self.noise_dim, (self.noise_samples, self.noise_dim))
        text_chunk, _ = self.generator(noise)

        # Calcula o tamanho aleatório do texto
        random_text_len = torch.randint(self.min_text_len, self.text_len + 1, (self.noise_samples,))

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

        # Ajusta o tamanho do texto para o tamanho aleatório
        sample = sample[:, :random_text_len.max()]

        return sample

if args.verbose == 'on':
    print('Definindo o Encoder')
def encoder(palavras, tipo, palavra_para_numero):
   # return [palavra_para_numero[tipo].get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
    return [palavra_para_numero[tipo].get(palavra, 0) for palavra in palavras]

def unpad(texto_entrada):
       unpad = []
       for texto in texto_entrada:
              texto = texto.tolist()
              while texto[-1] == 0:
                    texto.pop()
              unpad.append(texto)
              texto_saida = torch.tensor(unpad)
       return texto_saida

def comp_size(texto_a, texto_b):
      lista = []
      for texto in texto_a:
          while texto[-1] == 0:
            texto.pop()
          if len(texto) == len(texto_b):
             lista.append(texto)
      return lista

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
tamanho_lote = args.tamanho_lote
taxa_aprendizado_gerador = 0.001 #inicial 0.0001
noise_dim = args.noise_dim # entre 1 e 100
noise_samples = 1
max_tentativas = args.max_tentativas #tentativas de repasse para o gerador se não aprovado pela cnn
num_samples = args.num_samples #numero de amostras dentro da mesma época
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
    real_unpad = []
    for texto in textos_reais[tipo]:
         texto = texto.tolist()
         while texto[-1] == 0:
            texto.pop()
         real_unpad.append(texto)
 
    min_length = min(len(texto) for texto in real_unpad)
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
criterio_gerador = torch.nn.NLLLoss()

# Criando os modelos gerador,cnn e discriminador para cada tipo de texto
gerador, discriminador, cnn = {}, {}, {}
for tipo in types:
    output_size = max(max([len(t) for t in textos_reais[tipo]]), max([len(t) for t in textos_falsos[tipo]]))

    # Caminhos dos modelos
    gerador_path = os.path.expanduser('gerador_' + tipo[1:] + '.pt')

    print('Verificando se o gerador existe para o tipo: ', tipo[1:])
    if os.path.exists(gerador_path):
        print('Carregar o gerador')
        gerador[tipo] = torch.load(gerador_path)
    else:
        print('Criar novo gerador')
        gerador[tipo] = Gerador(len(numero_para_palavra[tipo]), 256, 512, output_size)

# Criando os otimizadores para cada modelo
otimizador_discriminador, otimizador_gerador, otimizador_cnn = {}, {}, {}
for tipo in types:
    otimizador_gerador[tipo] = torch.optim.Adam(gerador[tipo].parameters(), lr=taxa_aprendizado_gerador)

# Criando o dataset para as saídas do gerador
dataset_gerador = GeneratorOutputDataset(gerador[tipo], noise_dim, num_samples, noise_samples,max_length,min_length)
loader_gerador = DataLoader(dataset_gerador, batch_size=tamanho_lote, shuffle=True)

def generate_text(gerador, texto_entrada, input_len, min_len, text_len):
    # Inicializa o tensor de saída
    texto_saida = torch.zeros((texto_entrada.size(0), text_len), dtype=torch.long)

    # Divide o texto de entrada em chunks
    for i in range(0, input_len, noise_dim):
        # Obtém o próximo chunk do texto de entrada
        chunk_entrada = texto_entrada[:, i:i+noise_dim]

        # Verifica se o tamanho do chunk_entrada é menor que noise_dim
        if chunk_entrada.size(1) < noise_dim:
            # Se for, preenche o restante com zeros
            padding = torch.zeros((chunk_entrada.size(0), noise_dim - chunk_entrada.size(1)), dtype=torch.long)
            chunk_entrada = torch.cat([chunk_entrada, padding], dim=1)

        # Passa o chunk pelo gerador
        chunk_saida, _ = gerador(chunk_entrada)

        # Anexa o chunk de saída ao texto de saída
        texto_saida[:, i:i+noise_dim] = torch.argmax(chunk_saida, dim=-1)

    # Calcula o tamanho aleatório do texto
    random_text_len = torch.randint(min_len, text_len + 1, (texto_entrada.size(0),))

    # Ajusta o tamanho do texto para o tamanho aleatório
    texto_saida = texto_saida[:, :random_text_len.max()]

    return texto_saida

print('Iniciando o treinamento')
for epoca in range(num_epocas):
    for tipo in types:
        print(f'Epoca {epoca} - Tipo {tipo} - Treinamento')
        gerador[tipo].train()
        perda_gerador  = 0
        for (textos, rotulos), textos_falsos in zip(train_loaders[tipo], loader_gerador):
           print(f'Rotulo: {rotulos} e formato {textos.shape}')
           textos_falsos = textos_falsos.view(textos_falsos.size(0), -1)
           if args.verbose == 'on':
              print('Calculando a perda do gerador')
           #unpad do gerador, compara os textos reais com o tamanho do gerador e escolhe um para usar como rótulo da NLLLOSS
           print('Criando a lista de escolha')
           lista_real = []
           print(f'Formato da saida do gerador: {textos_falsos.shape}')        
           # Se rótulos for 1, adicione o conteúdo de textos em lista_real
           if rotulos == 1:
               for texto in textos:
                   while texto(-1) == 0:
                       texto.pop()
               lista_real.append(texto)
           else:
              # Se não for, procure um texto em real_unpad com a mesma largura do texto em textos_falsos e adicione em lista_real
              for fake in textos_falsos:
                   lista_real.extend(comp_size(real_unpad, fake))         
           # Se lista_real ainda assim estiver vazia ou não tiver o mesmo tamanho de textos_falsos
           novo_texto = textos_falsos
           while len(novo_texto) != len(lista_real[0]):
              print('Nenhum texto real com o mesmo tamanho da saida do gerador.')
              print('Gerando novo texto')
              prompt = torch.randint(0, len(numero_para_palavra[tipo]), (1, noise_dim))
              novo_texto = generate_text(gerador[tipo], prompt, len(prompt), len(textos), max_length)
              print(f'{novo_texto.shape}')
              lista_real.extend(comp_size(real_unpad, novo_texto))
              textos_falsos = novo_texto
           print('Texto encontrado')

           # Escolha um texto real_exemplo dessa lista
           indice_aleatorio = torch.randint(0, len(lista_real), (1,))
           real_exemplo = lista_real[indice_aleatorio.item()].float()
           real_exemplo = real_exemplo.squeeze(0)

           print(f'Formato do exemplo a ser usado como rotulos: {real_exemplo.shape} e do texto gerado: {textos_falsos.shape}')
           #log-probabilidades do gerador
           gerador_log = torch.log_softmax(textos_falsos.float(),dim=-1)
           gerador_log.requires_grad_()
           if args.verbose == 'on':
               print(f'Saida do gerador: {textos_falsos.shape}')
               print(f'Log da saida: {gerador_log.shape}')
           perda_gerador = criterio_gerador(gerador_log,real_exemplo)
           if args.verbose == 'on':
                  print('Atualizando os parâmetros do gerador')
           otimizador_gerador[tipo].zero_grad()
           perda_gerador.backward()
           otimizador_gerador[tipo].step()
           print(f'Tipo {tipo}, Epoca {epoca + 1} de {num_epocas}, Perda Gerador {perda_gerador.item():.4f}')
           estatisticas['tipo'].append(tipo)
           estatisticas['epoca'].append(epoca)
           estatisticas['num_epocas'].append(num_epocas)
           estatisticas['perda_gerador'].append(perda_gerador.item())
           # Save stats info
           with open(stats,'w') as f:
                json.dump(estatisticas, f)
           output = decoder(textos_falsos[0].tolist(),tipo,numero_para_palavra)
           if args.verbose == 'on' or perda_gerador < 5:
              print(f'Texto final: {output}')
              with open('gerador-treino.txt', 'a') as file:
                   file.write(output)

           if args.save_time == 'samples':
               if args.verbose == 'on' or args.verbose == 'cnn':
                  print('Salvando modelos')
               if arg.save_mode == 'local':
                  torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
               elif args.save_mode == 'nuvem':
                  gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)
        #Fim da epoca para o tipo atual
        if args.save_time == 'epoch':
           if args.verbose == 'on':
               print('Salvando modelos')
           if args.save_mode == 'local':
               torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
           elif args.save_mode == 'nuvem':
               gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)
    #Fim dos tipo para treinamento

#Fim da sessão. Incluir teste:
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
