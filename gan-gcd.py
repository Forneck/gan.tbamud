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

agora = datetime.datetime.now()
timestamp = agora.strftime("%Y-%m-%d_%H-%M-%S")
stats = f'session-gcd_{timestamp}.json'
pasta = os.path.expanduser('~/mud/gan/v1')
# Tipos de arquivos que você quer gerar
types = ['.mob']

# Inicialize um dicionário para armazenar as estatísticas
estatisticas = {
    'tipo': [],
    'epoca': [],
    'num_epocas': [],
    'perda_discriminador': [],
    'perda_cnn': [],
    'perda_gerador': [],
    'acuracia_discriminador': [],
    'acuracia_cnn': [],
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
parser.add_argument('--save_time', choices=['sample', 'epoch', 'session'], default='sample', help='Escolha quando salvar o modelo')
parser.add_argument('--num_epocas', type=int, default=1, help='Número de épocas para treinamento')
parser.add_argument('--num_samples', type=int, default=1, help='Número de amostras para cada época')
parser.add_argument('--noise_dim', type=limit_noise_dim, default=100, help='Dimensão do ruído para o gerador')
parser.add_argument('--noise_samples', type=int,default=1, help='Número de amostras de ruído para o gerador')
parser.add_argument('--verbose', choices=['on', 'off', 'cnn'], default='off', help='Mais informações de saída')
parser.add_argument('--max_tentativas', type=int,default=3, help='Número maximo de passagens de repasse pelo gerador')
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

if args.verbose == 'on':
    print('Definindo a arquitetura do modelo discriminador')
class Discriminador(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Discriminador, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.pooling(output.transpose(1, 2)).squeeze(2)
        output = self.classifier(output)
        output = self.sigmoid(output)
        return output, hidden

if args.verbose == 'on' or args.verbose == 'cnn':
    print('Definindo a arquitetura do modelo cnn')
class Cnn(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim,output_size):
        super(Cnn, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = torch.nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(256, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        x = embedded.permute(0, 2, 1)  # Change shape to (batch_size, input_channels, sequence_length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(2)
        x = self.fc(x)
        x = self.softmax(x)
        return x

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
taxa_aprendizado_cnn = 0.001 
taxa_aprendizado_gerador = 0.0001 #inicial 0.0001
cnn_output = 2  # Binary classification (prob real vs. prob fake)
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
criterio_gerador = torch.nn.NLLLoss()
criterio_cnn = torch.nn.MSELoss() #pode ser a BCELoss, mas a MSE penaliza as maiores diferenças entre a saida e os rótulos.


# Criando os modelos gerador,cnn e discriminador para cada tipo de texto
gerador, discriminador, cnn = {}, {}, {}
for tipo in types:
    output_size = max(max([len(t) for t in textos_reais[tipo]]), max([len(t) for t in textos_falsos[tipo]]))

    # Caminhos dos modelos
    gerador_path = os.path.expanduser('gerador_' + tipo[1:] + '.pt')
    discriminador_path = os.path.expanduser('discriminador_' + tipo[1:] + '.pt')

    print('Verificando se o gerador existe para o tipo: ', tipo[1:])
    if os.path.exists(gerador_path):
        print('Carregar o gerador')
        gerador[tipo] = torch.load(gerador_path)
    else:
        print('Criar novo gerador')
        gerador[tipo] = Gerador(len(numero_para_palavra[tipo]), 256, 512, output_size)

    print('Verificando se o discriminador existe para o tipo: ', tipo[1:])
    if os.path.exists(discriminador_path):
        print('Carregar o discriminador')
        discriminador[tipo] = torch.load(discriminador_path)
    else:
        print('Criar novo discriminador')
        discriminador[tipo] = Discriminador(len(numero_para_palavra[tipo]), 256, 512)
    
    print('Verificando se a cnn existe para o tipo: ', tipo[1:])
    if os.path.exists(cnn_path):
        print('Carregar o cnn')
        cnn[tipo] = torch.load(cnn_path)
    else:
        print('Criar novo cnn')
        cnn[tipo] = Cnn(len(numero_para_palavra[tipo]),output_size, cnn_output)


# Criando os otimizadores para cada modelo
otimizador_discriminador, otimizador_gerador, otimizador_cnn = {}, {}, {}
for tipo in types:
    otimizador_discriminador[tipo] = torch.optim.Adam(discriminador[tipo].parameters(), lr=taxa_aprendizado_discriminador)
    otimizador_cnn[tipo] = torch.optim.Adam(cnn[tipo].parameters(), lr=taxa_aprendizado_cnn)
    otimizador_gerador[tipo] = torch.optim.Adam(gerador[tipo].parameters(), lr=taxa_aprendizado_gerador)

# Criando o dataset para as saídas do gerador
dataset_gerador = GeneratorOutputDataset(gerador[tipo], noise_dim, num_samples, noise_samples,max_length,min_length)
loader_gerador = DataLoader(dataset_gerador, batch_size=tamanho_lote, shuffle=True)

print('Iniciando o treinamento')
for epoca in range(num_epocas):
    for tipo in types:
        gerador[tipo].train()
        discriminador[tipo].train()
        cnn[tipo].train()
        perda_discriminador, perda_gerador, perda_cnn = 0, 0, 0
        acuracia_discriminador, acuracia_gerador,acuracia_cnn = 0, 0, 0
        for (textos, rotulos), textos_falsos in zip(train_loaders[tipo], loader_gerador):
           if args.verbose == 'on':
              print('Obtendo os textos e os rótulos do lote / amostra')
              print(f'Texto codificado de treinamento: {textos} e o rótulo: {rotulos}')
           if args.verbose == 'cnn':
              print(f'Rotulo do texto de treinamento: {rotulos}')
           if args.verbose == 'on':
              print('Zerando a acurácia para a amostra')
           acuracia_discriminador, acuracia_gerador, acuracia_cnn = 0, 0, 0
           if args.verbose == 'on':
              print('Calculando a perda do discriminador, usando os textos reais e os textos falsos')
           if args.verbose == 'on':
              for amostra in textos_falsos:
                  for ruido in amostra:
                      falso = decoder(ruido.tolist(),tipo,numero_para_palavra)
                      print('Texto falso gerado: ', falso) 
           textos_falsos = textos_falsos.view(textos_falsos.size(0), -1)
           cnn_ok = 0
           #Passando o texto falso para o cnn
           while cnn_ok == 0:
               saida_real = cnn[tipo](textos)
               saida_falso = cnn[tipo](textos_falsos)
               if args.verbose == 'on' or args.verbose == 'cnn':
                  print(f'Saida da cnn com textos de treinamento {saida_real} e gerados {saida_falso}')
               rotulos_float = rotulos.float()
               rotulos_reshaped = rotulos_float.view(-1, 1).repeat(1, 2)
               perda_real = criterio_cnn(saida_real, rotulos_reshaped)
               perda_falso = criterio_cnn(saida_falso, torch.zeros_like(rotulos_reshaped))
               perda_cnn = (perda_real + perda_falso) / 2
               if args.verbose == 'on':
                  print('Atualizando os parâmetros do cnn')
               otimizador_cnn[tipo].zero_grad()
               perda_cnn.backward()
               otimizador_cnn[tipo].step()
               saida_falso = cnn[tipo](textos_falsos)
               if args.verbose == 'on' or args.verbose == 'cnn':
                  print(f'Saida da cnn com textos falsos apos otimização: {saida_falso}')
               if args.verbose == 'on':
                  print('Calculando a perda do gerador, usando os textos falsos e os rótulos invertidos')
               rotulos_reshaped = torch.ones(saida_falso.size(0), dtype=torch.long)
               rotulos_reshaped.view(-1)
               saida_falso = torch.log_softmax(saida_falso, dim=-1)
               perda_gerador = criterio_gerador(saida_falso,rotulos_reshaped)
               if args.verbose == 'on':
                  print('Atualizando os parâmetros do gerador')
               otimizador_gerador[tipo].zero_grad()
               perda_gerador.backward()
               otimizador_gerador[tipo].step()
               if args.verbose == 'on' or args.verbose == 'cnn':
                  print('Calculando a acurácia do cnn e do gerador')
               acuracia_cnn += ((saida_real > 0.5) == rotulos).float().mean()
               acuracia_cnn += ((saida_falso < 0.5) == torch.zeros_like(rotulos)).float().mean()
               acuracia_gerador += ((saida_cnn[:, 0] > 0.5) == torch.ones_like(rotulos)).float().mean()
               cnn_ok = acuracia_gerador
               print(f'Tipo {tipo}, Epoca {epoca + 1} de {num_epocas}, Perda cnn {perda_cnn.item():.4f}, Perda Gerador {perda_gerador.item():.4f}, Acuracia cnn {acuracia_cnn.item() / 2:.4f}, Acuracia Gerador {acuracia_gerador.item():.4f}')
               if args.verbose == 'on' and cnn_ok == 0:
                  print('Texto passou pela Cnn e não obteve o ok')
               tentativa = 0
               while acuracia_gerador == 0 and tentativa < max_tentativas:
                  tentativa=tentativa+1
                  print(f'Tentativa {tentativa} de repassar o texto pelo gerador')
                  if tentativa == 1:
                     texto_falso = torch.argmax(textos_falsos, dim=-1).unsqueeze(-1)
                  else:
                     texto_falso = torch.argmax(texto_falso, dim=-1).unsqueeze(-1)
                  if len(texto_falso) < max_length:
                     texto_falso = pad_sequence([torch.cat((t, torch.zeros(max_length - len(t), dtype=torch.int64))) for t in texto_falso], batch_first=True)
                  texto_falso ,_ = gerador[tipo](texto_falso)
                  texto_falso = torch.argmax(texto_falso, dim=2)
                  saida_cnn = torch.exp(saida_cnn)
                  saida_cnn = cnn[tipo](texto_falso)
                  if args.verbose == 'on' or args.verbose == 'cnn':
                     print(f'Saida da Cnn para esta tentativa: {saida_cnn}')
                  acuracia_gerador += ((saida_cnn[:, 0] > 0.5) == torch.ones_like(rotulos)).float().mean()
                  saida_cnn = torch.log_softmax(saida_cnn, dim=-1) #revertendo para log_softmax para calcular a perda     
                  perda_gerador = criterio_gerador(saida_cnn,rotulos_reshaped)
                  otimizador_gerador[tipo].zero_grad()
                  perda_gerador.backward()
                  otimizador_gerador[tipo].step()
                  cnn_ok = acuracia_gerador
                  print(f'Texto novo analisado pela cnn: {texto_falso}')
                  print(f'Tentativa: {tentativa}, Perda cnn {perda_cnn.item():.4f}, Perda Gerador {perda_gerador.item():.4f}, Acuracia cnn {acuracia_cnn.item() / 2:.4f}, Acuracia Gerador {acuracia_gerador.item():.4f}')
               if cnn_ok == 0:
                   print('Cnn não aprovou o Gerador. Tentando novamente.')
           #Cnn deu ok, continuando do while.
           if cnn_ok > 0:
               print('Cnn deu ok. Passando pelo discriminador')
               textos_falsos = texto_falso
               #Passando o texto falso para o discriminador
               saida_real, _ = discriminador[tipo](textos)
               saida_falso, _ = discriminador[tipo](textos_falsos)
               rotulos_float = rotulos.float()
               rotulos_reshaped = rotulos_float.view(-1, 1).repeat(1, 2)
               perda_real = criterio_discriminador(saida_real, rotulos_reshaped)
               perda_falso = criterio_discriminador(saida_falso, torch.zeros_like(rotulos_reshaped))
               perda_discriminador = (perda_real + perda_falso) / 2
               if args.verbose == 'on':
                  print('Atualizando os parâmetros do discriminador')
               otimizador_discriminador[tipo].zero_grad()
               perda_discriminador.backward()
               otimizador_discriminador[tipo].step()
               saida_falso, _ = discriminador[tipo](textos_falsos)
               if args.verbose == 'on':
                  print('Calculando a perda do gerador, usando os textos falsos e os rótulos invertidos')
               rotulos_reshaped = torch.ones(saida_falso.size(0), dtype=torch.long)
               rotulos_reshaped.view(-1)
               if args.verbose == 'on':
                  print(f'Saida do discriminador: {saida_falso}')
               saida_falso = torch.log_softmax(saida_falso, dim=-1)
               perda_gerador = criterio_gerador(saida_falso,rotulos_reshaped)
               if args.verbose == 'on':
                  print('Atualizando os parâmetros do gerador')
               otimizador_gerador[tipo].zero_grad()
               perda_gerador.backward()
               otimizador_gerador[tipo].step()
               if args.verbose == 'on':
                  print('Calculando a acurácia do discriminador e do gerador')
               acuracia_discriminador += ((saida_real > 0.5) == rotulos).float().mean()
               acuracia_discriminador += ((saida_falso < 0.5) == torch.zeros_like(rotulos)).float().mean()
               acuracia_gerador += ((saida_falso > 0.5) == torch.ones_like(rotulos)).float().mean()
               #Imprimindo as perdas e as acurácias
               print(f'Tipo {tipo}, Epoca {epoca + 1} de {num_epocas}, Perda Discriminador {perda_discriminador.item():.4f}, Perda Gerador {perda_gerador.item():.4f}, Acuracia Discriminador {acuracia_discriminador.item() / 2:.4f}, Acuracia Gerador {acuracia_gerador.item():.4f}')
           else:
               print('Cnn ainda não deu o ok.')
           #No final de cada época, adicione as estatísticas à lista
           estatisticas['tipo'].append(tipo)
           estatisticas['epoca'].append(epoca)
           estatisticas['num_epocas'].append(num_epocas)
           estatisticas['perda_discriminador'].append(perda_discriminador.item())
           estatisticas['perda_gerador'].append(perda_gerador.item())
           estatisticas['acuracia_discriminador'].append(acuracia_discriminador.item() / 2)
           estatisticas['acuracia_gerador'].append(acuracia_gerador.item())
           estatisticas['perda_cnn'].append(perda_cnn.item())
           estatisticas['acuracia_cnn'].append(acuracia_cnn.item() / 2)
           if args.verbose == 'on':
               print('Save stats info')
           with open(stats,'w') as f:
               json.dump(estatisticas, f)
           if args.save_time == 'samples':
               if args.verbose == 'on' or args.verbose == 'cnn':
                  print('Salvando modelos')
               if arg.save_mode == 'local':
                  torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
                  torch.save(cnn[tipo], os.path.expanduser('cnn_' + tipo[1:] + '.pt'))
                  torch.save(discriminador[tipo], os.path.expanduser('discriminador_' + tipo[1:] + '.pt'))
               elif args.save_mode == 'nuvem':
                  gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)
                  cnn[tipo].save_pretrained('https://huggingface.co/' + 'cnn' + tipo[1:], use_auth_token=token)
                  discriminador[tipo].save_pretrained('https://huggingface.co/' + 'discriminador_' + tipo[1:], use_auth_token=token)
        #Fim da epoca para o tipo atual
        if args.save_time == 'epoch':
           if args.verbose == 'on':
               print('Salvando modelos')
           if args.save_mode == 'local':
               torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
               torch.save(discriminador[tipo], os.path.expanduser('discriminador_' + tipo[1:] + '.pt'))
               torch.save(cnn[tipo], os.path.expanduser('cnn_' + tipo[1:] + '.pt'))
           elif args.save_mode == 'nuvem':
               gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)
               cnn[tipo].save_pretrained('https://huggingface.co/' + 'cnn_' + tipo[1:], use_auth_token=token)
               discriminador[tipo].save_pretrained('https://huggingface.co/' + 'discriminador_' + tipo[1:], use_auth_token=token)
    #Fim dos tipo para treinamento. Incluir validação:

#Fim da sessão. Incluir teste:
if args.save_time == 'session':
    if args.verbose == 'on':
        print('Salvando modelos')
    if args.save_mode == 'local':
        torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
        torch.save(cnn[tipo], os.path.expanduser('cnn_' + tipo[1:] + '.pt'))
        torch.save(discriminador[tipo], os.path.expanduser('discriminador_' + tipo[1:] + '.pt'))
    elif args.save_mode == 'nuvem':
        gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)
        cnn[tipo].save_pretrained('https://huggingface.co' + 'cnn_' + tipo[1:], use_auth_token=token)
        discriminador[tipo].save_pretrained('https://huggingface.co/' + 'discriminador_' + tipo[1:], use_auth_token=token)
