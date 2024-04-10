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

print('Iniciando treinamento de qualidade do Gerador')
agora = datetime.datetime.now()
timestamp = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Inicio da sessão: {timestamp}')
stats = f'session-quality-{timestamp}.json'
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
parser.add_argument('--num_epocas', type=int, default=10, help='Número de épocas para treinamento')
parser.add_argument('--noise_dim', type=limit_noise_dim, default=100, help='Dimensão do ruído para o gerador')
parser.add_argument('--verbose', choices=['on', 'off'], default='on', help='Mais informações de saída')
parser.add_argument('--modo', choices=['auto','manual', 'real'],default='real', help='Modo do Prompt: Automatico, Manual ou Real')
args = parser.parse_args()

verbose = args.verbose

if verbose == 'on':
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

if verbose == 'on':
    print('Definindo o Encoder')
def encoder(texto, tipo, palavra_para_numero):
    return [palavra_para_numero[tipo].get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
    #return [palavra_para_numero[tipo].get(palavra, 0) for palavra in palavras]

def unpad(texto_entrada):
       unpad = []
       for texto in texto_entrada:
              texto = texto.tolist()
              while texto[-1] == 0:
                    texto.pop()
              unpad.append(texto)
              texto_saida = torch.tensor(unpad)
       return texto_saida

if verbose == 'on':
    print('Definindo o Decoder')
def decoder(texto_codificado, tipo, numero_para_palavra):
      return ' '.join([numero_para_palavra[tipo].get(numero, '<UNK>') for numero in texto_codificado])

if verbose == 'on':
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

if verbose == 'on':
    print('Definindo os parâmetros de treinamento')
num_epocas = args.num_epocas
taxa_aprendizado_gerador = 0.1 #inicial 0.0001
noise_dim = args.noise_dim # entre 1 e 100
noise_samples = 1
modo = args.modo
textos_falsos = {}

palavra_para_numero, numero_para_palavra,textos_reais = carregar_vocabulario(pasta, types)

for tipo in types:
    if args.verbose == 'on':
        print("Formato dos textos reais para o tipo {tipo}: ",textos_reais[tipo].shape)
    max_length = max([len(t) for t in textos_reais[tipo]])
    rotulos = [1]*len(textos_reais[tipo])

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

if verbose == 'on':
    print('Definindo o objetivo de aprendizado')
criterio_gerador = torch.nn.MSELoss()

# Criando os modelos gerador,cnn e discriminador para cada tipo de texto
gerador = {}
for tipo in types:
    output_size = 50

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
otimizador_gerador = {}
for tipo in types:
    otimizador_gerador[tipo] = torch.optim.Adam(gerador[tipo].parameters(), lr=taxa_aprendizado_gerador)

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
        chunk_saida, _ = gerador(chunk_entrada.long())

        # Calcula o argmax
        argmax = torch.argmax(chunk_saida, dim=-1)

        # Anexa o chunk de saída ao texto de saída
        if text_len <= noise_dim:
            # Se o tamanho do texto for menor ou igual a noise_dim, use apenas a parte necessária do texto gerado
            texto_saida[:, :text_len] = argmax[:, :text_len]
        else:
            # Se o tamanho do texto for maior que noise_dim, use o código anterior para gerar o texto em pedaços
            for i in range(text_len // noise_dim):
                texto_saida[:, i*noise_dim:(i+1)*noise_dim] = argmax
            if text_len % noise_dim != 0:
                start_index = (text_len // noise_dim) * noise_dim
                texto_saida[:, start_index:] = argmax[:, :text_len-start_index]

    # Calcula o tamanho aleatório do texto
    random_text_len = torch.randint(min_len, text_len + 1, (texto_entrada.size(0),))

    # Ajusta o tamanho do texto para o tamanho aleatório
    texto_saida = texto_saida[:, :random_text_len.max()]

    return texto_saida

print('Iniciando o treinamento')
for tipo in types:
        print(f'Tipo {tipo} - Treinamento')
        perda_gerador  = 0
        pontuacao = 0
        epoca = 1
        while epoca <= num_epocas:
           gerador[tipo].train()
           # Gere um novo texto
           if verbose == 'on':
               print(f'Gerando um novo texto no modo: {modo}')
           if modo == 'manual':
              prompt = input(f'> ')
              if len(prompt.strip())==0:
                  prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,noise_dim))
              else:
                  prompt = encoder(prompt,tipo,palavra_para_numero)
                  prompt = torch.tensor(prompt).unsqueeze(0)
           elif modo == 'real':
                  rand = torch.randint(0,len(textos_reais[tipo]), (1,))
                  prompt = textos_reais[tipo][rand]
                  decoded = decoder(prompt[0].tolist(),tipo,numero_para_palavra)
                  if verbose == 'on':
                     print(f'Prompt: {decoded}')
           else:
               prompt = torch.randint(0,len(numero_para_palavra[tipo]),(1,noise_dim))

           if modo != 'real' or prompt.size(-1)<noise_dim:  
               prompt = pad_sequence([torch.cat((t, torch.zeros(noise_dim - len(t)))) for t in prompt], batch_first=True)
           textos_falsos = generate_text(gerador[tipo],prompt,len(prompt),max_length,max_length)
           
           texto_int = textos_falsos.to(torch.int64)
           output = decoder(texto_int[0].tolist(),tipo,numero_para_palavra)
           if verbose == 'on':
              print(f'Saida Gerador: {output}')
           vocab = len(numero_para_palavra[tipo])
           fake = textos_falsos/vocab
           pontuacao = prompt/vocab
           if verbose == 'on':
              print(f'{fake.shape} e {pontuacao.shape}')
           fake.requires_grad_()
           perda_gerador = criterio_gerador(fake,pontuacao)

           # Atualize os parâmetros do gerador
           otimizador_gerador[tipo].zero_grad()
           perda_gerador.backward()
           otimizador_gerador[tipo].step()
           print(f'Tipo {tipo} - Epoca: {epoca}, Perda Gerador {perda_gerador.item()}')
           estatisticas['tipo'].append(tipo)
           estatisticas['perda_gerador'].append(perda_gerador.item())
           epoca = epoca + 1
           #Save stats info
           with open(stats,'w') as f:
                json.dump(estatisticas, f)

        #Fim da epoca para o tipo atual
        if args.save_time == 'epoch':
           if verbose == 'on':
               print('Salvando modelos')
           if args.save_mode == 'local':
               torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
           elif args.save_mode == 'nuvem':
               gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)
    #Fim dos tipo para treinamento

#Fim da sessão. Incluir teste:
if args.save_time == 'session':
    if verbose == 'on':
        print('Salvando modelos')
    if args.save_mode == 'local':
        torch.save(gerador[tipo], os.path.expanduser('gerador_' + tipo[1:] + '.pt'))
    elif args.save_mode == 'nuvem':
        gerador[tipo].save_pretrained('https://huggingface.co/' + 'gerador_' + tipo[1:], use_auth_token=token)

agora = datetime.datetime.now()
fim = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Início da sessão de treinamento do gerador em {timestamp} com fim em {fim}')
