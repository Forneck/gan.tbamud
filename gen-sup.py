import os
import torch
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
from nltk.sentiment import SentimentIntensityAnalyzer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
# Baixar as stopwords
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
#nltk.download('vader_lexicon')
print('Iniciando treinamento de qualidade do Gerador')
agora = datetime.datetime.now()
timestamp = agora.strftime("%H:%M:%S_%d-%m-%Y")
print(f'Inicio da sessão: {timestamp}')
#stats = f'session-quality-{timestamp}.json'
pasta = os.path.expanduser('~/gan/v1')
# Tipos de arquivos que você quer gerar
types = ['.mob']
# Inicialize um dicionário para armazenar as estatísticas
stats = 'perda.json'
try:
    with open(stats, "r") as f:
        estatisticas = json.load(f)  # Carregar os dados existentes
except FileNotFoundError:
    estatisticas = {"perda_gerador": [], "sequencia": []}  # Caso o arquivo não exista
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
parser.add_argument('--num_epocas', type=int, default=10, help='Número de épocas para treinamento')
parser.add_argument('--num_samples', type=int, default=2, help='Número de amostras para cada época')
parser.add_argument('--rep', type=int, default=1, help='Quantidade de repetições')
parser.add_argument('--verbose', choices=['on', 'off'], default='on', help='Mais informações de saída')
parser.add_argument('--modo', choices=['auto','manual', 'curto', 'longo'],default='curto', help='Modo do Prompt: auto, manual ou real')
parser.add_argument('--debug', choices=['on', 'off'], default='off', help='Debug Mode')
parser.add_argument('--treino', choices=['abs','rel'], default='rel', help='Treino Absoluto ou Relativo')
parser.add_argument('--valor', choices=['auto','seq', 'cont'], default='seq', help='Valor é automatico ou sequencial')
parser.add_argument('--output_tipo', choices=['same','next'], default='same', help='Atual ou proximo texto')
parser.add_argument('--smax', choices=['on','off'], default = 'on', help='Softmax direto no gerador?')
args = parser.parse_args()

smax = args.smax

class Gerador(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_size, smax='off'):
        super(Gerador, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=2, 
            dropout=0.1, 
            batch_first=True, 
            bidirectional=True
        )

        # Camada de Atenção
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_combine = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        # Camada Linear para saída final
        self.linear = nn.Linear(hidden_dim * 2, output_size)

        # Softmax opcional
        self.smax = smax
        if self.smax == 'on':
            self.softmax = nn.Softmax(dim=-1)

        # Camada para o autorreferenciamento (autoref)
        self.autoref = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Gera um score único (ex.: confiança)
        )
        self.hidden_proj = nn.Linear(hidden_dim, 768)  # Ajuste dimensões conforme necessário
        self.output_proj = nn.Linear(output_size, 768)

        # Inicializa o analisador de sentimentos
        self.sia = SentimentIntensityAnalyzer()
        
    def forward(self, input, hidden=None, mask=None,tokens_relevantes=None):
        embedded = self.embedding(input)

        # Inicializar hidden state caso não seja fornecido
        #if hidden is None:
        #   batch_size = input.size(0)  # Tamanho do batch
        #   num_directions = 2 if self.lstm.bidirectional else 1
        #   hidden = (
        #    torch.zeros(
        #        self.lstm.num_layers * num_directions,
        #        batch_size,
        #        self.lstm.hidden_size,
        #        device=input.device,
        #   ),
        #     torch.zeros(
        #        self.lstm.num_layers * num_directions,
        #        batch_size,
        #        self.lstm.hidden_size,
        #        device=input.device,
        #     ),
        #   )


        # Passar pela LSTM
        output, hidden = self.lstm(embedded, hidden)

        # Cálculo da atenção
        attention_weights = self.attention(output)
       
        # Aplicar a máscara, se fornecida
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(attention_weights)
            attention_weights = attention_weights * mask
        
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Vetor de contexto
        context_vector = torch.sum(attention_weights * output, dim=1)

        # Combinação com a saída original
        context_vector = context_vector.unsqueeze(1).expand(-1, output.size(1), -1)
        combined = torch.cat((context_vector, output), dim=-1)
        combined = self.attention_combine(combined)

        # Previsão final
        output = self.linear(combined)

        if self.smax == 'on':
            output = self.softmax(output)

            # Seleção automática de tokens relevantes, caso não seja fornecido
        if tokens_relevantes is None:
          attn_avg = attention_weights.mean(dim=-1)  # Média ao longo das dimensões de features
          tokens_relevantes = []
          top_k = 3  # Seleciona os 3 tokens mais relevantes
          for i in range(attn_avg.size(0)):
            sample_attn = attn_avg[i]  # (seq_len,)
            topk = torch.topk(sample_attn, k=min(top_k, sample_attn.size(0)))
            tokens_relevantes.append(topk.indices.tolist())  # Índices relevantes na sequência

        score, entropy, similarity, attention_dispersion = self.autorreferenciamento(output, hidden,tokens_relevantes)
        #score, entropy, similarity = self.autorreferenciamento(output, hidden)
        # output = self.ajustar_saida(output, score, entropy, similarity)
        return output, hidden, score, entropy, similarity, attention_dispersion

    def reset_linear_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)  # Xavier Uniform
        nn.init.zeros_(self.linear.bias)            # Bias inicializado como 0  

    def reset_attention_weights(self):
        nn.init.xavier_uniform_(self.attention.weight)  # Xavier Uniform
        nn.init.zeros_(self.attention.bias)            # Bias inicializado como 0
        nn.init.xavier_uniform_(self.attention_combine.weight)  # Xavier Uniform
        nn.init.zeros_(self.attention_combine.bias)            # Bias inicializado como 0
    
    def reset_lstm_weights(self):
      # Reinicializa os pesos de todas as camadas e direções da LSTM
      for name, param in self.lstm.named_parameters():
        if 'weight_ih' in name:  # Pesos das entradas para os nós ocultos
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:  # Pesos das conexões recursivas (estado oculto)
            nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:  # Biases
            nn.init.zeros_(param.data)

    def autorreferenciamento(self, output, hidden, tokens_relevantes=None):
      # Extraia o estado oculto (hidden) do tuple retornado pelo LSTM
      hidden, _ = hidden  # Pega apenas o estado oculto

      # Pegue a última camada do estado oculto
      hidden_last_layer = hidden[-1]  # (batch_size, hidden_dim)
      output_last_timestep = output[:, -1, :]  # (batch_size, output_dim)

      # Projeção linear para ajustar dimensões
      hidden_last_layer = self.hidden_proj(hidden_last_layer)  # Projeta para dimensão esperada
      output_last_timestep = self.output_proj(output_last_timestep)  # Projeta para dimensão esperada

      # Concatenar após projeção
      concat_input = torch.cat((output_last_timestep, hidden_last_layer), dim=1)

      # Métrica 1: Entropia da saída (confiança)
      probabilities = F.softmax(output_last_timestep, dim=-1)
      entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)  # (batch_size,)

      # Métrica 2: Similaridade semântica entre entrada e saída
      similarity = torch.cosine_similarity(output_last_timestep, hidden_last_layer, dim=1)  # (batch_size,)

      # Métrica 3: Dispersão da atenção (quanto o modelo se focou no contexto)
      attention_probabilities = F.softmax(output, dim=-1)  # (batch_size, seq_len, vocab_size)
      timestep_entropy = -torch.sum(
        attention_probabilities * torch.log(attention_probabilities + 1e-9), dim=-1
      )  # (batch_size, seq_len)
      attention_dispersion = timestep_entropy.mean(dim=1)  # Média ao longo da sequência
      # Cálculo da dispersão em tokens relevantes
      if tokens_relevantes is not None:
        batch_size = attention_probabilities.size(0)
        relevant_disp_list = []
        for i in range(batch_size):
            indices = tokens_relevantes[i]
            if len(indices) > 0:
                sample_relevant_attn = attention_probabilities[i, indices, :]
                sample_relevant_attn_avg = sample_relevant_attn.mean(dim=-1)
                sample_relevant_dispersion = -torch.sum(
                     sample_relevant_attn_avg * torch.log(sample_relevant_attn_avg + 1e-9)
                )
            else:
                sample_relevant_dispersion = torch.tensor(0.0, device=attention_probabilities.device)
            relevant_disp_list.append(sample_relevant_dispersion)
        relevante_dispersion = torch.stack(relevant_disp_list)
        attention_dispersion = (attention_dispersion + relevante_dispersion) / 2

      # Combine as métricas para calcular o score final
      score = self.autoref(concat_input)  # Projeção final da combinação de estados
      score = score - 0.05 * entropy + 0.1 * similarity + 0.1 * attention_dispersion
      
      score = torch.log(torch.abs(score) + 1e-9) * torch.sign(score) #norma log
      score = 2 * torch.sigmoid(score) - 1 #scaled_sigmoid -1,1

      return score, entropy, similarity, attention_dispersion

    def sentiment_analysis(self, text):
        """
        Realiza análise de sentimentos em uma sequência de texto.
        Retorna um dicionário com os scores de positividade, negatividade, neutralidade e sentimento geral.
        """
        
        if isinstance(text, list):  # Se a entrada for uma lista de tokens
           text = [token for token in text if token != "NULO"]
           text = " ".join(text)
        else:  # Se a entrada for uma string
           text = text.replace("NULO", "").strip()

        sentiment_scores = self.sia.polarity_scores(text)
        #print(f'DEBUG: SIA no texto: {text}')
        return sentiment_scores

    def calcular_tokens_unicos(self, sequencia, padding_token=None):
        """
        Calcula a proporção de tokens únicos em uma sequência, ajustada para a presença de padding.
        Inclui tolerância dinâmica baseada no comprimento da sequência.
        
        Args:
            sequencia (list): Lista de tokens na sequência.
            padding_token (int, optional): Token de padding que deve ser ignorado. Default é None.
        
        Returns:
            float: Penalização proporcional ao desvio dos tokens únicos da tolerância dinâmica.
        """
        # Remove tokens de padding, se especificado
        if isinstance(sequencia, list):  # Se a entrada for uma lista de tokens
           if padding_token is not None:
              text = [token for token in sequencia if token != padding_token]
              sequencia_sem_padding = " ".join(text)
           else:
              sequencia_sem_padding = sequencia
        else:  # Se a entrada for uma string 
              sequencia_sem_padding = sequencia.replace("NULO", "").strip()
           
        # Calcula o número de tokens únicos e o comprimento da sequência sem padding
        num_tokens_unicos = len(set(sequencia_sem_padding))
        comprimento_real = len(sequencia_sem_padding)

        # Define os limites de tokens únicos esperados
        min_tokens_unicos = 1  # Sempre pelo menos 1 token único
        max_tokens_unicos = comprimento_real  # No melhor caso, todos tokens são únicos

        # Calcula a tolerância dinâmica como uma fração do comprimento real
        tolerancia = max(1, round(0.1 * comprimento_real))  # Ajustável, 10% por padrão

        # Verifica se o número de tokens únicos está dentro da faixa esperada
        if num_tokens_unicos < min_tokens_unicos:
            penalizacao = min_tokens_unicos - num_tokens_unicos
        elif num_tokens_unicos > max_tokens_unicos:
            penalizacao = num_tokens_unicos - max_tokens_unicos
        else:
            penalizacao = 0  # Dentro da faixa aceitável

        # Normaliza a penalização para retornar valores proporcionais
        return penalizacao / (tolerancia + comprimento_real)

    def calcular_perda_sentimentos(self, sentimentos_entrada, sentimentos_saida, sentimentos_esperados, alpha=0.5, pesos={"compound": 0.4, "neg": 0.2, "neu": 0.2, "pos": 0.2}):
       """
       Calcula a perda emocional considerando todas as dimensões dos sentimentos (neg, neu, pos, compound) e as variações relativas.
       Inclui pesos relativos para cada componente emocional.
       """
       perda_total = 0.0

       # Calcula a perda para cada componente emocional
       for componente in ["compound", "neg", "neu", "pos"]:
           # Perda baseada na discrepância entre saída e esperado
           perda_componente = abs(sentimentos_esperados[componente] - sentimentos_saida[componente])
        
           # Perda baseada na discrepância na variação emocional
           variacao_esperada = abs(sentimentos_entrada[componente] - sentimentos_esperados[componente])
           variacao_real = abs(sentimentos_entrada[componente] - sentimentos_saida[componente])
           perda_variacional = abs(variacao_esperada - variacao_real)
        
           # Combina as perdas do componente com peso alpha
           perda_componente_total = perda_componente + alpha * perda_variacional
        
           # Ajusta pela importância relativa do componente
           perda_total += pesos[componente] * perda_componente_total

       return perda_total


if args.verbose == 'on':
    print('Definindo o Encoder')
def encoder(texto, tipo, palavra_para_numero):
    return [palavra_para_numero[tipo].get(palavra, OOV[tipo]) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar
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
      return ' '.join([numero_para_palavra[tipo].get(numero, '<OOV>') for numero in texto_codificado])

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
taxa_aprendizado_gerador = 0.01
#LR principal
lr_emb = 0.1
lr_lstm = 0.1
lr_att = 0.1
lr_ac = 0.01
lr_linear = 0.01
#LR autoreferencial:
lr_autoref = 0.01
lr_hproj = 0.01
lr_oproj = 0.01
#LR penalidades:
lr_score = 0.01
lr_sentiment = 0.1
lr_diversidade = 0.05 #0.1 para ativar 0 para desativar
wdecay = 0 #1e-04 para começar a esquecee embeddings
num_samples = args.num_samples #numero de amostras dentro da mesma época
treino = args.treino
textos_falsos = {}
valor = args.valor
output_tipo = args.output_tipo

palavra_para_numero, numero_para_palavra,textos_reais = carregar_vocabulario(pasta, types)

aprendeu = 0
PAD ={}
OOV = {}
for tipo in types:
    PAD[tipo] = len(palavra_para_numero[tipo]) - 1
    OOV[tipo] = PAD[tipo] - 1
print(f'Confirmando Token PAD: {PAD}')
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
   smax == 'off'
   criterio_gerador = torch.nn.KLDivLoss()
else:
   criterio_gerador = torch.nn.MSELoss()

# Criando os modelos gerador,cnn e discriminador para cada tipo de texto
gerador = {}
gerador_novo = {}
for tipo in types:
    # Caminhos dos modelos
    output_size = max_length[tipo]
        
    #gerador_path = os.path.expanduser('doador.pt')
    gerador_path = os.path.expanduser('gerador_gan_'+tipo[1:]+'.pt')
    print('Verificando se o gerador existe para o tipo: ', tipo[1:])
    if os.path.exists(gerador_path):
        print('Carregar o gerador')
        gerador[tipo] = torch.load(gerador_path, map_location=torch.device('cpu'))
        #print('Atualizando gerador')
        #gerador_novo[tipo] = Gerador(len(numero_para_palavra[tipo]),256, 512,len(numero_para_palavra[tipo]))
        #print('Carregar os pesos do modelo original (com as camadas antigas) ')
        #gerador_novo[tipo].load_state_dict(gerador[tipo].state_dict(), strict=False)
        #print('salvando modelo novo')
        #torch.save(gerador_novo[tipo], os.path.expanduser('gerador_gan_' + tipo[1:] + '.pt'))
        #gerador[tipo] = torch.load(gerador_path)
    else:
        print('Criar novo gerador')
        gerador[tipo] = Gerador(len(numero_para_palavra[tipo]),256, 512,len(numero_para_palavra[tipo]))
        #embbeding 256, hidden 512

    #print("Pesos de attention_combine (antes):", gerador[tipo].attention_combine.weight)
    #print("Attention Reset") 
    #gerador[tipo].reset_attention_weights()
    #print("Resetando Linear")
    #gerador[tipo].reset_linear_weights()
    #print("Pesos de attention_combine (depois):", gerador[tipo].attention_combine.weight)
    #print("Resetando LSTM")
    #gerador[tipo].reset_lstm_weights()      
                                                
# Criando os otimizadores para cada modelo
otimizador_gerador = {}
scheduler_gerador = {}
peso_inicio = {}
for tipo in types:
    print('Testando otimizador Adam')
    otimizador_gerador[tipo] = torch.optim.Adam([
        {'params': gerador[tipo].embedding.parameters(), 'lr': lr_emb, 'weight_decay': wdecay},  # Somente no embedding
        {'params': gerador[tipo].lstm.parameters(), 'lr': lr_lstm},  # Sem weight_decay
        {'params': gerador[tipo].attention.parameters(), 'lr': lr_att},
    {'params': gerador[tipo].attention_combine.parameters(), 'lr': lr_ac},
    {'params': gerador[tipo].linear.parameters(), 'lr': lr_linear},
    {'params': gerador[tipo].autoref.parameters(), 'lr': lr_autoref},
    {'params': gerador[tipo].hidden_proj.parameters(), 'lr': lr_hproj},
    {'params': gerador[tipo].output_proj.parameters(), 'lr': lr_oproj}
], lr=taxa_aprendizado_gerador)
    
    
    scheduler_gerador[tipo] = torch.optim.lr_scheduler.ExponentialLR(otimizador_gerador[tipo], gamma=0.99)
    #scheduler_gerador[tipo] = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador_gerador[tipo], mode='min', factor=0.1, patience=10)
    peso_inicio[tipo] = {key: value.clone() for key, value in gerador[tipo].state_dict().items()}
    if debug == 'on':
       print(f'Pesos antes do treinamento: {peso_inicio[tipo]}')

torch.autograd.set_detect_anomaly(True)
print('Iniciando o treinamento')
for tipo in types:
        if args.verbose == 'on':
           print(f'Tipo {tipo} - Treinamento')
        perda_gerador  = 0
        epoca = 1
        
        while epoca <= num_epocas:
           if (epoca % 100 == 0):
               print(f'Epoca multiplo de 100: salvando') 
               torch.save(gerador[tipo], os.path.expanduser('gerador_gan_' + tipo[1:] + '.pt'))
           gerador[tipo].train()
           if args.verbose == 'on':
             print(f'\n\n\nEpoca {epoca}/{num_epocas}')
           textos_reais[tipo].requires_grad_()
           max_len = textos_reais[tipo].size(0)
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

              if seq == max_len:
                #seq = seq % max_len
                seq = 0
                torch.save(gerador[tipo], os.path.expanduser('gerador_gan_' + tipo[1:] + '.pt'))
                print(f'\n\nREINICIANDO DO INICIO DOS TEXTOS!!!\n')
                #print('PARANDO POR ENQUANTO PARA ADICIONAR NOVOS TEXTOS')
                #break
                #rep = rep + 1
                #print(f'\nAUMENTANDO AS REPETIÇÕES\n')
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
                  while texto[-1] == PAD[tipo]:
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
                   new = decoded
               if  modo != 'manual':
                   if output_tipo == 'next':
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
                  decoded = decoder(prompt[0].tolist(),tipo,numero_para_palavra)
                  new = decoded
                  #if noise_dim < esp_size:
                     #prompt = pad_sequence([torch.cat((t, torch.ones(esp_size - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
                  #prompt = pad_sequence([torch.cat((t,torch.zeros(max_length[tipo] - len(t), dtype=torch.int64))) for t in prompt], batch_first=True)
              else:
                  #prompt = ' INICIO ' + prompt + ' FIM '
                  new = prompt
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
               new = decoded
               if args.verbose == 'on':
                   print(f'Prompt aleatorio: {decoded}')
           elif modo == 'curto':
               new = decoded #Para sentimento do Esperado
               match = re.match(r"# (\d+)", decoded)
               vnum = match.group(1) if match else ""
               # Capturar todos os números presentes
               decoded = re.sub(r'\b\d+\b', '', decoded)
               words = nltk.word_tokenize(decoded)
               # Remover stopwords
               stop_words = set(stopwords.words('english'))
               words = [word for word in words if word.casefold() not in stop_words and word != '~' or word in ["T", "$"]]
               # Manter apenas substantivos e nomes próprios (NN, NNP)
               tagged = pos_tag(words)
               #words = [word for word, pos in tagged if pos in ['NN', 'NNP']]
               words = [word for word, pos in tagged if pos in ['NN', 'NNP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'] or word in ["T","$"]] #nomes, verbos e adjetivos
               #Remover palavras repetidas
               seen = set()
               words = [word for word in words if not (word in seen or seen.add(word))]
               # Juntar as palavras de volta em uma string
               words = [vnum] + words
               tokens = nltk.word_tokenize(new)
               words =  [word if word in words else "MASK"  for word in tokens]
               decoded_final = ' '.join(words)
               prompt = encoder(decoded_final,tipo,palavra_para_numero)
               if args.verbose == 'on':
                  print(f'Prompt simplificado: {decoded_final}\n{len(prompt)}')
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
                   while texto[-1] == PAD[tipo]:
                       texto.pop()
                   lista.append(texto)
               prompt = torch.tensor(lista)
           lprompt = prompt.to(torch.int64)
           entrada = lprompt

           if output_tipo == 'next':
               prox = rand + 1
               if prox >= max_len:
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
           if output_tipo == 'same':
               prompt_unpad = prompt_unpad.unsqueeze(0)
           
           max_size = textos_reais[tipo].size(1)
           # Criar uma máscara de atenção
           valid_token_count = lprompt.size(1)  # Quantidade de tokens válidos antes do padding   
           mascara = torch.ones(lprompt.size(0), valid_token_count).bool() # Cria uma máscara de 1s
           mascara = F.pad(mascara, (0, max_length[tipo] - valid_token_count), value=False)  # Preenche o padding com False (0)
           entrada = F.pad(entrada, (0, max_length[tipo] - valid_token_count), value=PAD[tipo])  # Adiciona o token PAD se necessário
           #entrada = pad_sequence([torch.cat((t, torch.full((max_size - entrada.size(1),), PAD[tipo], dtype=torch.int64))) for t in entrada], batch_first=True)
           prompt_unpad = pad_sequence([torch.cat((t, torch.full((max_size - prompt_unpad.size(1),), PAD[tipo] , dtype=torch.int64))) for t in prompt_unpad], batch_first=True)
           
           print(f'Entrada: {entrada}\n forma {entrada.shape}\nmascara {mascara}\nforma {mascara.shape}')
           #print(f'DEBUG: {entrada} \n {prompt_unpad}')
           #exto_falso, hidden, score, entropy, similarity = gerador[tipo](entrada, mask=mascara)
           
           texto_falso, hidden, score, entropy, similarity, attention_dispersion = gerador[tipo](entrada, mask=mascara,tokens_relevantes=None)
           #texto_falso, _, _, entropy, similarity, _ = gerador[tipo](entrada, mask=mascara)

           print(f"Score de confiança: {score}")
           if debug == 'on':
              print(f'Formato saida gerador {texto_falso.shape}')
           #if smax == 'off':
           #    texto_falso = torch.softmax(texto_falso,dim=1)
               #print(f'Saida pos-max {texto_falso}')
           prompt_unpad = prompt_unpad.to(torch.int64)

           #print('1-hot do esperado')
           prompt_hot = F.one_hot(prompt_unpad,len(numero_para_palavra[tipo])).float()
           #prompt_hot = prompt_hot.squeeze(0)
           
           if treino == 'rel':
              #print(f'Log na saida do gerador para {texto_falso.shape}')
              saida_original = texto_falso
              texto_falso = torch.log_softmax(texto_falso,dim=-1)
              #print(f'shape apos log: {texto_falso.shape}')
              print(f'Convertendo saida esperada em probabilidades com forma {prompt_hot.shape}')
              epsilon = 0.01
              smoothed = (1 - epsilon) * prompt_hot + (epsilon / len(numero_para_palavra[tipo]))
              #print(f'Forma do esperado depois da suavização {smoothed.shape} e valor: {smoothed}')
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

           sentiment_entrada = gerador[tipo].sentiment_analysis(decoded)
           sentiment_esperado = gerador[tipo].sentiment_analysis(new)
           sentiment_saida = gerador[tipo].sentiment_analysis(saida)
           
           print(f'Sentinentos Entrada: {sentiment_entrada},\n Saida: {sentiment_saida}\n e Esperado: {sentiment_esperado}\n')

           #if args.verbose == 'on':
           #   print(f'Forma do texto falso {texto_falso.shape} e do Prompt 1-hot: {prompt_hot.shape}')
           perda_gerador = criterio_gerador(texto_falso, prompt_hot)
           
           #perda_gerador = torch.clamp(perda_gerador, max=95)
           print(f'Perda do gerador foi {perda_gerador}')
           # Penalizar confiança com base em -1, 0.5 e 1
           confidence_ideal = 0.5
           penalty_confidence = torch.mean((score - confidence_ideal)**2)  # Penaliza desvios de 0.5
           print(f'Penalidade da Confiança: {penalty_confidence}')

           # Penalizar alta entropia (baixa confiança nas saídas)
           penalty_entropy = torch.mean(entropy) 

           # Recompensar alta similaridade semântica
           reward_similarity = torch.mean(similarity)
           print(f"Recompensa da Similaridade {reward_similarity}")
           # Penalizar dispersão inadequada na atenção
           target_dispersion = 0.7 # 0.3 foca mais, 0.7 abre mais o contexto
           penalty_attention = (target_dispersion - attention_dispersion) ** 2
           print(f'Dispersão Atual: {attention_dispersion}')
           # Calcular o total_penalty com os pesos apropriados
           total_penalty = (
                (0.01 * penalty_confidence) +
                (0.01 * penalty_entropy) +
                (0.001 * reward_similarity) +
                (0.01 * penalty_attention)
           ).squeeze()
           #total_penalty = total_penalty / total_penalty.norm(p=1)  # Normalização L1
           print(f'Calculando pena no autoreferenciamento: {total_penalty}\nEntropia {penalty_entropy} e Similaridade Semantica {reward_similarity}\nPenalidade da Disperção da Atenção {penalty_attention}')
           print(f'Atenção está em {attention_dispersion}')
           #if attention_dispersion < 0.5:
           #    print("Modelo concentrado")
           #elif attention_dispersion > 0.5:
           #    print("Modelo olhando o contexto")
           #else:
           #    print("Modelo equilibrado")
           sentiment_loss = gerador[tipo].calcular_perda_sentimentos(sentiment_entrada, sentiment_saida, sentiment_esperado)
           print(f'Calculando pena no desalinhamento de sentimentos: {sentiment_loss}')
           
           unicos_esperado = gerador[tipo].calcular_tokens_unicos(new, padding_token=None)
           unicos_saida = gerador[tipo].calcular_tokens_unicos(saida, padding_token=PAD[tipo])

           # Calcula a perda de diversidade
           perda_diversidade = abs(unicos_esperado - unicos_saida)

           print(f"Calculando pena na falta de diversidade: {perda_diversidade} por Esperado: {unicos_esperado} e {unicos_saida}")
           print('Aplicando penas na perda')
           perda_gerador += (lr_sentiment * sentiment_loss) + (lr_diversidade * perda_diversidade)
           perda_gerador += total_penalty # Adicionar o total_penalty de forma separada
           print(f'Perda Final depois dos modificadores: {perda_gerador}')
           perda_gerador.backward()
           #torch.nn.utils.clip_grad_norm_(gerador[tipo].parameters(), max_norm=0.5)
           # Aplicar clamp nos pesos do embedding
           #with torch.no_grad():
           #     gerador[tipo].embedding.weight.data.clamp_(-1, 1)
           for nome, parametro in gerador[tipo].named_parameters():
               if parametro.grad is not None:
                  print(f"{nome}: {parametro.grad.norm()}")
           # Atualize os parâmetros do gerador
           if debug == 'on':
               print(f'Gradiente do gerador: {texto_falso.grad}\n Gradiente do esperado: {prompt_hot.grad}')
               #Imprimir os gradientes
               for name, param in gerador[tipo].named_parameters():
                   if param.requires_grad:
                      print(name, param.grad)
           
           otimizador_gerador[tipo].step()
           otimizador_gerador[tipo].zero_grad()
           
           epoca = epoca + 1
           
           if args.verbose == 'on':
              print(f'Tipo {tipo}, Epoca {epoca-1}/{num_epocas} - Perda Final {perda_gerador} Modo - {modo} Indice {rand}')
           else:
              print(f'Perda: {perda_gerador}')
           
           #estatisticas['tipo'].append(tipo)
           estatisticas['perda_gerador'].append(perda_gerador.item())
           estatisticas['sequencia'].append(rand.item())
           #Save stats info
           with open(stats,'w') as f:
                json.dump(estatisticas, f, indent=4)
                f.close()
           with open("saidas.txt",'a') as f:
                f.write("Saida: {saida}\n")
                f.close()
        
           #Fim da epoca para o tipo atual 
        #Fim do tipo atual

#Fim da sessão. Incluir teste:
peso_fim = {}
for tipo in types:
        peso_fim[tipo] = {key: value.clone() for key, value in gerador[tipo].state_dict().items()}
        print(f'Comparando os pesos para o tipo {tipo}:')
        compare_state_dicts(peso_inicio[tipo], peso_fim[tipo])

if args.save_time == 'session':
    print('Salvando modelos')
    if args.save_mode == 'local':
        torch.save(gerador[tipo], os.path.expanduser('gerador_gan_' + tipo[1:] + '.pt'))
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
