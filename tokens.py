import torch
import json
import nltk
import os
import torch.nn as nn
import torch.nn.functional as F
pasta = os.path.expanduser('~/gan/v1')
# Tipos de arquivos que você quer gerar
types = ['.mob']


class Gerador(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_size):
        super(Gerador, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
                                                                          # Camada de Atenção
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_combine = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        # Camada Linear para saída final

        self.linear = nn.Linear(hidden_dim * 2, output_size)
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
        output = self.softmax(output)
        return output, hidden

print('Definindo o Vocabulario')
def carregar_vocabulario(pasta, types):
    palavra_para_numero = {}
    numero_para_palavra = {}

    for tipo in types:
        print(f'Carregando o vocabulário para o tipo {tipo}')
        # Correção na formatação do nome do arquivo JSON
        with open(os.path.join(pasta, f'vocabulario{tipo}.json'),'r') as f:
            palavra_para_numero[tipo] = json.load(f)
            # Criando o dicionário numero_para_palavra
            numero_para_palavra[tipo] = {i: palavra for palavra, i in palavra_para_numero[tipo].items()}
    
    return palavra_para_numero, numero_para_palavra

def export_embeddings_to_json(model, numero_para_palavra, tipo, output_file="embeddings_dump.json"):
    # Pega os pesos da camada de embedding
    embeddings = model.embedding.weight.data

    # Dicionário para armazenar os embeddings no formato JSON
    embeddings_dict = {}

    # Para cada índice no vocab (palavras conhecidas e UNK)
    for index in range(len(embeddings)):
        # Obtém a palavra correspondente ao índice ou usa '<UNK>' se não existir
        token = numero_para_palavra[tipo].get(index, '<UNK>')
        
        # Pega o vetor de embedding associado ao índice
        embedding_vector = embeddings[index].tolist()

        # Adiciona ao dicionário no formato: {índice: {token: vetor de embedding}}
        embeddings_dict[index] = {
            "token": token,
            "embedding_vector": embedding_vector
        }

    # Salva o dicionário de embeddings em um arquivo JSON
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f, indent=4)

palavra_para_numero, numero_para_palavra = carregar_vocabulario(pasta, types)
UNK = len(palavra_para_numero) - 1
gerador = {}
for tipo in types:
    gerador_path = os.path.expanduser('gerador_' + tipo[1:] + '.pt')
    print('Verificando se o gerador existe para o tipo: ', tipo[1:])
    if os.path.exists(gerador_path):
        print('Carregar o gerador')
        gerador[tipo] = torch.load(gerador_path)

print('Exportando Embeddings')
# Itera sobre os tipos individualmente
for tipo in types:
    export_embeddings_to_json(gerador[tipo], numero_para_palavra, tipo=tipo, output_file=f'embeddings_{tipo[1:]}.json')
