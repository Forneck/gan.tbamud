import json
import torch
import os
import torch.nn as nn

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



def load_embeddings_from_json(json_file):
    with open(json_file, 'r') as f:
        embeddings_dict = json.load(f)
    embeddings_data = {int(index): data['embedding_vector'] for index, data in embeddings_dict.items()}
    return embeddings_data

def set_embeddings_to_model(model, embeddings_data, embedding_dim):
    vocab_size = len(embeddings_data)
    embedding_tensor = torch.zeros(vocab_size, embedding_dim)
    for index, vector in embeddings_data.items():
        embedding_tensor[index] = torch.tensor(vector)
    model.embedding.weight.data = embedding_tensor

# Suponha que você tenha uma lista de tipos
types = ['.mob']

# Defina a dimensão dos embeddings (se conhecida, ajuste conforme necessário)
embedding_dim = 256  # exemplo de valor

receptor = {}
for tipo in types:
    receptor_path = os.path.expanduser('receptor_' + tipo[1:] + '.pt')
    print('Verificando se o receptor existe para o tipo: ', tipo[1:])
    if os.path.exists(receptor_path):
        print('Carregar o receptor')
        receptor[tipo] = torch.load(receptor_path)

# Suponha que `modelo_receptor` é o seu modelo receptor
for tipo in types:
    json_file = f'embeddings_{tipo[1:]}.json'
    print(f'Carregando embeddings de {json_file}')
    
    embeddings_data = load_embeddings_from_json(json_file)
    
    # Atualize o modelo com os embeddings carregados
    set_embeddings_to_model(receptor[tipo], embeddings_data, embedding_dim)
    torch.save(receptor[tipo], os.path.expanduser(receptor_path))
