import torch
import torch.nn as nn
import torch.nn.functional as F
import os

pasta = os.path.expanduser('~/novels/gen/')

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

# Caminhos dos modelos
doador_path = os.path.expanduser('doador.pt')
print('Verificando se o doador existe:')
if os.path.exists(doador_path):
    print('Carregar o doador')
    doador = torch.load(doador_path)

receptor_path = os.path.expanduser('receptor.pt')
print('Verificando se o receptor existe:')
if os.path.exists(receptor_path):
    print('Carregar o receptor')
    receptor = torch.load(receptor_path)
print('Realizando o Transplante de embeddings')
# Obtenha os pesos da camada de embeddings do modelo doador
donor_embeddings = doador.embedding.weight.data
    
# Verifique se os tamanhos das camadas de embedding são compatíveis
if donor_embeddings.size() != receptor.embedding.weight.data.size():
        raise ValueError("As dimensões das camadas de embedding dos modelos doador e receptor não são compatíveis.")
    
# Copia diretamente todos os pesos da camada de embeddings do doador para o receptor
receptor.embedding.weight.data = donor_embeddings.clone()
print('Salvando Receptor')
torch.save(receptor, os.path.expanduser(receptor_path))
