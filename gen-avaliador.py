import torch
import os

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
