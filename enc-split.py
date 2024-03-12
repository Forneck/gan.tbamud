import os
import re
import torch
import nltk
import pathlib

# Pasta onde estão as subpastas .wld, .obj, .mob, .zon, .qst, .shp e .trg
pasta = os.path.expanduser('~/mud/tba2024/lib/world')

types = ['.wld', '.obj', '.mob', '.qst', '.shp', '.trg']

# Construindo o vocabulário
vocab = set()
for nome_arquivo in pathlib.Path(pasta).glob('*/*.*'):
    if 'index' in nome_arquivo.name or nome_arquivo.suffix == '.pl':  # Ignorando os arquivos 'index', 'index.mini' e '*.pl'
        continue
    with open(nome_arquivo, 'r') as arquivo:
        texto = arquivo.read()
        for palavra in nltk.word_tokenize(texto):  # usando o nltk para tokenizar
            vocab.add(palavra)

# Mapeando cada palavra para um número único
palavra_para_numero = {palavra: i for i, palavra in enumerate(vocab)}

# Encoder
def encoder(texto):
    return [palavra_para_numero.get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar

# Função para dividir o texto em sessões
def dividir_em_sessoes(texto):
    sessoes = re.split(r'(?=#\d)', texto)  # divide o texto em sessões
    if len(sessoes) > 1 and not sessoes[0].startswith('#'):
        sessoes[1] = sessoes[0] + sessoes[1]
        sessoes = sessoes[1:]
    return sessoes

# Lendo e codificando cada arquivo por tipo
for tipo in types:
    dados_codificados = []
    for nome_arquivo in pathlib.Path(pasta).glob(tipo[1:] + '/*' + tipo):
        if 'index' in nome_arquivo.name or nome_arquivo.suffix == '.pl':  # Ignorando os arquivos 'index', 'index.mini' e '*.pl'
            continue
        print(nome_arquivo)  # imprime o nome do arquivo na tela
        with open(nome_arquivo, 'r') as arquivo:
            texto = arquivo.read()
            sessoes = dividir_em_sessoes(texto)
            for sessao in sessoes:
                texto_codificado = encoder(sessao)
                # Convertendo para torch.float32
                texto_codificado = torch.tensor(texto_codificado, dtype=torch.float32)
                dados_codificados.append(texto_codificado)
    # Padronizando o tamanho dos textos codificados
    dados_codificados = torch.nn.utils.rnn.pad_sequence(dados_codificados, batch_first=True)
    # Salvando os dados codificados
    torch.save(dados_codificados, os.path.expanduser('~/mud/gan/' + tipo[1:] + 'r1' + '.pt'))
