import os
import re
import torch
import nltk
import pathlib
import json

# Pasta onde estão as subpastas .wld, .obj, .mob, .zon, .qst, .shp e .trg
pasta = os.path.expanduser('~/mud/tba2024/lib/world')

types = ['.mob']

# Função para construir e salvar o vocabulário para cada tipo
def carregar_vocabulario(pasta, tipo):
    vocab = set()
    # Ajuste na função glob para buscar dentro das subpastas corretas
    for nome_arquivo in pathlib.Path(pasta).rglob('*' + tipo):
        if 'index' in nome_arquivo.name or nome_arquivo.suffix == '.pl':  # Ignorando os arquivos 'index', 'index.mini' e '*.pl'
            continue
        print(f"Lendo o arquivo: {nome_arquivo}")  # Confirmação de que o arquivo está sendo lido
        with open(nome_arquivo, 'r') as arquivo:
            texto = arquivo.read()
            for palavra in nltk.word_tokenize(texto):  # usando o nltk para tokenizar
                vocab.add(palavra)
    if not vocab:
        print(f"Nenhum vocabulário foi construído para o tipo {tipo}. Verifique os arquivos de entrada.")
    else:
        print(f"Vocabulário para o tipo {tipo} construído com sucesso. Número de palavras: {len(vocab)}")
    palavra_para_numero = {palavra: i for i, palavra in enumerate(sorted(vocab))}
    with open(f'vocabulario{tipo}.json', 'w') as f:
        json.dump(palavra_para_numero, f)
    return palavra_para_numero

# Função para dividir o texto em sessões
def dividir_em_sessoes(texto):
    sessoes = re.split(r'(?=#\d)', texto)  # divide o texto em sessões
    if len(sessoes) > 1 and not sessoes[0].startswith('#'):
        sessoes[1] = sessoes[0] + sessoes[1]
        sessoes = sessoes[1:]
    return sessoes

#Encoder
def encoder(texto, palavra_para_numero):
    return [palavra_para_numero.get(palavra, 0) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar

# Lendo, construindo vocabulário e codificando cada arquivo por tipo
for tipo in types:
    print(f"Processando o tipo de arquivo: {tipo}")
    palavra_para_numero = carregar_vocabulario(pasta, tipo)
    # Validação do vocabulário
    print(f"Amostra do vocabulário para o tipo {tipo}: {list(palavra_para_numero.items())[:10]}")
    dados_codificados = []
    for nome_arquivo in pathlib.Path(pasta).glob(tipo[1:] + '/*' + tipo):
        if 'index' in nome_arquivo.name or nome_arquivo.suffix == '.pl':  # Ignorando os arquivos 'index', 'index.mini' e '*.pl'
            continue
        print(nome_arquivo)
        with open(nome_arquivo, 'r') as arquivo:
            texto = arquivo.read()
            sessoes = dividir_em_sessoes(texto)
            for sessao in sessoes:
                texto_codificado = encoder(sessao, palavra_para_numero)
                # Convertendo para torch.float32
                texto_codificado = torch.tensor(texto_codificado, dtype=torch.float32)
                dados_codificados.append(texto_codificado)
                
    # Padronizando o tamanho dos textos codificados
    dados_codificados = torch.nn.utils.rnn.pad_sequence(dados_codificados, batch_first=True)
    # Salvando os dados codificados
    torch.save(dados_codificados, os.path.expanduser(f'~/mud/gan/v1/{tipo[1:]}.pt'))
