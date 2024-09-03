import os
import re
import torch
import nltk
import pathlib
import json

# Pasta onde estão as subpastas .wld, .obj, .mob, .zon, .qst, .shp e .trg
pasta = os.path.expanduser('~/mud/tba2024/lib/world')
pasta_vocab = os.path.expanduser('~/mud/gan/v1/')
types = ['.mob']
UNK = 17921
def remover_caracteres_unicode(texto):
    texto_codificado = texto.encode('ascii', 'ignore')  # Codifica o texto para ASCII e ignora erros
    texto_decodificado = texto_codificado.decode()  # Decodifica o texto de volta para uma string
    return texto_decodificado

# Função para construir e salvar o vocabulário para cada tipo
def carregar_vocabulario(pasta):
    palavra_para_numero = {}
    numero_para_palavra = {}
    
    print(f'Carregando o vocabulário')
    # Correção na formatação do nome do arquivo JSON
    nome = 'vocabulario.mob.json'
    with open(os.path.join(pasta_vocab, nome), 'r') as f:
            palavra_para_numero = json.load(f)
            # Criando o dicionário numero_para_palavra
            numero_para_palavra = {i: palavra for palavra, i in palavra_para_numero.items()}

    return palavra_para_numero, numero_para_palavra

# Função para dividir o texto em sessões
def dividir_em_sessoes(texto):
    #Pre processamento do texto
    texto = re.sub(r'~', ' ~ ', texto)
    texto = re.sub(r'-', ' - ', texto)
    texto = re.sub(r'\+', ' \+ ', texto)
    texto = re.sub(r'\.', ' \. ', texto)
    texto = re.sub(r'@n', ' @ n ', texto)
    texto = re.sub(r'@d', ' @ d ', texto)
    texto = re.sub(r'@b', ' @ b ', texto)
    texto = re.sub(r'@g', ' @ g ', texto)
    texto = re.sub(r'@c', ' @ c ', texto)
    texto = re.sub(r'@r', ' @ r ', texto)
    texto = re.sub(r'@m', ' @ m ', texto)
    texto = re.sub(r'@y', ' @ y ', texto)
    texto = re.sub(r'@w', ' @ w ', texto)
    texto = re.sub(r'@p', ' @ p ', texto)
    texto = re.sub(r'@o', ' @ o ', texto)
    texto = re.sub(r'@D', ' @ D ', texto)
    texto = re.sub(r'@B', ' @ B ', texto)
    texto = re.sub(r'@G', ' @ G ', texto)
    texto = re.sub(r'@C', ' @ C ', texto)
    texto = re.sub(r'@R', ' @ r ', texto)
    texto = re.sub(r'@M', ' @ M ', texto)
    texto = re.sub(r'@Y', ' @ Y ', texto)
    texto = re.sub(r'@W', ' @ W ', texto)
    texto = re.sub(r'@P', ' @ P ', texto)
    texto = re.sub(r'@O', ' @ O ', texto)
    sessoes = re.split(r'(?=#\d)', texto)
    # divide o texto em sessões
    if len(sessoes) > 1 and not sessoes[0].startswith('#'):
        sessoes[1] = sessoes[0] + sessoes[1]
        sessoes = sessoes[1:]
    return sessoes

#Encoder
def encoder(texto, palavra_para_numero):
    return [palavra_para_numero.get(palavra, UNK) for palavra in nltk.word_tokenize(texto)]  # usando o nltk para tokenizar nltk.word_tokenize(texto)

def decoder(texto_codificado, numero_para_palavra):
       # Decodificar o texto usando o dicionário numero_para_palavra do tipo de arquivo correspondente
      return ' '.join([numero_para_palavra.get(numero, '<OOV>') for numero in texto_codificado])

# Lendo, construindo vocabulário e codificando cada arquivo por tipo

palavra_para_numero, numero_para_palavra = carregar_vocabulario(pasta)
# Validação do vocabulário
print(f"Amostra do vocabulário para o tipo: {list(palavra_para_numero.items())[:10]}")
for tipo in types:
 dados_codificados = []
 for nome_arquivo in pathlib.Path(pasta).rglob('*' + tipo):
        if 'index' in nome_arquivo.name or nome_arquivo.suffix == '.pl':  # Ignorando os arquivos 'index', 'index.mini' e '*.pl'
           continue
        print(nome_arquivo)
        with open(nome_arquivo, 'r') as arquivo:
            texto = arquivo.read()
            texto = remover_caracteres_unicode(texto)
            sessoes = dividir_em_sessoes(texto)
            for sessao in sessoes:
                #sessao = ' '.join(sessao)
                #print(f'\nSessao: {sessao}')
                texto_codificado = encoder(sessao, palavra_para_numero)
                if len(texto_codificado) < 1:
                    print(f"Sessao curta: {sessao}")
                elif not sessao.strip():
                    print(f'Sessao vazia: {sessao}')
                elif UNK in texto_codificado:
                    decoded = decoder(texto_codificado,numero_para_palavra)
                    print(f'\nPalavra desconhecida em: {decoded}')
                    print(f'Original: {sessao}')
                
                # Convertendo para torch.float32
                texto_codificado = torch.tensor(texto_codificado, dtype=torch.float32)
                dados_codificados.append(texto_codificado)
                
 # Padronizando o tamanho dos textos codificados
 dados_codificados = torch.nn.utils.rnn.pad_sequence(dados_codificados, batch_first=True)
 # Salvando os dados codificados
 torch.save(dados_codificados, os.path.expanduser(f'~/mud/gan/v1/{tipo[1:]}.pt'))
 print(f'Tipo {tipo} - Formato final: {dados_codificados.shape}')
