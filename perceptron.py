import math
# from matplotlib import pyplot as plt


# funções de ativação -----------------------------------------------------
def sign(alfa):
    if alfa <= 0:
        return -1
    else:
        return 1

def logistic(alfa):
    return 1 / 1 + math.exp(-alfa)

def tanh(alfa):
    return (math.exp(2 * alfa) - 1) / (math.exp(2 * alfa) + 1)

def relu(alfa):
    if alfa <= 0:
        return 0
    else:
        return alfa
# --------------------------------------------------------------------------

# função que calcula o produto interno de dois vetores ---------------------
def inner_product(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += float(v1[i]) * float(v2[i])
    return result
# --------------------------------------------------------------------------

# função que multiplica um elemento por um vetor ---------------------------
def product(e, v):
    result = []
    for i in range(len(v)):
        x = float(e) * float(v[i])
        result.append(x)
    return result
# --------------------------------------------------------------------------

# função que soma dois vetores ---------------------------------------------
def sum(v1, v2):
    result = []
    for i in range(len(v1)):
        x = float(v1[i]) + float(v2[i])
        result.append(x)
    return result
# --------------------------------------------------------------------------

# função que calcula a porcentagem de similaridade entre dois vetores ------
def score(v1, v2):
    hits = 0
    for i in range(len(v1)):
        if(float(v1[i]) == float(v2[i])):
            hits += 1
    return hits/len(v1)
# --------------------------------------------------------------------------

# coletando os dados de entada X e y ---------------------------------------
X = []
y = []

arq_address = './datasets/xor.dat'
f = open(arq_address,"r")
row = f.readline().replace('\n','')
while row:
    columns = row.split(" ")
    # guardo todas as colunas, menos a última (predição)
    X.append(columns[:len(columns) - 1])
    # guardo a última coluna (predição)
    y.append(columns[-1])
    row = f.readline().replace('\n', '')
f.close()

# bies
b = -1
# adicionando bies no início de todos exemplos para igualar com o W que passará
# a ter o peso do bies incluso no índice 0
for i in range(len(X)):
    X[i].insert(0, b)

# --------------------------------------------------------------------------

# pesos de conexão ---------------------------------------------------------
W = []

# para cada coluna (feature), adiciono pesos Wi
for i in range(len(X[0])):
    # insere o peso do bies no ínice 0 de W
    if i == 0:
        W.append(0.5)
    # insere os demais pesos da(s) entrada(s)
    else:
        W.append(1) # pode ser randomizado (intervalo de -1 até 1, por exemplo)

# --------------------------------------------------------------------------

# efetuando o treinamento do Perceptron ------------------------------------
# predições treinadas
y_train = []
# número de iterações t do treinamento (Épocas)
T = 1
for t in range(T):
    # guardara as predições em treinamento
    y_training = []

    # percorre todas as predições (vetor y)
    for n in range(len(y)):
        # calculando a predição do perceptron aplicando a função de ativação
        yn = 1 if inner_product(W, X[n]) >= 0 else 0 # sign(inner_product(W, X[n])) 

        # averiguando se houve erro de classificação para aplicar a correção
        if y[n] != yn:
            W = sum(W, product(y[n], X[n]))

        y_training.append(yn)
    
    # atualizando as predições
    y_train = y_training
# --------------------------------------------------------------------------

# plotagem da superfície de decisão (dados + hiperplano) -------------------
# transferindo toda coluna (feature) de X para um array
'''
exemples_in_column = []
for j in range(len(X[0])):
    ex = []
    for i in range(len(X)):
        ex.append(X[i][j])
    exemples_in_column.append(ex)

# plt.plot(W, [0,0,0])
plt.scatter(exemples_in_column[1], exemples_in_column[2], c = y_train)

plt.show()
'''
# --------------------------------------------------------------------------

# acerto(%) das predições treinadas em relação as reais usadas para o treino
print("pesos:", W)
print("score:", score(y_train, y))
# --------------------------------------------------------------------------