#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


#Tipos de dados
black_friday.describe()


# In[4]:


# COLUNAS EXISTENTES
black_friday.columns


# In[5]:


black_friday.info()


# In[6]:


# Quantidade de dados
# Isso vai mostrar a quantidade de dados em cada coluna, se houver colunas sem dados isso deve mostrar agora
black_friday.count()


# In[7]:


# Total de observações e total de colunas
black_friday.shape


# In[8]:


# Total de Colunas e Total de Registros
# Armazenando cada inforação em uma variavel 
observacoes,colunas = black_friday.shape


# In[9]:


# Dados Unicos por coluna
black_friday.nunique()


# In[10]:


black_friday.head(5)


# In[11]:


# Total de Observaçoes
print('Total de Observações : {}'.format(observacoes))


# In[12]:


print('Total de Colunas : {}'.format(colunas))


# In[13]:


# Verificamdo valores Unicos em cada coluna : 
for i in black_friday.columns:
    print('Dados unicos da coluna {0} : {1}'.format(i,black_friday[i].nunique()))


# In[14]:


# Valores faltantes por Coluna
for i in black_friday.columns:
    print('Total de dados faltantes na coluna {0} : {1}'.format(i,black_friday[i].isna().sum()))


# In[15]:


# Porcentagem de dados faltantes em cada coluna
for i in black_friday.columns:
    porcentagem = (black_friday[i].isna().sum()/(observacoes))
    print('Porcentagem de dados Faltantes na coluna  {} : {:.4f}%'.format(i,porcentagem))


# In[16]:


# Porcentagem de dados Presentes 
# Porcentagem de dados faltantes em cada coluna
for i in black_friday.columns:
    porcentagem = black_friday[i].count()/(observacoes/100)
    print('Porcentagem de dados Faltantes na coluna  {} : {:.2f}%'.format(i,porcentagem))


# In[17]:


black_friday[black_friday['Age'] == '26-35'][['User_ID','Age']].head(5)


# In[18]:


black_friday[black_friday['Age'] == '26-35'][['User_ID']].isna().sum()


# In[19]:


black_friday[black_friday['Age'] == '26-35'][['User_ID']].nunique()


# In[20]:


# Contagem de Valores por Coluna

black_friday['Age'].value_counts()


# In[21]:


black_friday.dtypes.nunique()


# In[22]:


# Média das Colunas Product_Category_3 e Product_Category_2
mean_Product_Category_2 = black_friday['Product_Category_2'].mean()
mean_Product_Category_3 = black_friday['Product_Category_3'].mean()
print("{} - {}".format(mean_Product_Category_2,mean_Product_Category_3))


# In[23]:


# Média das Colunas Product_Category_3 e Product_Category_2
std_Product_Category_2 = black_friday['Product_Category_2'].std()
std_Product_Category_3 = black_friday['Product_Category_3'].std()
#print("{} - {}".format(std_Product_Category_2,std_Product_Category_3))
norma_Product_Category_3 = ( black_friday['Product_Category_3'] - mean_Product_Category_3) / std_Product_Category_3
print(norma_Product_Category_3)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[24]:


def q1():
    return black_friday.shape
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[25]:


def q2():
    # Retorne aqui o resultado da questão 2.
    mulheres26_35 = black_friday.loc[black_friday['Age'] == '26-35'].loc[black_friday['Gender']=='F'][['User_ID']].count()
    return int(mulheres26_35)
q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[26]:


def q3():
    # Retorne aqui o resultado da questão 3.
    unicos = black_friday['User_ID'].nunique()
    return unicos
q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[27]:


def q4():
    # Retorne aqui o resultado da questão 4.
    tipos = black_friday.dtypes.nunique()
    #pass
    return tipos
q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[28]:


# Qual porcentagem dos registros possui ao menos um valor null
def q5():
    # Retorne aqui o resultado da questão 5.
    # 0.694
    observacoes = q1()[0]
    porcentagem = (black_friday['Product_Category_3'].isna().sum()/(observacoes))
    return porcentagem
q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.
# 

# In[29]:


def q6():
    # Retorne aqui o resultado da questão 6.
    maior = black_friday['Product_Category_3'].isna().sum()
    return int(maior)
q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[30]:


def q7():
    # Retorne aqui o resultado da questão 7.''
    resposta = black_friday['Product_Category_3'].mode()[0]
    return resposta
q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[31]:


# Normalizar a variavel Purchase
def q8():
    # Retorne aqui o resultado da questão 8.
    norm_Purchase = (black_friday['Purchase'] - black_friday['Purchase'].min())/(black_friday['Purchase'].max() - black_friday['Purchase'].min())
    norm_mean = norm_Purchase.mean()
    return norm_mean
q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[32]:


def q9():        
    #  admissoes[(admissoes.Data > 2017) & (admissoes.Data < 2018)]
    zscore = ((black_friday['Purchase'] - black_friday['Purchase'].mean())/black_friday['Purchase'].std())
    Z = (zscore >-1)&(zscore <1)
    #Z.mean
    #Z = pd.cut(zscore,labels=('menor que 1', 'maior que 1'), bins = (-2,0,2)).value_counts()
    return Z.value_counts()[1]
q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[40]:


def q10():
    # Retorne aqui o resultado da questão 10.
    resp10_a = black_friday['Product_Category_2'].isna().sum() == black_friday['Product_Category_3'].isna().sum()
    resp10_b = black_friday['Product_Category_3'].isna().sum() == black_friday['Product_Category_2'].isna().sum()
    x = (resp10_a,resp10_b)
    return x
q10()


# In[ ]:




