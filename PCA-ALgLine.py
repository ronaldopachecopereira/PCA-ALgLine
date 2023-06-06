import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregar os dados do arquivo Excel
dados = pd.read_excel('Luxurywatch.xlsx')

# Selecionar as colunas relevantes para a análise de PCA
dados_selecionados = dados[['Brand', 'Model', 'Case_mat_strap', 'Thickness', 'price']]

# Codificar as variáveis categóricas usando one-hot encoding
dados_codificados = pd.get_dummies(dados_selecionados)

# Normalizar os dados
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(dados_codificados)

# Criar o objeto PCA e ajustá-lo aos dados normalizados
pca = PCA()
dados_transformados = pca.fit_transform(dados_normalizados)

# Calcular a média das distâncias euclidianas entre cada ponto de dados e o ponto médio
ponto_medio = np.mean(dados_transformados, axis=0)
distancias = np.linalg.norm(dados_transformados - ponto_medio, axis=1)

# Encontrar o índice do ponto com o menor distância (melhor padrão de dispersão)
indice_melhor_padrao = np.argmin(distancias)

# Obter o índice do autovetor mais significativo
indice_autovetor_mais_significativo = np.argmax(pca.explained_variance_ratio_)

# Obter a linha correspondente ao autovetor mais significativo
linha_mais_significativa = dados_selecionados.iloc[indice_autovetor_mais_significativo]
# Obter o tamanho da matriz de autovetores
tamanho_matriz = pca.components_.shape

# Obter o índice do autovetor mais significativo
indice_autovetor_mais_significativo = np.argmax(pca.explained_variance_ratio_)

# Obter a linha correspondente ao autovetor mais significativo
linha_mais_significativa = dados_selecionados.iloc[indice_autovetor_mais_significativo]

# Imprimir o tamanho da matriz de autovetores
print("Tamanho da matriz de autovetores:", tamanho_matriz)

# Imprimir a "Brand" e o "Model" do autovetor mais significativo
print("Marca do autovetor mais significativo:", linha_mais_significativa['Brand'])
print("Modelo do autovetor mais significativo:", linha_mais_significativa['Model'])


# Obter os autovalores correspondentes aos componentes principais
autovalores = pca.explained_variance_

# Ordenar os autovalores em ordem decrescente
autovalores_ordem_decrescente = np.sort(autovalores)[::-1]

# Imprimir os 10 principais autovalores e a variância explicada correspondente
print("Os 10 melhores autovalores:")
for i in range(10):
    autovalor = autovalores_ordem_decrescente[i]
    variancia_explicada = autovalor / np.sum(autovalores)  # Variância explicada pelo autovalor
    print(f"Autovalor {i+1}: {autovalor:.4f}, Variância explicada: {variancia_explicada:.4f}")

#A proporção total de variância explicada pelos componentes principais:
print("Proporção total de variância explicada pelos componentes principais:")
total_variance_ratio = np.sum(pca.explained_variance_ratio_)
print(f"Total: {total_variance_ratio:.4f}")

#Número de componentes principais necessários para explicar uma determinada proporção da variância total:
proportion = 0.95  # Exemplo: explicar 95% da variância total
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= proportion) + 1
print(f"Número de componentes para explicar {proportion*100}% da variância total: {n_components}")



# Obter a variável categórica para colorir os pontos
categorias = pd.factorize(dados_selecionados['Brand'])[0]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].scatter(dados_transformados[:, 0], dados_transformados[:, 1], c=categorias)
axs[0, 0].set_xlabel('Componente Principal 1')
axs[0, 0].set_ylabel('Componente Principal 2')
axs[0, 0].set_title('Gráfico de Dispersão em 2D (Colorido)')

axs[0, 1].scatter(dados_transformados[:, 0], dados_transformados[:, 2], c=categorias)
axs[0, 1].set_xlabel('Componente Principal 1')
axs[0, 1].set_ylabel('Componente Principal 3')
axs[0, 1].set_title('Gráfico de Dispersão em 2D (Colorido)')

axs[1, 0].scatter(dados_transformados[:, 0], dados_transformados[:, 3], c=categorias)
axs[1, 0].set_xlabel('Componente Principal 1')
axs[1, 0].set_ylabel('Componente Principal 4')
axs[1, 0].set_title('Gráfico de Dispersão em 2D (Colorido)')

axs[1, 1].scatter(dados_transformados[:, 1], dados_transformados[:, 2], c=categorias)
axs[1, 1].set_xlabel('Componente Principal 2')
axs[1, 1].set_ylabel('Componente Principal 3')
axs[1, 1].set_title('Gráfico de Dispersão em 2D (Colorido)')

# Adicionar a legenda com o nome da linha mais significativa (Brand e Model)
for ax in axs.flat:
    ax.text(dados_transformados[indice_autovetor_mais_significativo, 0],
            dados_transformados[indice_autovetor_mais_significativo, 1],
            linha_mais_significativa['Brand'] + ' ' + linha_mais_significativa['Model'],
            fontsize=10, color='red')

# Plotar o gráfico 3D com cores e legenda
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(dados_transformados[:, 0], dados_transformados[:, 1], dados_transformados[:, 2], c=categorias)
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
plt.title('Gráfico de Dispersão em 3D (Colorido)')
plt.colorbar(sc, label='Marca')

# Adicionar a legenda com o nome da linha mais significativa (Brand e Model)
ax.text(dados_transformados[indice_autovetor_mais_significativo, 0],
        dados_transformados[indice_autovetor_mais_significativo, 1],
        dados_transformados[indice_autovetor_mais_significativo, 2],
        linha_mais_significativa['Brand'] + ' ' + linha_mais_significativa['Model'],
        fontsize=10, color='red')

# Plotar o gráfico 2D adicional com o melhor padrão de dispersão
plt.figure(figsize=(6, 4))
plt.scatter(dados_transformados[:, 0], dados_transformados[:, 1], c=categorias)
plt.scatter(dados_transformados[indice_melhor_padrao, 0], dados_transformados[indice_melhor_padrao, 1], c='red')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Melhor Padrão de Dispersão')

plt.tight_layout()
plt.show()
