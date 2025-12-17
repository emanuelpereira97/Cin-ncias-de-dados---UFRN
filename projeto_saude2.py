import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set_context("paper")

#Após baixar os CSV do SINAN usei esse modelo usado para filtrar as planilhas de dengue por município, criando uma nova planilha com o resultado.
# i = 15
# while i == 25:

#    nome1 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_brutos\DENGBR{i}.csv'
#    nome2 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_filtrados\DENGBR{i}_FILTRED.csv'

#    df1 = pd.read_csv(nome1, low_memory=False)
#    df_filtred1 = df1[df1["SG_UF_NOT"]==24]
#    df_filtred2 = df_filtred1[df_filtred1["ID_MUNICIP"]==240810]
#    df_filtred2.to_csv(nome2)

#    i += 1

#-----------------------------------------------------------------------------------------------


#Criei um novo arquivo CSV com as colunas selecionadas, de 15-20 existe a coluna 'DT_NASC' e entre 21-25 a coluna 'ANO_NASC'
# anos1 = [15,16,17,18,19,20]

# for ano in anos1:
#     nome4 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_filtrados\DENGBR{ano}_FILTRED.csv'
#     nome5 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_select_cols\Dengue_{ano}Cols.csv'

#     colunas = ['DT_NOTIFIC','ID_MUNICIP','ID_UNIDADE','DT_NASC','CS_SEXO','CS_GESTANT','CS_RACA','DT_OBITO']

#     df = pd.read_csv(nome4, low_memory=False)
#     df_selec = df[colunas]
#     df_selec.rename(columns={'DT_NASC' : 'ANO_NASC'}, inplace=True)
#     df_selec['ANO_NASC'] = pd.to_datetime(df_selec['ANO_NASC'])
#     df_selec.loc[:,'ANO_NASC'] = df_selec['ANO_NASC'].dt.year
#     df_selec.loc[:,'ANO_NASC'] = pd.to_numeric(df_selec['ANO_NASC'], errors='coerce').astype('Int64')
#     df_selec.to_csv(nome5)

# anos1 = [21,22,23,24,25]

# for ano in anos1:
#     nome6 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_filtrados\DENGBR{ano}_FILTRED.csv'
#     nome7 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_select_cols\Dengue_{ano}Cols.csv'
#     colunas = ['DT_NOTIFIC','ID_MUNICIP','ID_UNIDADE','ANO_NASC','CS_SEXO','CS_GESTANT','CS_RACA','DT_OBITO']
#     df = pd.read_csv(nome6, low_memory=False)
#     df_selec = df[colunas]
#     df_selec['ANO_NASC'] = pd.to_datetime(df_selec['ANO_NASC'])
#     df_selec.loc[:,'ANO_NASC'] = pd.to_numeric(df_selec['ANO_NASC'], errors='coerce').astype('Int64')

#     df_selec.to_csv(nome7)

#-------------------------------------------------------------------------------------------------------------------------------

# arquivos = [f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_select_cols\Dengue_{ano}Cols.csv' for ano in range(15, 26)]
# df_final = pd.concat((pd.read_csv(arq, low_memory=False) for arq in arquivos), ignore_index=True)
# df_final_sort = df_final.sort_values('DT_NOTIFIC')
# df_final_sort.to_csv("D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_final\DENG_NAT_15_25.csv", index=False)

#-------------------------------------------------------------------------------------------------------------------------------

#carregando e trantando dados das unidades de saude notificadoras, comparação com o cadastro nacional de estabelecimentos de saude-----------

# nome1 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_brutos\cnes_estabelecimentos.csv'
# nome2 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_filtrados\cnes_filtrado.csv'
# nome3 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_filtrados\cnes_filtrado2.csv'
# nome4 = f'D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_final\cnes_selec.csv'

# df1 = pd.read_csv(nome1, sep=';',low_memory=False, encoding='ISO-8859-1')
# df_filtred1 = df1[df1["CO_UF"]==24]
# df_filtred1.to_csv(nome2)

# df2 = pd.read_csv(nome2,low_memory=False, encoding='ISO-8859-1')
# df_filtred2 = df2[df2['CO_IBGE']==240810]
# df_filtred2.to_csv(nome3)

# colunas = ['CO_CNES','CO_UNIDADE','CO_UF','CO_IBGE','NO_FANTASIA','NO_BAIRRO']
# df3 = pd.read_csv(nome3,low_memory=False, encoding='ISO-8859-1')
# df_selec = df3[colunas]
# df_selec.to_csv(nome4)

#----------------------------------------------------------------------------------------------------------------------------------------

#Carrega a fonte de dados esalva em uma variavel
fonte_dados_deng = "D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_final\DENG_NAT_15_25.csv"
fonte_dados_cnes = "D:\Emanuel - Projetos\Indroducao-a-ciencia-de-dados\Projeto und. 3\dados_final\cnes_selec.csv"

df_deng = pd.read_csv(fonte_dados_deng)
df_cnes = pd.read_csv(fonte_dados_cnes)

#remove a coluna que ficou apos a concatenação
df_deng = df_deng.drop('Unnamed: 0', axis=1)
df_cnes = df_cnes.drop('Unnamed: 0', axis=1)

#trata colunas, passando para o tipo inteiro
df_cnes['CO_CNES'] = pd.to_numeric(df_cnes['CO_CNES']).astype('Int64')
df_cnes['CO_UNIDADE'] = pd.to_numeric(df_cnes['CO_UNIDADE']).astype('Int64')
df_cnes['CO_UF'] = pd.to_numeric(df_cnes['CO_UF']).astype('Int64')
df_cnes['CO_IBGE'] = pd.to_numeric(df_cnes['CO_IBGE']).astype('Int64')

#trata o tipo do dado da coluna ano_nasc, cs_gestant, cs_raca e id_unidade, de float para inteiro, de object para datetime (coluna dt)
df_deng['ANO_NASC'] = pd.to_numeric(df_deng['ANO_NASC']).astype('Int64')
df_deng['ID_UNIDADE'] = pd.to_numeric(df_deng['ID_UNIDADE']).astype('Int64')
df_deng['CS_RACA'] = pd.to_numeric(df_deng['CS_RACA']).astype('Int64')
df_deng['DT_NOTIFIC']=pd.to_datetime(df_deng['DT_NOTIFIC'])

#cria as colunas dia, mes, ano e idade com base na data da notificação, normalizando idade
df_deng['DIA']=df_deng['DT_NOTIFIC'].dt.day
df_deng['MES']=df_deng['DT_NOTIFIC'].dt.month
df_deng['ANO']=df_deng['DT_NOTIFIC'].dt.year
df_deng['IDADE_PAC']=df_deng['ANO'] - df_deng['ANO_NASC']
df_deng = df_deng[df_deng['ANO_NASC'] != df_deng['ANO_NASC'].min()]
print(df_deng['IDADE_PAC'].mean())
#normalizando a idade
scaler_idade = MinMaxScaler()
df_deng['IDADE_NORM'] = scaler_idade.fit_transform(df_deng[['IDADE_PAC']])

#substitui o codigo pelo significado, segundo documentação do SINAN
df_deng['CS_SEXO'] = df_deng['CS_SEXO'].map({
    'F': 1, #'FEMININO',
    'M': 0, #'MASCULINO'
})
df_deng['CS_SEXO'] = pd.to_numeric(df_deng['CS_SEXO']).fillna(False).astype('Int64')

df_deng['CS_GESTANT'] = df_deng['CS_GESTANT'].map({
    1: 1, # considero 'sim' - valor na documentação '1 TRIMESTRE'
    2: 1, # considero 'sim' - valor na documentação '2 TRIMESTRE'
    3: 1, # considero 'sim' - valor na documentação '3 TRIMESTRE'
    4: 1, # considero 'sim' - valor na documentação 'IDADE GESTACIONAL IGNORADA',
    5: 0, # Considero 'não' - valor na documentação 'NAO',
    6: 0, # Considero 'não' - valor na documentação 'NAO SE APLICA',
    9: 0, # Considero 'não' - valor na documentação 'IGNORADO'
})
df_deng['CS_GESTANT'] = pd.to_numeric(df_deng['CS_GESTANT']).fillna(False).astype('Int64')

#cria uma coluna binaria de referencia a coluna obito, caso ocorra morte, grava 1 se n, grava 0
df_deng['OBITO_B'] = df_deng['DT_OBITO'].notna().astype(int)

df_deng['BRANCA_B'] = (df_deng['CS_RACA']== 1).fillna(False).astype(int)
df_deng['PRETA_B'] = (df_deng['CS_RACA']== 2).fillna(False).astype(int)
df_deng['AMARELA_B'] = (df_deng['CS_RACA']== 3).fillna(False).astype(int)
df_deng['PARDA_B'] = (df_deng['CS_RACA']== 4).fillna(False).astype(int)
df_deng['INDIGENA_B'] = (df_deng['CS_RACA']== 5).fillna(False).astype(int)
df_deng['IGNORADO_B'] = (df_deng['CS_RACA']== 9).fillna(False).astype(int)

#Junto os dois datafremes------------------------
df_deng_cnes = pd.merge(df_deng,df_cnes,left_on='ID_UNIDADE',right_on='CO_CNES',how='left')

# df_deng_cnes['PETROPOLIS_B'] = (df_deng_cnes['NO_BAIRRO'] == 'PETROPOLIS').fillna(False).astype(int)
# df_deng_cnes['CIDADE_DA_ESPERANCA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'CIDADE DA ESPERANCA') | (df_deng_cnes['NO_BAIRRO'] == 'CIDADE DA ESPERACA').fillna(False).astype(int)
# df_deng_cnes['LAGOA_NOVA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'LAGOA NOVA').fillna(False).astype(int)
# df_deng_cnes['QUINTAS_B'] = (df_deng_cnes['NO_BAIRRO'] == 'QUINTAS').fillna(False).astype(int)
# df_deng_cnes['PITIMBU_B'] = (df_deng_cnes['NO_BAIRRO'] == 'PITIMBU').fillna(False).astype(int)
# df_deng_cnes['N_S_APRESENTACAO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'N S APRESENTACAO') |  (df_deng_cnes['NO_BAIRRO'] == 'NOSSA SRA DA APRESEN') | (df_deng_cnes['NO_BAIRRO'] == 'NOSSA SENHORA DA APR').fillna(False).astype(int)
# df_deng_cnes['TIROL_B'] = (df_deng_cnes['NO_BAIRRO'] == 'TIROL').fillna(False).astype(int)
# df_deng_cnes['CANDELARIA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'CANDELARIA').fillna(False).astype(int)
# df_deng_cnes['CIDADE_ALTA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'CIDADE ALTA').fillna(False).astype(int)
# df_deng_cnes['MAE_LUIZA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'MAE LUIZA').fillna(False).astype(int)
# df_deng_cnes['ROCAS_B'] = (df_deng_cnes['NO_BAIRRO'] == 'ROCAS').fillna(False).astype(int)
# df_deng_cnes['NAZARE_B'] = (df_deng_cnes['NO_BAIRRO'] == 'NAZARE').fillna(False).astype(int)
# df_deng_cnes['PAJUCARA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'PAJUCARA').fillna(False).astype(int)
# df_deng_cnes['ALECRIM_B'] = (df_deng_cnes['NO_BAIRRO'] == 'ALECRIM').fillna(False).astype(int)
# df_deng_cnes['POTENGI_B'] = (df_deng_cnes['NO_BAIRRO'] == 'POTENGI').fillna(False).astype(int)
# df_deng_cnes['REDINHA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'REDINHA').fillna(False).astype(int)
# df_deng_cnes['IGAPO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'IGAPO').fillna(False).astype(int)
# df_deng_cnes['CAPIM_MACIO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'CAPIM MACIO').fillna(False).astype(int)
# df_deng_cnes['FELIPE_CAMARAO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'FELIPE CAMARAO').fillna(False).astype(int)
# df_deng_cnes['VALE_DOURADO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'VALE DOURADO').fillna(False).astype(int)
# df_deng_cnes['NEOPOLIS_B'] = (df_deng_cnes['NO_BAIRRO'] == 'NEOPOLIS').fillna(False).astype(int)
# df_deng_cnes['CIDADE_NOVA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'CIDADE NOVA').fillna(False).astype(int)
# df_deng_cnes['LAGOA_AZUL_B'] = (df_deng_cnes['NO_BAIRRO'] == 'LAGOA AZUL').fillna(False).astype(int)
# df_deng_cnes['BOM_PASTOR_B'] = (df_deng_cnes['NO_BAIRRO'] == 'BOM PASTOR').fillna(False).astype(int)
# df_deng_cnes['NOVA_CIDADE_B'] = (df_deng_cnes['NO_BAIRRO'] == 'NOVA CIDADE').fillna(False).astype(int)
# df_deng_cnes['PONTA_NEGRA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'PONTA NEGRA').fillna(False).astype(int)
# df_deng_cnes['DIX_PET_ROSADO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'DIX PET ROSADO').fillna(False).astype(int)
# df_deng_cnes['SOLEDADE_II_B'] = (df_deng_cnes['NO_BAIRRO'] == 'SOLEDADE II').fillna(False).astype(int)
# df_deng_cnes['PRAIA_DO_MEIO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'PRAIA DO MEIO').fillna(False).astype(int)
# df_deng_cnes['LAGOA_SECA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'LAGOA SECA').fillna(False).astype(int)
# df_deng_cnes['GUARAPES_B'] = (df_deng_cnes['NO_BAIRRO'] == 'GUARAPES').fillna(False).astype(int)
# df_deng_cnes['RIBEIRA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'RIBEIRA').fillna(False).astype(int)
# df_deng_cnes['NORDESTE_B'] = (df_deng_cnes['NO_BAIRRO'] == 'NORDESTE').fillna(False).astype(int)
# df_deng_cnes['NOVA_DESCOBERTA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'NOVA DESCOBERTA').fillna(False).astype(int)
# df_deng_cnes['MORRO_BRANCO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'MORRO BRANCO').fillna(False).astype(int)
# df_deng_cnes['AREIA_PRETA_B'] = (df_deng_cnes['NO_BAIRRO'] == 'AREIA PRETA').fillna(False).astype(int)
# df_deng_cnes['BARRO_VERMELHO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'BARRO VERMELHO').fillna(False).astype(int)
# df_deng_cnes['PLANALTO_B'] = (df_deng_cnes['NO_BAIRRO'] == 'PLANALTO').fillna(False).astype(int)

#colunas_b = ['IDADE_NORM','CS_SEXO','CS_GESTANT','OBITO_B','BRANCA_B','PRETA_B','AMARELA_B','PARDA_B','INDIGENA_B','IGNORADO_B','PETROPOLIS_B','CIDADE_DA_ESPERANCA_B','LAGOA_NOVA_B','QUINTAS_B','PITIMBU_B','N_S_APRESENTACAO_B','TIROL_B','CANDELARIA_B','CIDADE_ALTA_B','MAE_LUIZA_B','ROCAS_B','NAZARE_B','PAJUCARA_B','ALECRIM_B','POTENGI_B','REDINHA_B','IGAPO_B','CAPIM_MACIO_B','FELIPE_CAMARAO_B','VALE_DOURADO_B','NEOPOLIS_B','CIDADE_NOVA_B','LAGOA_AZUL_B','BOM_PASTOR_B','NOVA_CIDADE_B','PONTA_NEGRA_B','DIX_PET_ROSADO_B','SOLEDADE_II_B','PRAIA_DO_MEIO_B','LAGOA_SECA_B','GUARAPES_B','RIBEIRA_B','NORDESTE_B','NOVA_DESCOBERTA_B','MORRO_BRANCO_B','AREIA_PRETA_B','BARRO_VERMELHO_B','PLANALTO_B']

#no inicio considerei os bairros, mas n parece funcionar bem, prefiro agrupar pelo perfil demografico

colunas_b = ['IDADE_NORM','CS_SEXO','CS_GESTANT','OBITO_B','BRANCA_B','PRETA_B','AMARELA_B','PARDA_B','INDIGENA_B','IGNORADO_B']

df_deng_cnes_b=df_deng_cnes[colunas_b]

# #Minha memoria não aguentou
# # obtendo medida de distância baseada no cosseno
# similarity_distance = 1 - cosine_similarity(df_deng_cnes_b)

# # computa a sequência de grupos unidos usando a ligação média
# mergings = linkage(similarity_distance, method='average')

# # configura o dendrograma
# dendrogram_ = dendrogram(mergings,
#                leaf_rotation=90
# )

# quantidade de grupos que vamos realizar os testes
ks = range(1, 8) #comecei com 15 e desci pra 8
# armazena o valor do índice para cada partição gerada
inertias = []
for k in ks:
  # executa o k-means para aquela quantidade de grupos
  km = KMeans(n_clusters=k, random_state=8, n_init='auto')
  # aplica a configuração do agrupamento gerado aos dados para obter o índice
  km.fit(df_deng_cnes_b)
  # calcula o erro quadrático médio
  inertias.append(km.inertia_)

# plota o gráfico
#plt.plot(ks, inertias, marker='o')
#diff = [(1 - (inertias[i] / inertias[i-1]))*100 if i > 0 else 0 for i in range(0,len(inertias))]
#plt.plot(range(1,len(diff)+1), diff, marker='o')

# configuração do gráfico que vai ser gerado
#fig, axs = plt.subplots(3,4, figsize=(15, 6), facecolor='w', edgecolor='k')
#axs = axs.ravel()

# dataframe para armazenar os dados
df = pd.DataFrame()

# intervalo com a quantidade de grupos que vamos dividir os dados
ks = range(2, 9) #comecei com 15 apos verificar pelo metodo do cotovelo, reduzi para 8
for k in ks:
  # executa o kmeans para cada quantidade de grupos
  km = KMeans(n_clusters=k, random_state=8, n_init='auto')
  # relaciona o resultado gerado com os dados da base
  km.fit(df_deng_cnes_b)
  # resgata os rótulos (grupos) para cada objeto da base
  labels = km.labels_.tolist()
  # cria um dataframe para facilitar a contagem dos objetos em cada grupo
  df['labels'] = labels
  # plota a quantidade de objetos em cada grupo
  #df['labels'].value_counts().plot(kind='bar',ax=axs[k-2],ylim=(0,110000))

#Modelo-----(k = 3)---------------------------------------------------------------

kmeans_final = KMeans(
    n_clusters=8,
    random_state=8,
    n_init='auto'
)
kmeans_final.fit(df_deng_cnes_b)
centers = kmeans_final.cluster_centers_

centroids = pd.DataFrame(centers)
# imprimindo os títulos das colunas
centroids.columns = df_deng_cnes_b.keys().values

#print(centroids.head(10))

labels_final = kmeans_final.labels_

df_clusterizado = df_deng_cnes_b.copy()
df_clusterizado['CLUSTER'] = labels_final

#Cria a coluna regiao --------------------------------------------

df_clusterizado['REGIAO'] = df_deng_cnes['NO_BAIRRO'].map({

    'PAJUCARA': 'NORTE',
    'POTENGI': 'NORTE',
    'REDINHA': 'NORTE',
    'IGAPO': 'NORTE',
    'LAGOA AZUL': 'NORTE',
    'N S APRESENTACAO':'NORTE',
    'NOSSA SENHORA DA APR':'NORTE',
    'NOSSA SRA DA APRESEN':'NORTE',
    'LAGOA NOVA': 'SUL',
    'CANDELARIA': 'SUL',
    'CAPIM MACIO': 'SUL',
    'PITIMBU': 'SUL',
    'NEOPOLIS': 'SUL',
    'PONTA NEGRA': 'SUL',
    'NOVA DESCOBERTA': 'SUL',
    'PETROPOLIS': 'LESTE',
    'CIDADE ALTA': 'LESTE',
    'TIROL': 'LESTE',
    'AREIA PRETA': 'LESTE',
    'MAE LUIZA': 'LESTE',
    'ALECRIM': 'LESTE',
    'ROCAS': 'LESTE',
    'RIBEIRA': 'LESTE',
    'PRAIA DO MEIO': 'LESTE',
    'BARRO VERMELHO': 'LESTE',
    'LAGOA SECA': 'LESTE',
    'MORRO BRANCO': 'LESTE',
    'QUINTAS': 'OESTE',
    'NORDESTE': 'OESTE',
    'DIX PET ROSADO': 'OESTE',
    'BOM PASTOR': 'OESTE',
    'NAZARE': 'OESTE',
    'FELIPE CAMARAO': 'OESTE',
    'CIDADE DA ESPERANCA': 'OESTE',
    'CIDADE DA ESPERACA': 'OESTE',
    'PLANALTO': 'OESTE',
    'CIDADE NOVA': 'OESTE',
    'GUARAPES': 'OESTE',
    'NOVA CIDADE': 'OESTE',
    'VALE DOURADO': 'OESTE',
    'SOLEDADE II': 'OESTE'
})


df_clusterizado['FAIXA_ETARIA'] = pd.cut(
    df_deng_cnes['IDADE_PAC'],
    bins=[-1,14,29,59,200],
    labels=['0–14','15–29','30–59','60+']
)

variaveis_modelo = [
    'IDADE_NORM','CS_SEXO','CS_GESTANT','OBITO_B',
    'BRANCA_B','PRETA_B','AMARELA_B','PARDA_B',
    'INDIGENA_B','IGNORADO_B'
]

baseline = (
    df_clusterizado[variaveis_modelo]
        .mean()
        .to_frame(name='BASELINE')
)
#print(baseline)

centroids = (
    df_clusterizado
        .groupby('CLUSTER')[variaveis_modelo]
        .mean()
)
#print(centroids)


comparacao = centroids.subtract(baseline['BASELINE'], axis=1)
comparacao = comparacao.astype(float)

#plt.figure(figsize=(10,6))
sns.heatmap(
    comparacao,
    cmap='RdBu_r',
    center=0,
    annot=True,
    fmt=".2f"
)
#plt.title('Diferença dos clusters em relação ao baseline')
#plt.show()

#print(df_clusterizado['CLUSTER'].value_counts(normalize=True)*100)

sns.barplot(
    data=df_clusterizado,
    x='CLUSTER',
    y='CS_SEXO',
    estimator='mean',
    errorbar=None
)

plt.title('Proporção média de sexo por cluster')
plt.ylabel('Proporção')
plt.xlabel('Cluster')
#plt.show()


sns.barplot(
    data=df_clusterizado,
    x='CLUSTER',
    y='OBITO_B',
    estimator='mean',
    errorbar=None
)

plt.title('Proporção média de óbitos por cluster')
plt.ylabel('Proporção de óbitos')
plt.xlabel('Cluster')
#plt.show()

sns.catplot(
  data=df_clusterizado,
  x='CLUSTER',
  kind='count'
)
# plt.title('Tamanho dos Clusters')
# plt.xlabel('Cluster')
# plt.show()

sns.catplot(
  x='FAIXA_ETARIA',
  data=df_clusterizado,
  hue='CLUSTER',
  kind='count'
)
# plt.title('Distribuição de faixa etária por Clusters')
# plt.xlabel('Faixa')
# plt.show()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA para redução de dimensionalidade
pca = PCA(n_components=2, random_state=8)
componentes = pca.fit_transform(df_deng_cnes_b)

# DataFrame para plotagem
df_pca = pd.DataFrame(
    componentes,
    columns=['PC1', 'PC2']
)
df_pca['CLUSTER'] = labels_final

# Gráfico de dispersão
plt.figure(figsize=(8,6))
for cluster in sorted(df_pca['CLUSTER'].unique()):
    subset = df_pca[df_pca['CLUSTER'] == cluster]
    plt.scatter(
        subset['PC1'],
        subset['PC2'],
        label=f'Cluster {cluster}',
        alpha=0.6
    )

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Clusters com PCA')
plt.legend()
plt.tight_layout()
plt.show()
