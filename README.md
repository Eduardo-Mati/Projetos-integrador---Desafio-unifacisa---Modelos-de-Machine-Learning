<img width="1768" height="716" alt="image" src="https://github.com/user-attachments/assets/a7241bea-dd53-466e-967f-1a97fbb3206e" />## 🚀 Como Executar as Questões no Google Colab

O projeto está dividido em múltiplos notebooks, cada um correspondendo a uma questão ou etapa da análise. Todos foram preparados para rodar diretamente no Google Colab.

Abaixo você encontrará o mapeamento de cada notebook, os dados que ele utiliza e um link direto para abri-lo no Colab.

### Mapeamento de Questões e Datasets

| Arquivo do Notebook | Dataset(s) Utilizado(s)             | Descrição da Questão                                       | Link Direto para o Colab                                                                                                                                                             |
| ------------------- | ----------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Q5.ipynb** | `heart_desease_data.csv`     | <img width="1215" height="569" alt="image" src="https://github.com/user-attachments/assets/8becbb15-4e47-46b5-94e7-4c3cbb54714c" /> | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eduardo-Mati/Projetos-integrador---Desafio-unifacisa---Modelos-de-Machine-Learning/blob/main/Q5.ipynb) |
| **Q6.ipynb** | `California Housing do Scikit-Learn, já presente importado no código`     | <img width="1199" height="493" alt="image" src="https://github.com/user-attachments/assets/937f351c-7431-4eeb-a88c-257fcd63565e" /> | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eduardo-Mati/Projetos-integrador---Desafio-unifacisa---Modelos-de-Machine-Learning/blob/main/Q6.ipynb) |
| **Q7.ipynb** | `Groceries_dataset.csv`     | <img width="1161" height="465" alt="image" src="https://github.com/user-attachments/assets/aef20693-4f65-4ac9-a84f-50c8b47fb633" /> | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eduardo-Mati/Projetos-integrador---Desafio-unifacisa---Modelos-de-Machine-Learning/blob/main/Q7.ipynb) |
| **Q8.ipynb** | `avaliacoes_filmes.csv`     | <img width="1178" height="517" alt="image" src="https://github.com/user-attachments/assets/eb281138-b7ef-4da9-b1fb-cc78a44664ef" /> | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eduardo-Mati/Projetos-integrador---Desafio-unifacisa---Modelos-de-Machine-Learning/blob/main/Q8.ipynb) |
| **Q9.ipynb** | `paultimothymooney/chest-xray-pneumonia já importado dentro do código`     | <img width="1191" height="535" alt="image" src="https://github.com/user-attachments/assets/b80529ba-60eb-43c4-938a-4a1227973309" /> | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eduardo-Mati/Projetos-integrador---Desafio-unifacisa---Modelos-de-Machine-Learning/blob/main/Q9.ipynb) |
| **Q10.ipynb** | `store.csv`  | <img width="918" height="732" alt="image" src="https://github.com/user-attachments/assets/4fcbfef3-8f04-4310-81e0-13e36c55c976" /> | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eduardo-Mati/Projetos-integrador---Desafio-unifacisa---Modelos-de-Machine-Learning/blob/main/Q10.ipynb) |


### Instruções de Execução

1.  **Escolha a Questão:** Na tabela acima, clique no botão "Open In Colab" correspondente à questão que você deseja executar.
2.  **Execute a Célula de Configuração:** Como antes, a **primeira célula** de cada notebook deve ser executada primeiro para clonar o repositório e instalar as bibliotecas. Isso garante que o notebook terá acesso aos arquivos de dataset corretos.
3.  **Execute o Restante do Código:** Após a configuração, execute as outras células em sequência para ver a análise completa daquela questão.

### Funcionamento do código por etapa


Questão 5 (Intermediário) - Diagnóstico de Doenças Cardíaca


<img width="585" height="851" alt="image" src="https://github.com/user-attachments/assets/0a36d36a-3398-4010-bd09-8d8f25f8310c" />

 
1. Importação das Bibliotecas
O código é organizado em seções para importar diferentes tipos de ferramentas:

•	Manipulação de Dados:

 o	import pandas as pd: A biblioteca principal para carregar e manipular os dados em tabelas (DataFrames).

 o	import numpy as np: Usada para operações matemáticas e numéricas.

•	Visualização:

 o	import matplotlib.pyplot as plt e import seaborn as sns: Duas bibliotecas usadas para criar gráficos e visualizar os dados.

•	Pré-processamento:

 o	from sklearn.model_selection import train_test_split: Função essencial para dividir os dados em conjuntos de treino e teste.

 o	from sklearn.preprocessing import ...: Importa ferramentas para "limpar" os dados antes de usá-los:

  	StandardScaler, MinMaxScaler: Para normalizar ou padronizar as colunas numéricas (colocá-las na mesma escala).

  	OneHotEncoder: Para converter colunas de texto (categóricas) em números que o modelo possa entender.

 o	from sklearn.impute import SimpleImputer: Ferramenta para preencher valores ausentes (NaN) nos dados.

•	Modelos de Classificação:

 o	from sklearn.ensemble import RandomForestClassifier: Importa o modelo Random Forest, um modelo robusto baseado em árvores de decisão.

 o	from sklearn.svm import SVC: Importa o Support Vector Machine (SVM), outro modelo de classificação poderoso.

 o	from sklearn.linear_model import LogisticRegression: Importa a Regressão Logística, um terceiro modelo (linear) que também é ótimo para classificação.

•	Métricas de Avaliação:

 o	from sklearn.metrics import ...: Importa todas as funções necessárias para avaliar o desempenho dos modelos, conforme solicitado na atividade: accuracy_score (acurácia), precision_score (precisão), recall_score (recall), f1_score, roc_auc_score (curva ROC-AUC) e confusion_matrix (matriz de confusão).

•	Explicabilidade (XAI):

 o	import shap: Importa a biblioteca SHAP. Esta é uma ferramenta avançada que será usada para responder à segunda pergunta da atividade: "Quais variáveis mais impactaram na previsão?".

•	Configurações Adicionais:

 o	sns.set_style(...): Define um estilo visual (whitegrid) para todos os gráficos do Seaborn.

3. Carregamento dos Dados

•	df = pd.read_csv('heart_desease_data.csv'): Este comando usa o Pandas para ler o arquivo heart_desease_data.csv e o armazena na variável df (DataFrame), que é a nossa tabela principal de trabalho.

<img width="886" height="493" alt="image" src="https://github.com/user-attachments/assets/af3b4d66-8179-4aa2-9654-1faa8a85508c" />

 
Este bloco de código está preparando os dados para o modelo de machine learning.

1.	Ele primeiro separa o dataset em duas partes: as features (dados de entrada, como idade, colesterol) e o target (o que queremos prever: se tem ou não doença).

2.	Em seguida, ele usa o StandardScaler para padronizar todas as features.

3.	Por que fazer isso? A padronização coloca todas as variáveis na mesma escala (média 0, desvio padrão 1). Isso é crucial para que modelos como o SVM funcionem corretamente e não deem importância indevida a variáveis que possuem números grandes (como 'colesterol') em vez de variáveis com números pequenos (como 'sex').

 <img width="886" height="388" alt="image" src="https://github.com/user-attachments/assets/03ef4bbd-5ad4-4a59-9831-02c4a7a874ea" />

Este bloco de código define uma função chamada plot_corr para visualizar a matriz de correlação dos dados.

1.	corr = df_scaled.corr(): Este comando calcula a matriz de correlação. Ela mostra um valor (entre -1 e 1) que mede a força da relação linear entre cada par de variáveis (por exemplo, como a "idade" se correlaciona com o "colesterol").

2.	ax.matshow(corr): Este comando usa o Matplotlib para desenhar um mapa de calor (heatmap) dessa matriz. As cores no gráfico representam os valores da correlação (cores quentes para correlação positiva, frias para negativa).

3.	plot_corr(df_scaled): Finalmente, o código chama a função para exibir esse mapa de calor.

Por que fazer isso? Para entender visualmente quais features (variáveis) estão fortemente conectadas umas com as outras, o que pode ser importante para a seleção de features e para entender o comportamento do dataset.

<img width="886" height="528" alt="image" src="https://github.com/user-attachments/assets/b71832b9-0e13-4b69-9dbb-02c1e7841f40" />

Este bloco de código realiza duas verificações cruciais da qualidade e integridade dos dados após a padronização:

1.	df_scaled.isnull().sum()

 o	O que faz: Este comando verifica cada coluna do DataFrame df_scaled e conta quantos valores nulos (ausentes, NaN) existem em cada uma.

 o	Por que fazer isso: Modelos de machine learning não conseguem lidar com dados ausentes. A saída (que mostra 0 para todas as colunas) é uma excelente notícia, pois confirma que não há valores nulos e que não precisaremos de técnicas de imputação (preenchimento de dados faltantes).

2.	df_scaled.describe()

 o	O que faz: Este comando gera um resumo estatístico detalhado para cada coluna numérica.

 o	Por que fazer isso: Esta é a prova final de que o StandardScaler (do bloco anterior) funcionou perfeitamente. Podemos observar na saída:

  	mean (média): Todos os valores da média são números extremamente pequenos, muito próximos de 0 (ex: 4.690515e-17, que é 0.00...0469).

  	std (desvio padrão): Todos os valores do desvio padrão são exatamente 1.001654e+00 (que é 1.001654, um valor muito próximo de 1, a pequena diferença se deve ao cálculo amostral do pandas).

 o	Isso confirma que todas as features estão agora na mesma escala padronizada, prontas para serem usadas pelos modelos.

<img width="886" height="566" alt="image" src="https://github.com/user-attachments/assets/94432f73-d26b-4710-ad29-28fa7c386a92" />

 
Este bloco de código realiza duas tarefas principais: dividir os dados em conjuntos de treino/teste e treinar o primeiro modelo de classificação.

1.	Preparação dos Dados para sklearn:

 o	data_treino = np.array(df_scaled): Converte o DataFrame df_scaled (as features padronizadas) em um array NumPy.

 o	data_classif = np.array(target): Converte a série target (as respostas "0" ou "1") em um array NumPy.

 o	Por quê? As bibliotecas do sklearn (como train_test_split e os modelos) são otimizadas para trabalhar com arrays NumPy.

2.	Definição da Base de Treino e Teste:

 o	x_treino, x_val, y_treino, y_val = train_test_split(...): Este comando divide os dados em quatro conjuntos:

  	x_treino e y_treino: 70% dos dados, que serão usados para treinar o modelo.

  	x_val e y_val: 30% dos dados (definido por test_size=0.30), que serão guardados para testar o modelo.

 o	Por quê? Esta é a etapa mais importante para uma avaliação honesta. O modelo só pode "aprender" com os dados de treino. Os dados de teste (x_val, y_val) são usados no final para ver o quão bem o modelo generaliza para dados que ele nunca viu.

3.	Imprimindo os Resultados (Verificação):

 o	Os comandos print(...) apenas calculam e mostram as porcentagens da divisão (69.97% treino, 30.03% teste), confirmando que a separação funcionou.

4.	Treinando o Modelo Random Forest:

 o	rf = RandomForestClassifier(random_state=42): Inicializa o primeiro modelo, o RandomForestClassifier. O random_state=42 é usado para garantir que os resultados sejam reproduzíveis.

 o	rf.fit(x_treino, y_treino): Este é o comando de treinamento. O modelo rf "aprende" os padrões nos dados de treino (x_treino) para prever as respostas (y_treino).

<img width="886" height="908" alt="image" src="https://github.com/user-attachments/assets/85af8f11-ab03-4985-841a-339dbab26e89" />

 
Este bloco de código avalia o desempenho do modelo RandomForestClassifier que acabamos de treinar.

Ele faz duas avaliações distintas:

1.	Avaliação nos Dados de TREINO:

 o	y_pred_treino = rf.predict(x_treino): O modelo faz previsões sobre os próprios dados que usou para treinar.

 o	accuracy_score(y_treino, y_pred_treino): Compara as previsões com as respostas corretas.

 o	Resultado (1.0): A acurácia foi de 100%. Isso indica que o modelo "decorou" perfeitamente os dados de treino, um sinal clássico de overfitting. O que importa de verdade é o teste abaixo.

2.	Avaliação nos Dados de TESTE (Validação):

 o	y_pred_val = rf.predict(x_val): O modelo faz previsões sobre os dados de teste (x_val), que ele nunca viu antes. Este é o teste real.

 o	confusion_matrix(...): Gera a Matriz de Confusão.

  	A saída [[41 9] [11 30]] significa (para as classes [1, 0]):

  	41 Verdadeiros Positivos: Acertou 41 pacientes que tinham doença.

  	9 Falsos Negativos: Errou 9 pacientes que tinham doença (disse que não tinham).

  	11 Falsos Positivos: Errou 11 pacientes que não tinham doença (disse que tinham).

  	30 Verdadeiros Negativos: Acertou 30 pacientes que não tinham doença.

 o	classification_report(...): Gera o relatório completo de métricas.

  	accuracy (Acurácia Total): 0.78 (O modelo acertou 78% de todas as previsões).

  	precision (classe 1): 0.79 (Quando o modelo disse "tem doença", ele estava certo 79% das vezes).

  	recall (classe 1): 0.82 (O modelo conseguiu identificar 82% de todos os pacientes que realmente tinham a doença).

<img width="600" height="827" alt="image" src="https://github.com/user-attachments/assets/69ec9de8-acd4-40e9-b1b4-6e4115d3f734" />

 
Este bloco de código treina e avalia o segundo modelo da atividade, o Support Vector Machine (SVM), para que possamos compará-lo com o Random Forest.

1.	Treinando o Modelo SVM:

 o	svm = SVC(random_state=42): Inicializa o modelo SVC (Support Vector Classifier).

 o	svm.fit(x_treino, y_treino): Treina o modelo SVM usando os mesmos dados de treino (x_treino, y_treino) que o Random Forest usou.

2.	Avaliação nos Dados de TREINO:

 o	y_pred_treino = svm.predict(x_treino): O modelo faz previsões sobre os dados de treino.

 o	accuracy_score(...): Calcula a acurácia.

 o	Resultado (0.929...): A acurácia foi de 93%. Isso é alto, mas (ao contrário do Random Forest) não é 100%, o que sugere que o SVM está "decorando" menos os dados de treino e pode ter uma capacidade de generalização melhor.

3.	Avaliação nos Dados de TESTE (Validação):

 o	y_pred_val = svm.predict(x_val): O modelo SVM faz previsões sobre os dados de teste (x_val).

 o	confusion_matrix(...): Gera a Matriz de Confusão.

  	A saída [[43 7] [14 27]] significa (para as classes [1, 0]):

  	43 Verdadeiros Positivos: Acertou 43 pacientes que tinham doença.

  	7 Falsos Negativos: Errou 7 pacientes que tinham doença (disse que não tinham).

  	14 Falsos Positivos: Errou 14 pacientes que não tinham doença (disse que tinham).

  	27 Verdadeiros Negativos: Acertou 27 pacientes que não tinham doença.

 o	classification_report(...): Gera o relatório de métricas.

  	accuracy (Acurácia Total): 0.77 (O modelo acertou 77% de todas as previsões).

  	precision (classe 1): 0.75 (Quando o modelo disse "tem doença", ele estava certo 75% das vezes).

  	recall (classe 1): 0.86 (O modelo conseguiu identificar 86% de todos os pacientes que realmente tinham a doença).

Comparação rápida (SVM vs. RF): O SVM teve uma acurácia total ligeiramente menor (77% vs 78% do RF), mas alcançou um Recall maior (86% vs 82%), o que significa que foi um pouco melhor em encontrar os pacientes que de fato estavam doentes.

<img width="886" height="498" alt="image" src="https://github.com/user-attachments/assets/1082899f-67ae-4790-a154-abeb000abba7" />

 
Este bloco de código responde à segunda pergunta da atividade: "Quais variáveis mais impactaram na previsão?", especificamente para o modelo Random Forest.

1.	importances = rf.feature_importances_:

 o	O que faz: O modelo Random Forest (rf), após ser treinado, armazena um "score" de importância para cada feature (coluna) em um atributo chamado .feature_importances_. Esse score mede o quanto cada feature (idade, sexo, colesterol, etc.) contribuiu para tomar as decisões corretas.

2.	indices = np.argsort(importances)[::-1]:

 o	O que faz: Este comando pega a lista de scores (importances) e a ordena, mas em vez de retornar os scores ordenados, ele retorna os índices (posições) das colunas, da mais importante para a menos importante.

  	np.argsort() ordena do menor para o maior.

  	[::-1] inverte a lista, para que o mais importante venha primeiro.

3.	print(indices):

 o	O que faz: Imprime o ranking das features por importância.

 o	Saída: [ 2 7 9 0 11 12 4 3 10 8 1 6 5]

 o	O que significa? Usando a tabela df_scaled como referência (onde 'age' é o índice 0, 'sex' é o 1, 'cp' é o 2, etc.), este ranking nos diz que:

  	A feature mais importante é a de índice 2 (cp - tipo de dor no peito).

  	A segunda mais importante é a de índice 7 (thalach - frequência cardíaca máxima).

  	A terceira mais importante é a de índice 9 (oldpeak).

  	A quarta mais importante é a de índice 0 (age).

 e assim por diante.

<img width="886" height="652" alt="image" src="https://github.com/user-attachments/assets/056fac18-556d-425f-9119-9b7bce70f39e" />

 
Este bloco de código instala uma nova biblioteca (eli5) para responder à mesma pergunta ("Quais variáveis mais impactaram?"), mas desta vez para o modelo SVM.

O SVM não possui um atributo .feature_importances_ simples como o Random Forest, por isso é necessário usar uma técnica diferente chamada "Importância por Permutação".

1.	%pip install eli5: Instala a biblioteca eli5, que é especializada em "explicar" modelos de machine learning.

2.	from eli5.sklearn import PermutationImportance: Importa a ferramenta de "Importância por Permutação".

3.	perm = PermutationImportance(svm, ...).fit(x_val, y_val): Este é o comando principal.

 o	O que faz: Ele usa o modelo svm treinado e os dados de teste (x_val, y_val).

 o	Como funciona:

1.	Primeiro, ele mede a acurácia do SVM nos dados de teste (a "pontuação base").

2.	Depois, ele pega uma coluna (ex: 'age'), embaralha os valores dela aleatoriamente e mede a acurácia de novo.

3.	Se a acurácia cair muito, significa que essa coluna era muito importante.

4.	Se a acurácia não mudar (ou até melhorar um pouco), significa que a coluna era irrelevante ou prejudicial.

 o	Ele repete esse processo para todas as colunas.

4.	eli5.show_weights(perm, feature_names=feature_names): Este comando exibe os resultados em uma tabela formatada.

O Resultado (A Tabela):

A tabela mostra o "Peso" (o quanto a acurácia do modelo caiu quando a feature foi embaralhada).

•	Para o SVM, a feature mais importante foi ca (número de vasos principais).

•	A segunda foi thal.

•	A terceira foi cp (tipo de dor no peito).

•	Interessante notar que chol (colesterol) teve uma importância negativa (-0.0223), o que sugere que essa feature pode ter mais confundido o SVM do que ajudado.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Questão 6 (Avançado) – Previsão do Valor de Imóveis

Trabalho desenvolvido no Google Colab utilizando técnicas de Machine Learning para prever o valor médio de imóveis na Califórnia, com base no dataset California Housing.

Bibliotecas usadas

Antes de executar o código, foram utilizadas as seguintes bibliotecas (algumas precisam ser instaladas):

- pandas: manipulação e análise de dados (DataFrame).
- numpy: operações numéricas e arrays.
- matplotlib.pyplot: criação de gráficos.
- seaborn: aprimora visualizações estatísticas (estilos e gráficos).
- scikit-learn (sklearn): carregamento do dataset e ferramentas de pré-processamento, divisão de dados e modelos (fetch_california_housing, train_test_split, StandardScaler, LinearRegression, métricas como mean_squared_error e r2_score).
- xgboost: algoritmo de boosting baseado em árvores (XGBRegressor).
- tensorflow / keras: construção e treinamento de redes neurais (Sequential, Dense).

Comandos usados para instalar (exemplo):
`!pip install pandas numpy scikit-learn matplotlib seaborn xgboost tensorflow`

Essas bibliotecas permitem carregar, preparar, visualizar os dados e aplicar diferentes modelos de regressão.

<img width="900" height="468" alt="image" src="https://github.com/user-attachments/assets/a72a9225-1889-4519-8754-fe99a2cd3c46" />

 
Aqui iniciamos o projeto importando o dataset California Housing, que contém informações sobre a renda, localização e características das casas da Califórnia.

<img width="900" height="591" alt="image" src="https://github.com/user-attachments/assets/9ae7ce7e-6f83-4ec6-8e5e-9a1a6e355aca" />

 
Nesta parte, são instaladas e importadas as bibliotecas necessárias, como Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn, XGBoost e TensorFlow.

<img width="900" height="463" alt="image" src="https://github.com/user-attachments/assets/a1bf6511-3dc1-4f5d-8066-30baefeecb23" />

 
Aqui configuramos o estilo visual dos gráficos com o Seaborn e o Matplotlib, deixando os resultados mais claros e bonitos.

<img width="900" height="417" alt="image" src="https://github.com/user-attachments/assets/056fc582-1fee-4b41-9433-b9dc0a7f6783" />

 
Nesta célula, ocorre o pré-processamento dos dados: normalização e separação entre treino e teste para preparar o dataset.


<img width="900" height="422" alt="image" src="https://github.com/user-attachments/assets/49163ab5-43b7-4651-a650-c5dcbb7068d2" />

 
Aqui visualizamos as primeiras linhas do dataset e analisamos o significado de cada coluna, como renda mediana, idade média das casas e número médio de cômodos.


<img width="900" height="186" alt="image" src="https://github.com/user-attachments/assets/ca10c488-47dc-49c8-b353-3e10f44a0485" />

 
Nesta parte, o código faz a normalização com o StandardScaler, deixando os dados padronizados para os modelos.


<img width="900" height="349" alt="image" src="https://github.com/user-attachments/assets/c7fdbcb7-8430-4f28-8bfb-e929679468aa" />

 
A seguir, é criada uma matriz de correlação, que mostra quais variáveis têm mais influência sobre o valor das casas.


<img width="900" height="205" alt="image" src="https://github.com/user-attachments/assets/06c2102c-ed30-48b8-bba6-1bdf9e5372ef" />

 
Aqui os dados são divididos entre treino (80%) e teste (20%) para avaliar o desempenho dos modelos.


<img width="900" height="384" alt="image" src="https://github.com/user-attachments/assets/a22eb35d-a126-45ef-8aed-95abf36d7c62" />

 
O primeiro modelo utilizado é a Regressão Linear, que apresentou um R² de aproximadamente 0.57, explicando parte da variação dos preços.


<img width="900" height="342" alt="image" src="https://github.com/user-attachments/assets/6c9d3033-240b-402a-a688-1301aff48f16" />

 
Depois testamos o XGBoost e uma Rede Neural Artificial, obtendo melhor desempenho — especialmente o XGBoost, com um R² de 0.83.


<img width="900" height="267" alt="image" src="https://github.com/user-attachments/assets/7394a460-0da0-4460-995c-18ed7c321c04" />

 
Aqui está a avaliação final da Rede Neural após treinamento: o RMSE final foi de aproximadamente 0.439 e o R² atingiu cerca de 0.80, mostrando bom desempenho da rede.


<img width="900" height="266" alt="image" src="https://github.com/user-attachments/assets/a786ac46-9d6a-466e-a5a2-22cd1034ffb6" />

 
Pergunta/Resposta: Qual modelo teve menor erro de previsão? Como otimizar ainda mais o desempenho?

- O modelo com menor erro de previsão foi o XGBoost, com RMSE em torno de 0.40 e R² de aproximadamente 0.83.

- Para melhorar ainda mais o desempenho, algumas ações possíveis são:

  • Fazer ajuste de hiperparâmetros (grid search ou randomized search) em modelos como XGBoost e na Rede Neural.

  • Realizar engenharia de features: criar novas variáveis relevantes, tratar outliers e explorar interações entre variáveis.

  • Coletar ou incluir mais dados/contexto (por exemplo, informações sobre zoneamento, proximidade a serviços, etc.).

  • Usar validação cruzada (cross-validation) para ter estimativas mais robustas do desempenho.

  • Experimentar ensembles de modelos e técnicas de regularização para reduzir overfitting.


-------------------------------------------------------------------------------------------------------------------------------------

Questão 7 (Intermediário) - Recomendação de Produtos em um Supermercado

Bibliotecas Utilizadas
- pandas
- mlxtend.frequent_patterns
- numpy
- matplotlib.pyplot
- seaborn
- warnings


<img width="900" height="332" alt="image" src="https://github.com/user-attachments/assets/fa58f6c2-0c43-40c4-9190-3f1a0dd95df6" />

Neste primeiro trecho, importamos as bibliotecas necessárias para análise de dados, visualização e criação das regras de associação. O 'pandas' gerencia os dados em formato de tabela; 'mlxtend' traz o algoritmo Apriori; 'numpy' oferece suporte matemático; 'matplotlib' e 'seaborn' ajudam na visualização; e 'warnings' é usado para suprimir avisos desnecessários.


<img width="900" height="306" alt="image" src="https://github.com/user-attachments/assets/7c99fd0d-2a01-4e92-a4a6-55f5c20e72bb" />


Aqui, o código lê o arquivo CSV com os dados das compras no supermercado, criando um DataFrame. A função df.head(10) mostra as primeiras linhas para termos uma visão inicial dos dados — contendo o número do cliente, a data e o item comprado.

<img width="900" height="348" alt="image" src="https://github.com/user-attachments/assets/7c299ad8-e3ac-425f-ba06-a7dfd9ee2783" />

 
Nesta etapa, removemos a coluna 'Date', pois a data da compra não é relevante para descobrir relações entre produtos. A função df.drop elimina essa coluna do DataFrame.

<img width="900" height="295" alt="image" src="https://github.com/user-attachments/assets/7e03adf6-3536-4ce5-bcdc-159df5526dcd" />

 
Agora, renomeamos as colunas para facilitar o entendimento: 'Member_number' vira 'ID' e 'itemDescription' vira 'Itens'. Isso deixa o dataset mais limpo e intuitivo para análise.

<img width="900" height="313" alt="image" src="https://github.com/user-attachments/assets/35579733-20f0-4c79-8430-ab99fa1ee4ac" />

 
Em seguida, agrupamos os dados por cliente (ID) e unimos todos os itens comprados em uma única linha, separados por vírgula. Assim, temos uma visão do conjunto de produtos adquiridos por cada pessoa, o que é essencial para o algoritmo Apriori identificar padrões de compra.

<img width="900" height="316" alt="image" src="https://github.com/user-attachments/assets/93a25229-96b6-45f1-b6e4-48b89a98abee" />

 
Aqui aplicamos o algoritmo Apriori com suporte mínimo de 2%. Isso significa que só consideramos combinações de produtos que aparecem em pelo menos 2% das transações. Em seguida, geramos as regras de associação com base no 'lift' e ordenamos as regras mais fortes primeiro.

<img width="900" height="245" alt="image" src="https://github.com/user-attachments/assets/d52c99b6-8872-4c4f-81d4-fe22d366ef36" />

 
Por fim, temos a análise dos resultados. A tabela mostra as combinações de produtos (antecedentes e consequentes) e as métricas que indicam a força da relação entre eles, como suporte, confiança e lift. Isso nos permite entender quais produtos tendem a ser comprados juntos.

<img width="900" height="200" alt="image" src="https://github.com/user-attachments/assets/203b2661-8f6e-435e-a99f-1d6f340fdf8b" />

 
Na conclusão textual, respondemos à pergunta sobre as regras mais relevantes e como aplicá-las no mercado. A ideia é que os produtos mais frequentemente comprados juntos sejam colocados próximos nas prateleiras, estimulando vendas cruzadas e aumentando o faturamento.

Conclusão
Neste projeto, aprendemos a aplicar o algoritmo Apriori para identificar padrões de compra em um supermercado. Com isso, é possível recomendar produtos e planejar estratégias de vendas baseadas em dados. O uso de regras de associação mostrou como a análise de dados pode gerar insights práticos para o comércio.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Questão 8 (Avançado) - Recomendação de Filmes com Filtragem Colaborativa


<img width="886" height="360" alt="image" src="https://github.com/user-attachments/assets/fc9890ae-33c7-49a7-9f76-a6ae88406ae0" />

 
Este bloco de código é a Etapa 0: Configuração do Ambiente e Instalação de Bibliotecas.

Aqui, o código usa o comando !pip install para garantir que todas as ferramentas necessárias para o projeto estejam instaladas no ambiente.

•	!pip install mlxtend: Instala a biblioteca mlxtend, que é muito usada para Regras de Associação (como o algoritmo Apriori, que descobre quais itens são frequentemente comprados juntos).

•	!pip install tensorflow, !pip install keras, !pip install tensorflow-recommenders: Instalam o TensorFlow e o Keras, que são as principais bibliotecas para Deep Learning, e o tensorflow-recommenders, um pacote específico para construir sistemas de recomendação avançados.

•	import pandas as pd, import numpy as np: Importa as duas bibliotecas mais fundamentais para manipulação e análise de dados no Python.


<img width="532" height="492" alt="image" src="https://github.com/user-attachments/assets/8cc8a683-9462-477a-bda5-20212a8e5a9c" />

 
Neste bloco, o código está carregando o dataset principal para o sistema de recomendação.

•	O que faz: O comando pd.read_csv('avaliacoes_filmes.csv') lê o arquivo CSV que contém o histórico de avaliações dos usuários e o armazena em uma estrutura de tabela chamada DataFrame, que é nomeada df.

•	Por que: Esta é a etapa inicial e essencial. Precisamos carregar os dados na memória para poder analisá-los e usá-los para treinar o modelo.

•	A Saída: df.head(10) é usado para "espiar" as 10 primeiras linhas do arquivo. Isso confirma que os dados foram carregados corretamente e nos permite ver a estrutura: temos um user_id (o usuário), um movie_id (o filme), um rating (a nota que o usuário deu) e um timestamp (quando a avaliação foi feita).


<img width="759" height="1459" alt="image" src="https://github.com/user-attachments/assets/13714db2-02bf-475f-b8cc-755a94034c0f" />


 
Este bloco de código é o núcleo da Filtragem Colaborativa Clássica. Ele constrói, treina e avalia dois modelos diferentes para comparar suas performances:

1.	Filtragem Colaborativa Baseada no Usuário (User-Based)

2.	Filtragem Colaborativa Baseada no Item (Item-Based)

Tudo isso é feito usando a biblioteca surprise, que é especializada nesse tipo de sistema de recomendação.
________________________________________

Análise Detalhada do Bloco:

1. Preparação dos Dados para a Biblioteca surprise

•	O que faz: Antes de usar o surprise, o DataFrame do Pandas precisa ser convertido.

o	Reader(rating_scale=(1, 5)): Primeiro, é criado um Reader para informar à biblioteca que as notas (ratings) no dataset variam de 1 a 5.

o	Dataset.load_from_df(...): Carrega os dados do DataFrame (df), especificando quais colunas são o user_id, movie_id e rating.

o	train_test_split(...): Divide o conjunto de dados em 75% para treino e 25% para teste (test_size=.25).

•	Por que: O surprise tem seu próprio formato de dados otimizado. A divisão treino/teste é fundamental para avaliar se o modelo é bom em prever notas que ele nunca viu antes.

2. Modelo 1: Filtragem Colaborativa Baseada no Usuário (User-Based)

•	O que faz: Constrói e treina o primeiro modelo.

o	sim_options_user = ...: Define as opções de similaridade.

	'name': 'cosine': Informa que usaremos a "Similaridade de Cossenos" para medir o quão parecidos dois usuários são.

	'user_based': True: Este é o comando principal. Ele diz ao modelo: "Encontre usuários com gostos similares (que deram notas parecidas para os mesmos filmes) e recomende filmes que esses usuários similares gostaram."

o	model_user = KNNBasic(...): Cria o modelo (KNNBasic) com essas opções.

o	model_user.fit(trainset): Treina o modelo usando os dados de treino.

o	accuracy.rmse(predictions_user): Testa o modelo nos dados de teste e calcula as métricas de erro RMSE (Raiz do Erro Quadrático Médio) e MAE (Erro Médio Absoluto).

•	Por que: O objetivo é testar a lógica de "me diga o que pessoas parecidas comigo gostaram".

3. Modelo 2: Filtragem Colaborativa Baseada no Item (Item-Based)

•	O que faz: Constrói e treina o segundo modelo, de forma muito parecida.

o	sim_options_item = ...: Define as opções de similaridade.

	'user_based': False: Esta é a mudança crucial. Agora, o modelo não busca usuários parecidos. Ele busca itens (filmes) parecidos.

o	A lógica agora é: "Se o usuário gostou do filme X, encontre outros filmes (Y e Z) que são 'similares' ao filme X (porque receberam notas parecidas dos mesmos usuários) e recomende-os."

o	O restante (treino e teste) é idêntico.

•	Por que: O objetivo é testar a lógica de "se você gostou disso, talvez goste daquilo".

A Saída (Resultados)

•	O que mostra: A saída imprime o RMSE e o MAE para ambos os modelos.

•	Por que: Essas métricas medem o erro médio das previsões. Quanto menor o número, melhor o modelo.

o	No seu resultado, o modelo Item-Based (RMSE: 1.0876, MAE: 1.0250) teve um erro menor que o User-Based (RMSE: 1.7150, MAE: 1.3333).

o	Conclusão: Para este dataset específico, a abordagem de "recomendar filmes similares" (Item-Based) foi mais precisa do que a de "recomendar o que usuários similares gostam" (User-Based).


<img width="886" height="499" alt="image" src="https://github.com/user-attachments/assets/befcf550-3275-4c01-ab40-f946cea737d4" />


este bloco, o código muda de abordagem. Ele sai da biblioteca surprise e começa a preparar os dados para um modelo de Deep Learning (Rede Neural), que é uma forma mais moderna de sistema de recomendação.

Esta etapa é toda sobre Pré-processamento e Transformação de Dados.

1. Divisão dos Dados (Treino/Teste)

•	O que faz: O train_test_split é usado para dividir o DataFrame original (df) em dois conjuntos: train_data (80% dos dados) e test_data (20% dos dados).

•	Por que: Esta é uma regra fundamental. O modelo só pode "aprender" com os dados de treino. Os dados de teste são guardados "em segredo" e só são usados no final, para verificar se o modelo realmente aprendeu a generalizar ou se apenas "decorou" as respostas do treino.

2. Criação da Matriz Usuário-Item

•	O que faz: O comando .pivot_table() transforma a lista de avaliações em uma matriz (uma grande tabela).

o	As linhas se tornam os user_id.

o	As colunas se tornam os movie_id.

o	Os valores dentro da tabela são os rating (notas).

•	Por que: Redes neurais e outros algoritmos de "Matrix Factorization" (Fatoração Matricial) não trabalham com listas, eles precisam dessa matriz "usuário-item" como entrada. O .fillna(0) é usado para preencher com 0 os filmes que um usuário ainda não avaliou.

3. Normalização dos Dados

•	O que faz: O MinMaxScaler é aplicado na matriz de treino. Ele pega todos os valores (que vão de 0 a 5) e os "comprime" para que fiquem na escala de 0 a 1.

•	Por que: Redes Neurais treinam de forma muito mais rápida, estável e eficiente quando todos os números de entrada estão em uma escala pequena e consistente, como 0 a 1.

A Saída

•	O que mostra: Dimensões da matriz de treino: (5, 4)

•	Por que: Confirma que, após todo o processamento, a matriz de treino (que será usada para alimentar a rede neural) tem 5 usuários (linhas) e 4 filmes (colunas).


<img width="886" height="1103" alt="image" src="https://github.com/user-attachments/assets/21588552-8256-46af-9421-3daf1293ca20" />

 

Este bloco de código é onde o modelo de Deep Learning (Rede Neural) é de fato construído. O modelo usado aqui é um tipo especial chamado Autoencoder.

Um Autoencoder tem duas partes:

1.	Encoder (Codificador): Pega os dados de entrada (a linha de avaliações do usuário) e os comprime em uma representação muito pequena, chamada de "espaço latente" ou "gargalo".

2.	Decoder (Decodificador): Pega essa representação comprimida ("gargalo") e tenta reconstruir a entrada original a partir dela.

Análise Detalhada do Bloco:

1. Arquitetura da Rede

•	n_movies = ...shape[1]: Primeiro, o código verifica quantas colunas (filmes) a matriz de treino tem. A rede neural precisa saber disso.

•	input_layer = Input(...): Define a camada de entrada. Ela tem o mesmo número de neurônios que o número de filmes (neste caso, 4, como visto no passo anterior).

•	Encoder (O Caminho de Compressão):

o	encoded = Dense(128, ...)

o	encoded = Dense(64, ...)

o	latent_view = Dense(32, ...)

o	O que faz: O input_layer com 4 neurônios é progressivamente "espremido" em camadas com 128, 64 e, finalmente, 32 neurônios. Essa camada de 32 neurônios é o "gargalo" (latent view). Ela é forçada a aprender uma representação super comprimida, mas muito rica, dos gostos daquele usuário.

•	Decoder (O Caminho de Reconstrução):

o	decoded = Dense(64, ...)

o	decoded = Dense(128, ...)

o	output_layer = Dense(n_movies, activation='sigmoid')

o	O que faz: O código pega o "gargalo" de 32 neurônios e faz o caminho inverso, "descomprimindo" de volta para 64, 128 e, finalmente, para o número original de filmes (4).

o	Ponto-chave (activation='sigmoid'): A última camada usa a ativação sigmoid. Isso é feito de propósito, porque sigmoid sempre retorna um valor entre 0 e 1. Como normalizamos nossos dados para ficarem entre 0 e 1 no passo anterior, isso ajuda o modelo a gerar previsões na mesma escala.

2. Compilação do Modelo

•	O que faz: O autoencoder.compile(...) prepara o modelo para o treino.

•	Por que:

o	optimizer=Adam(...): Define o otimizador Adam, que é um algoritmo eficiente para ajustar os pesos da rede.

o	loss='mean_squared_error': Esta é a parte mais importante. O "objetivo" do modelo (loss) é o Erro Quadrático Médio. O modelo tentará minimizar a diferença entre a input_layer (as notas reais) e a output_layer (as notas reconstruídas). Basicamente, ele será treinado para ficar muito bom em "adivinhar" as notas que o usuário deu.

A Saída (Model Summary)

•	O que mostra: O .summary() imprime um resumo da arquitetura, mostrando cada camada, seu formato de saída e o número de parâmetros (pesos) que ela precisa aprender.

•	Total params: 21,924: Informa que este modelo tem 21.924 "botões" (parâmetros) que serão ajustados durante o treinamento para minimizar o erro.


 <img width="719" height="1459" alt="image" src="https://github.com/user-attachments/assets/9b0f1d92-4e1a-4cb3-86ad-d4520033c940" />

Este bloco de código é uma versão completa e corrigida dos passos anteriores, juntando a preparação dos dados, a construção do modelo, o treinamento e a avaliação (as partes que faltavam).

________________________________________

Análise Detalhada do Bloco:

1. Divisão e Preparação dos Dados (Revisão Correta)

Este trecho de código refaz a preparação dos dados de forma mais robusta para evitar "vazamento de dados" (data leakage).

•	train_test_split (Feito Primeiro): O DataFrame original é dividido em treino e teste antes de qualquer outra transformação. Isso garante que o modelo não tenha nenhuma informação sobre o conjunto de teste durante o seu treinamento.

•	Criação das Matrizes: As matrizes usuário-item de treino e teste são criadas separadamente a partir dos dados já divididos.

•	Alinhamento (.reindex): Um passo crucial é adicionado. O .reindex força a matriz de teste (user_item_matrix_test_raw) a ter exatamente as mesmas colunas (filmes) e linhas (usuários) que a matriz de treino. Isso evita erros se, por acaso, um filme ou usuário só existir no conjunto de teste.

•	Normalização (MinMaxScaler Corrigida):

o	scaler.fit_transform(user_item_matrix_train): O MinMaxScaler é treinado (fit) e aplicado (transform) apenas nos dados de treino. Ele "aprende" qual é o valor mínimo e máximo (0 e 5) com os dados de treino.

o	scaler.transform(user_item_matrix_test): O scaler (já treinado) é apenas aplicado (transform) nos dados de teste. Isso garante que os dados de teste sejam normalizados usando a mesma regra dos dados de treino, sem "contaminá-los".

2. Definição do Modelo Autoencoder

Esta é a arquitetura da rede neural, similar à do passo anterior.

•	Arquitetura: O modelo é um Autoencoder. Ele possui:

o	Encoder (Codificador): Comprime a entrada (número de filmes) em camadas menores (Dense(64)) até um "gargalo" (latent_view = Dense(32)). Esta camada de 32 neurônios é uma representação compacta dos gostos do usuário.

o	Decoder (Decodificador): Pega o "gargalo" (latent_view) e tenta reconstruir a entrada original, "descomprimindo-a" de volta (Dense(64) e Dense(n_movies)).

•	Compilação: O modelo é compilado com:

o	loss='mean_squared_error': A função de perda que o modelo tentará minimizar (o erro entre a entrada e a saída reconstruída).

o	metrics=['rmse']: Além da perda, pedimos para ele monitorar o RMSE (RootMeanSquaredError), que é uma métrica mais fácil de interpretar por estar na mesma unidade das notas.

3. Treinamento do Modelo

Esta é a etapa onde o modelo "aprende".

•	autoencoder.fit(...): Inicia o treinamento.

o	x=X_train_scaled: Os dados de entrada.

o	y=X_train_scaled: Os dados de "saída esperada". No Autoencoder, a entrada e a saída são as mesmas. O objetivo do modelo é aprender a recriar o seu próprio input da forma mais fiel possível.

o	epochs=100: O modelo verá o dataset de treino 100 vezes.

o	batch_size=32: O modelo treina em "lotes" de 32 usuários por vez.

o	validation_data=(X_test_scaled, X_test_scaled): Ao final de cada época, o modelo tentará prever as notas do conjunto de teste (que ele nunca usou para treinar) e reportará o erro. Isso é vital para monitorar se o modelo está generalizando bem ou apenas "decorando" os dados de treino (overfitting).

4. Avaliação do Desempenho

Após o fim do treinamento, este bloco calcula o desempenho final e "real" do modelo.

•	autoencoder.evaluate(X_test_scaled, X_test_scaled): Roda o modelo uma última vez sobre o conjunto de teste.

•	Resultados: A saída imprime a "Perda (MSE)" e o "RMSE" finais. O RMSE é a métrica principal: ele nos diz, em média, o quão longe as previsões de notas do modelo (a saída reconstruída) ficaram das notas reais do conjunto de teste. Quanto menor o RMSE, melhor o modelo.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Questão 9 (Raio-X com CNNs)

Configuração e Desbalanceamento 

<img width="886" height="552" alt="image" src="https://github.com/user-attachments/assets/3182aee3-c411-421a-a4d5-45f3e44559f1" />

 
•	(Configuração): O primeiro passo foi a organização. Importamos as bibliotecas (TensorFlow, Keras, Pandas) e definimos os caminhos para os dados de treino, teste e validação no Kaggle.

<img width="643" height="630" alt="image" src="https://github.com/user-attachments/assets/36bcefae-523a-4646-826d-e5c4ada80504" />


•	(Correção do Desbalanceamento): Logo de cara, identificamos um problema crítico nos dados: tínhamos muito mais imagens de 'Pneumonia' (3875) do que 'Normais' (1341).

o	Por quê isso é um problema? Se ignorássemos isso, o modelo ficaria 'viciado' em chutar 'Pneumonia' e teria uma péssima performance em identificar pacientes saudáveis.o	Como resolvemos? Calculamos class_weights. Isso dá um 'peso' maior (1.94x) para cada amostra 'Normal', forçando o modelo a prestar mais atenção nela durante o treino."

• Construção e Compilação da CNN 

 <img width="642" height="545" alt="image" src="https://github.com/user-attachments/assets/611c2a61-405a-49e8-8907-fbe4792c2115" />
 
• (Construindo o Modelo): Usamos uma Rede Neural Convolucional (CNN).

o	Como? Empilhamos camadas Conv2D (para achar padrões) e MaxPooling2D (para reduzir a imagem e focar no que é importante).

o	Por quê? Esse processo permite que a rede aprenda características, desde bordas simples até padrões complexos. Também usamos BatchNormalization para estabilizar o treino e Dropout para evitar que o modelo decore as imagens (overfitting).

<img width="490" height="591" alt="image" src="https://github.com/user-attachments/assets/3a8f54a6-430f-4e07-bcfc-9387065c2f2d" />
 

• (Compilando o Modelo): Com a arquitetura pronta, 'compilamos' o modelo.

o	Como? Definimos o otimizador Adam (com uma taxa de aprendizado baixa para um ajuste fino), a função de perda binary_crossentropy (ideal para classificação sim/não) e a métrica accuracy."

• Geradores de Dados e Treinamento

<img width="638" height="605" alt="image" src="https://github.com/user-attachments/assets/ef7d1a72-e30b-4edd-bd72-b0cb4c6f3936" />

 
•	(Preparando Geradores): Usamos o ImageDataGenerator.

o	Por quê? Para aplicar Data Augmentation. O modelo poderia memorizar as poucas imagens de treino.

o	Como? O gerador de treino aplica zoom, rotações e giros aleatórios em tempo real. Isso 'cria' novas imagens e ensina o modelo a generalizar. O gerador de teste não faz isso, apenas redimensiona as imagens.

<img width="856" height="253" alt="image" src="https://github.com/user-attachments/assets/e0c46c2d-2d54-46a6-9a0b-17d8ad197ac0" />


•	(Treinando o Modelo): Aqui, usamos o model.fit().

o	Como? Passamos os geradores, as 25 épocas e, crucialmente, os class_weights que calculamos lá no Bloco 2. Agora o treino é justo."

• Avaliação do Modelo 

<img width="886" height="702" alt="image" src="https://github.com/user-attachments/assets/1b329f89-7542-4a7e-a773-125fd18c76fb" />

 
•	Como? Usamos o classification_report e a confusion_matrix nos dados de teste (que o modelo nunca viu).

•	Por quê? O report nos deu métricas vitais como precisão (quantos 'Pneumonia' que previmos estavam certos) e recall (quantos 'Pneumonia' reais nós conseguimos encontrar). A Matriz de Confusão mostrou visualmente onde o modelo acertou e errou."

--------------------------------------------------------------------------------------------------------------------------------------

Questão 10 (Previsão de Vendas)

• Carga e Exploração 

<img width="886" height="744" alt="image" src="https://github.com/user-attachments/assets/30b49606-b033-4919-9cd1-66667a5b80b5" />

 
•	(Carga): Começamos carregando o store.csv com o Pandas. O df.head() e df.describe() nos mostraram que tínhamos uma mistura de dados numéricos (Gastos) e categóricos (Tipo_Loja)."









• Feature Engineering 

<img width="886" height="360" alt="image" src="https://github.com/user-attachments/assets/3129a588-354b-442f-a93e-cb0407a89918" />

 
•	(Normalização): Vimos que 'Gastos_Publicidade' (milhares) e 'Numero_Funcionarios' (dezenas) tinham escalas muito diferentes.

o	Por quê? Modelos de regressão dão mais importância a números maiores, o que distorce a análise.

o	Como? Usamos MinMaxScaler para colocar todas as colunas numéricas na mesma escala (entre 0 e 1).

<img width="886" height="422" alt="image" src="https://github.com/user-attachments/assets/a44c7f1f-cbda-4d64-ab88-9a94b01fcd98" />

 
•	(Encoding): O modelo não entende o que 'Tipo_Loja_A' significa.

o	Como? Usamos One-Hot Encoding (get_dummies) para transformar colunas de texto em colunas numéricas (0 ou 1)."



• Regressão Linear e Árvore

  <img width="886" height="698" alt="image" src="https://github.com/user-attachments/assets/76955241-4550-4047-9a31-7e6198bff92e" />

  <img width="886" height="528" alt="image" src="https://github.com/user-attachments/assets/354fbc7c-185a-4efb-81db-7abd0b56ac07" />


•	(Split e Regressão Linear): Primeiro, separamos os dados em Treino e Teste (70/30). Em seguida, treinamos nosso primeiro modelo, a LinearRegression.

<img width="886" height="526" alt="image" src="https://github.com/user-attachments/assets/698322c6-b90a-40e6-ad03-4b38b874a65d" />

 
•	(Avaliação): Avaliamos a Regressão Linear com MAE, RMSE e R². 

<img width="886" height="524" alt="image" src="https://github.com/user-attachments/assets/851036ae-b0e7-43ea-9593-f4f0cc0b9176" />

• treinamos o segundo modelo, o DecisionTreeRegressor."

<img width="886" height="552" alt="image" src="https://github.com/user-attachments/assets/28146673-dea7-4115-897a-c8fa22ed52c3" />


• Avaliação e XGBoost 

<img width="886" height="511" alt="image" src="https://github.com/user-attachments/assets/f4886f01-9a3e-4f9d-b4ab-40ffec4a71e3" />

 
•	(Avaliação): Avaliamos a Árvore de Decisão. 

<img width="886" height="511" alt="image" src="https://github.com/user-attachments/assets/33fe28d6-f73f-4d9a-a891-f5748d7fe45e" />

 
•	(Avaliação ): Por fim, treinamos e avaliamos o XGBRegressor, um modelo mais robusto. 

----------

• Aqui nós avaliamos as features de mais importância nos modelos

<img width="886" height="644" alt="image" src="https://github.com/user-attachments/assets/7a40b9c6-734c-4d81-ab47-8b18dceaa2a0" />

<img width="886" height="780" alt="image" src="https://github.com/user-attachments/assets/a0628063-fc7e-4ab7-999c-2359c34ec7e1" />

<img width="886" height="796" alt="image" src="https://github.com/user-attachments/assets/96771a22-db0c-4242-a3f5-84b414c4c458" />




 
 
 


