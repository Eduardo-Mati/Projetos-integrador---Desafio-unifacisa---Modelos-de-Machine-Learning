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



