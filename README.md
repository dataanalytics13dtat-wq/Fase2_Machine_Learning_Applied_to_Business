# Classificando a Qualidade de Vinhos com Machine Learning

Tech Challenge - Fase 2 (POSTECH - DTAT)

## Contexto

A industria vitivinicola avalia a qualidade do vinho tradicionalmente por
analise sensorial de especialistas - um processo subjetivo e demorado. Este
projeto usa dados fisico-quimicos medidos durante a producao (acidez, teor
alcoolico, densidade, niveis de dioxido de enxofre etc.) para treinar
modelos de classificacao que preveem se um vinho e de **Alta Qualidade**
(nota de especialista >= 7) ou **Baixa/Media Qualidade** (nota < 7).

## Dataset

- Fonte: [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) (Kaggle)
- Arquivo: `data/WineQT.csv`
- 1.143 amostras, 11 variaveis fisico-quimicas de entrada + `quality` (nota, 0-10) + `Id`
- Classes: ~86% Baixa/Media Qualidade, ~14% Alta Qualidade (desbalanceado)

## Estrutura do repositorio

```
wine-quality-classification/
|
|-- data/                                   # Base de dados utilizada
|   `-- WineQT.csv
|
|-- notebooks/                              # Notebook com a analise e modelagem completa
|   `-- wine_quality_classification.ipynb
|
|-- src/                                    # Codigo modular (pre-processamento e modelagem)
|   |-- data_prep.py                        # carregamento, target binario, feature engineering, split/scale
|   |-- eda.py                              # graficos de analise exploratoria
|   |-- model_training.py                   # treino da Regressao Logistica e da Random Forest
|   |-- evaluation.py                       # metricas, validacao cruzada, matriz de confusao, interpretacao
|   |-- pipeline.py                         # roda a pipeline completa de ponta a ponta (CLI)
|   `-- wine_quality_classification.py      # versao em script puro, espelhando o notebook
|
|-- results/                                # Graficos e metricas gerados pela pipeline
|
|-- requirements.txt                        # Bibliotecas utilizadas
`-- README.md                               # Este arquivo
```

## Como rodar

1. Instalar as dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Opcao A - Notebook (recomendado para acompanhar a analise passo a passo,
   com graficos e comentarios): abrir `notebooks/wine_quality_classification.ipynb`
   e executar todas as celulas.

3. Opcao B - Pipeline via linha de comando (reproduz os mesmos resultados,
   usando o codigo modular de `src/`):

   ```bash
   python -m src.pipeline
   ```

   (executar a partir da raiz do repositorio, com `data/WineQT.csv` no lugar)

## Pipeline aplicada

1. **Compreensao do problema** - definicao da variavel alvo e da binarizacao (Alta >= 7 / Baixa-Media < 7).
2. **Analise Exploratoria de Dados (EDA)** - distribuicoes, correlacoes com `quality`, outliers (criterio IQR), balanceamento de classes.
3. **Pre-processamento** - checagem de dados faltantes (nenhum encontrado), padronizacao das variaveis, engenharia de 2 features novas (`total_free_sulfur_ratio`, `acid_balance`).
4. **Modelagem** - Regressao Logistica e Random Forest, ambas com `class_weight="balanced"`.
5. **Avaliacao** - split unico de teste (30%) e validacao cruzada estratificada de 5 dobras (referencia mais robusta).
6. **Interpretacao** - coeficientes da Regressao Logistica e importancia de variaveis da Random Forest.

## Principais resultados (validacao cruzada, 5-fold, media)

| Modelo               | Acuracia | Precisao | Recall | F1    | ROC-AUC |
|-----------------------|:--------:|:--------:|:------:|:-----:|:-------:|
| Regressao Logistica  | 0.778    | 0.366    | 0.799  | 0.501 | 0.875   |
| Random Forest         | 0.915    | 0.825    | 0.510  | 0.626 | 0.934   |

A Random Forest erra menos no geral e e mais precisa quando aponta "Alta
Qualidade", mas identifica menos da metade dos vinhos de alta qualidade
existentes (recall de 51%). A Regressao Logistica identifica quase 80%
dos vinhos de alta qualidade, ao custo de mais falsos positivos. A escolha
entre os dois depende de qual erro custa mais caro para o negocio.

## Principais insights (para o processo de producao)

- **Teor alcoolico** e a variavel mais associada a Alta Qualidade nos dois modelos.
- **Acidez volatil** e a variavel mais associada a Baixa/Media Qualidade - e um indicador quimico direto de defeito de producao (contaminacao acetica).
- **Sulfatos** e o equilibrio entre SO2 livre e total tem papel relevante na conservacao.
- O desbalanceamento de classes (~14% de vinhos de alta qualidade) e uma caracteristica real do problema, nao um artefato da amostra - deve ser levado em conta ao definir metas de recall em producao.

## Autoria

Grupo Tech Challenge - Fase 2 - POSTECH DTAT.
