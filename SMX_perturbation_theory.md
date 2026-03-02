# SMX — Spectral Model eXplainability via Perturbação Espectral

## Um Guia Teórico e Didático Abrangente

---

## Sumário

1. [Introdução e Motivação](#1-introdução-e-motivação)
2. [Visão Geral do Pipeline SMX](#2-visão-geral-do-pipeline-smx)
3. [Pré-processamento Espectral](#3-pré-processamento-espectral)
4. [Extração de Zonas Espectrais](#4-extração-de-zonas-espectrais)
5. [Agregação de Zonas Espectrais](#5-agregação-de-zonas-espectrais)
   - 5.1 [Agregadores Simples (Soma, Média, etc.)](#51-agregadores-simples)
   - 5.2 [Agregação por PCA — Fundamentação Teórica](#52-agregação-por-pca--fundamentação-teórica)
6. [Geração de Predicados por Quantis](#6-geração-de-predicados-por-quantis)
7. [Bagging de Predicados](#7-bagging-de-predicados)
8. [Perturbação Espectral — O Coração do Método](#8-perturbação-espectral--o-coração-do-método)
   - 8.1 [Filosofia da Perturbação](#81-filosofia-da-perturbação)
   - 8.2 [Modos de Perturbação](#82-modos-de-perturbação)
   - 8.3 [Quais Amostras São Envolvidas?](#83-quais-amostras-são-envolvidas)
   - 8.4 [Métricas de Impacto — Regressão](#84-métricas-de-impacto--regressão)
   - 8.5 [Métricas de Impacto — Classificação](#85-métricas-de-impacto--classificação)
   - 8.6 [Algoritmo Completo: calculate_predicate_perturbation](#86-algoritmo-completo-calculate_predicate_perturbation)
9. [Construção do Grafo de Predicados](#9-construção-do-grafo-de-predicados)
10. [Centralidade Local de Alcance (LRC)](#10-centralidade-local-de-alcance-lrc)
11. [Agregação Multi-Semente](#11-agregação-multi-semente)
12. [Mapeamento de Thresholds para o Espaço Natural](#12-mapeamento-de-thresholds-para-o-espaço-natural)
13. [Reconstrução do Threshold Multivariado](#13-reconstrução-do-threshold-multivariado--do-score-ao-espectro)
14. [Exemplo Completo: Do Espectro à Explicação](#14-exemplo-completo-do-espectro-à-explicação)
15. [Comparação com Outros Métodos de Explicabilidade](#15-comparação-com-outros-métodos-de-explicabilidade)
16. [Parâmetros Recomendados](#16-parâmetros-recomendados)
17. [Conclusão](#17-conclusão)

---

## 1. Introdução e Motivação

Modelos de aprendizado de máquina aplicados a dados espectrais (como espectroscopia de Fluorescência de Raios X — XRF) frequentemente funcionam como "caixas pretas": aprendem relações complexas entre centenas ou milhares de variáveis espectrais e uma resposta de interesse (classe ou valor contínuo), mas não revelam explicitamente **quais regiões do espectro são mais importantes** para as previsões.

O método **SMX (Spectral Model eXplainability)** propõe uma abordagem original para extrair explicações de modelos espectrais. O SMX combina três pilares fundamentais:

1. **Zonas espectrais** — O espectro contínuo é particionado em zonas com significado químico/físico (e.g., picos de emissão de Ca, Fe, Ti em XRF).
2. **Discretização por quantis** — Os valores agregados de cada zona são discretizados em predicados lógicos do tipo "Zona $\leq \tau$" ou "Zona $> \tau$", onde $\tau$ é um limiar derivado de quantis.
3. **Teoria de grafos** — Os predicados são organizados em um grafo dirigido, e a centralidade de cada nó (predicado) quantifica sua importância global.

O foco deste documento é a variante do SMX que utiliza **perturbação espectral** como mecanismo para avaliar o impacto de cada predicado. Essa abordagem é uma alternativa à covariância e à informação mútua, oferecendo uma medida causal mais direta de importância.

---

## 2. Visão Geral do Pipeline SMX

O pipeline completo do SMX com perturbação segue as seguintes etapas:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE SMX (Perturbação)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. DADOS BRUTOS  ──►  2. PRÉ-PROCESSAMENTO (Poisson + MC)             │
│                              │                                          │
│  3. TREINAMENTO DO MODELO (PLS-DA / SVM / MLP)                         │
│                              │                                          │
│  4. DEFINIÇÃO DAS ZONAS ESPECTRAIS (conhecimento do especialista)       │
│                              │                                          │
│  5. EXTRAÇÃO DAS ZONAS  ──►  6. AGREGAÇÃO POR PCA (scores PC1)         │
│                              │                                          │
│  7. GERAÇÃO DE PREDICADOS (quartis: 0.2, 0.4, 0.6, 0.8)               │
│                              │                                          │
│  8. BAGGING DE PREDICADOS (10 bags × 4 sementes)                        │
│                              │                                          │
│  9. PERTURBAÇÃO ESPECTRAL ──► cálculo do impacto por predicado          │
│                              │                                          │
│ 10. CONSTRUÇÃO DO GRAFO DIRIGIDO                                        │
│                              │                                          │
│ 11. LRC (Local Reaching Centrality) ──► ranking de predicados           │
│                              │                                          │
│ 12. MAPEAMENTO PARA ESPAÇO NATURAL                                      │
│                              │                                          │
│ 13. RECONSTRUÇÃO DO THRESHOLD MULTIVARIADO                              │
│                              │                                          │
│ 14. VISUALIZAÇÃO E INTERPRETAÇÃO                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pré-processamento Espectral

Antes de aplicar o SMX, os dados espectrais são pré-processados. O pré-processamento padrão utilizado nos experimentos é o **Escalonamento de Poisson com Centragem na Média (MC)**.

### 3.1 Escalonamento de Poisson

Para dados de contagem (como XRF), a variância de cada variável $x_j$ é proporcional à sua média $\bar{x}_j$. O escalonamento de Poisson corrige essa heteroscedasticidade:

$$
x_{ij}^{\text{Poisson}} = \frac{x_{ij}}{\sqrt{\bar{x}_j}}
$$

onde $\bar{x}_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij}$ é a média da variável $j$ no conjunto de calibração.

### 3.2 Centragem na Média (Mean Centering)

Após o escalonamento de Poisson, subtrai-se a média de cada variável:

$$
x_{ij}^{\text{prep}} = x_{ij}^{\text{Poisson}} - \overline{x_j^{\text{Poisson}}}
$$

O resultado é uma matriz $\mathbf{X}^{\text{prep}} \in \mathbb{R}^{n \times p}$ onde $n$ é o número de amostras e $p$ é o número de variáveis espectrais.

> **Importante:** O pré-processamento é ajustado (fit) apenas no conjunto de calibração. Para o conjunto de predição, os mesmos parâmetros (média e escala) são aplicados sem reajuste, garantindo que não há vazamento de informação.

---

## 4. Extração de Zonas Espectrais

### 4.1 Conceito

O espectro contínuo $\mathbf{x}_i = [x_{i,\lambda_1}, x_{i,\lambda_2}, \ldots, x_{i,\lambda_p}]$ é particionado em $M$ zonas espectrais, cada uma correspondendo a um intervalo de energia (ou comprimento de onda) com significado físico-químico.

Formalmente, uma zona $Z_m$ é definida como:

$$
Z_m = \{j : \lambda_{\text{start}}^{(m)} \leq \lambda_j \leq \lambda_{\text{end}}^{(m)}\}
$$

onde $\lambda_j$ é o comprimento de onda (ou energia) da $j$-ésima coluna.

### 4.2 Definição pelo Especialista

As zonas são definidas com base no conhecimento do especialista sobre o fenômeno em estudo. Para dados de XRF, cada zona tipicamente corresponde a uma linha de emissão elementar:

```python
spectral_cuts = [
    ('Ca ka', 3.50, 3.91),   # Linha K-alfa do Cálcio
    ('Fe ka', 6.15, 6.76),   # Linha K-alfa do Ferro
    ('Zn ka', 8.29, 8.80),   # Linha K-alfa do Zinco
    ...
]
```

### 4.3 Formalização

Dado o DataFrame $\mathbf{X} \in \mathbb{R}^{n \times p}$, a extração produz um dicionário de sub-matrizes:

$$
\mathbf{X}_{Z_m} \in \mathbb{R}^{n \times d_m}, \quad m = 1, \ldots, M
$$

onde $d_m = |Z_m|$ é o número de variáveis na zona $m$. Note que $\sum_{m=1}^{M} d_m \leq p$, pois pode existir partes do espectro não cobertas por nenhuma zona.

---

## 5. Agregação de Zonas Espectrais

Cada zona $Z_m$ contém $d_m$ variáveis espectrais. Para gerar predicados, precisamos resumir cada zona em um **único valor escalar** por amostra. Essa etapa é denominada **agregação**.

### 5.1 Agregadores Simples

Seja $\mathbf{x}_{i}^{(m)} = [x_{i,j_1}, x_{i,j_2}, \ldots, x_{i,j_{d_m}}]$ o vetor de valores da zona $m$ para a amostra $i$. Os agregadores simples disponíveis são:

| Agregador | Fórmula | Descrição |
|-----------|---------|-----------|
| **sum** | $s_i^{(m)} = \sum_{k=1}^{d_m} x_{i,j_k}$ | Soma dos valores da zona |
| **mean** | $s_i^{(m)} = \frac{1}{d_m}\sum_{k=1}^{d_m} x_{i,j_k}$ | Média aritmética |
| **median** | $s_i^{(m)} = \text{med}(x_{i,j_1}, \ldots, x_{i,j_{d_m}})$ | Mediana |
| **max** | $s_i^{(m)} = \max_k x_{i,j_k}$ | Valor máximo |
| **min** | $s_i^{(m)} = \min_k x_{i,j_k}$ | Valor mínimo |
| **std** | $s_i^{(m)} = \sqrt{\frac{1}{d_m-1}\sum_{k=1}^{d_m}(x_{i,j_k} - \bar{x}_i^{(m)})^2}$ | Desvio padrão |
| **var** | $s_i^{(m)} = \frac{1}{d_m-1}\sum_{k=1}^{d_m}(x_{i,j_k} - \bar{x}_i^{(m)})^2$ | Variância |
| **extreme** | $s_i^{(m)} = x_{i,j^*}$ onde $j^* = \arg\max_k |x_{i,j_k}|$ | Valor de maior magnitude |

O resultado é uma matriz $\mathbf{S} \in \mathbb{R}^{n \times M}$, onde cada coluna $m$ contém os valores agregados da zona $Z_m$.

### 5.2 Agregação por PCA — Fundamentação Teórica

A agregação por PCA é a abordagem **recomendada e utilizada em todos os experimentos** do SMX. Ela captura a direção de máxima variância dentro de cada zona, produzindo uma representação mais rica que os agregadores escalares simples.

#### 5.2.1 Fundamentação Matemática da PCA

Para cada zona $Z_m$, considere a sub-matriz $\mathbf{X}_{Z_m} \in \mathbb{R}^{n \times d_m}$. A PCA com 1 componente realiza os seguintes passos:

**Passo 1 — Centralização:**

$$
\tilde{\mathbf{X}}_{Z_m} = \mathbf{X}_{Z_m} - \mathbf{1}_n \cdot \bar{\mathbf{x}}_{Z_m}^T
$$

onde $\bar{\mathbf{x}}_{Z_m} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i^{(m)} \in \mathbb{R}^{d_m}$ é o vetor de médias da zona.

**Passo 2 — Decomposição em Valores Singulares (ou autovalores):**

A PCA encontra o vetor de pesos $\mathbf{w}_1^{(m)} \in \mathbb{R}^{d_m}$ que maximiza a variância projetada:

$$
\mathbf{w}_1^{(m)} = \arg\max_{\|\mathbf{w}\|=1} \text{Var}\left(\tilde{\mathbf{X}}_{Z_m} \mathbf{w}\right)
$$

Equivalentemente, $\mathbf{w}_1^{(m)}$ é o autovetor associado ao maior autovalor da matriz de covariância $\hat{\boldsymbol{\Sigma}} = \frac{1}{n-1}\tilde{\mathbf{X}}_{Z_m}^T \tilde{\mathbf{X}}_{Z_m}$.

**Passo 3 — Projeção (Scores):**

$$
t_i^{(m)} = \left(\mathbf{x}_i^{(m)} - \bar{\mathbf{x}}_{Z_m}\right)^T \mathbf{w}_1^{(m)}
$$

O score $t_i^{(m)}$ é a projeção da amostra $i$ na direção de máxima variância da zona $m$.

**Passo 4 — Variância Explicada:**

$$
\text{VE}^{(m)} = \frac{\sigma_1^2}{\sum_{k=1}^{d_m} \sigma_k^2}
$$

onde $\sigma_1^2$ é o maior autovalor. Essa fração indica quanto da variância total da zona é capturada pelo PC1.

#### 5.2.2 Resultado da Agregação PCA

O resultado é um DataFrame $\mathbf{T} \in \mathbb{R}^{n \times M}$ de scores e um dicionário de metadados PCA contendo, para cada zona $m$:

| Componente | Símbolo | Dimensão | Descrição |
|------------|---------|----------|-----------|
| **Loadings** | $\mathbf{w}_1^{(m)}$ | $d_m$ | Pesos de cada variável no PC1 |
| **Média** | $\bar{\mathbf{x}}_{Z_m}$ | $d_m$ | Vetor de médias da zona |
| **Variância Explicada** | $\text{VE}^{(m)}$ | escalar | Fração da variância capturada |
| **Colunas** | — | $d_m$ | Nomes das variáveis originais |

#### 5.2.3 Por que PCA e não Soma?

A agregação por PCA oferece vantagens fundamentais:

1. **Direcionalidade**: O PC1 captura a direção de **máxima diferenciação** entre amostras, ponderando as variáveis de forma ótima.
2. **Invertibilidade**: A transformação PCA é invertível — é possível **reconstruir o limiar (threshold) no espaço espectral original**, gerando um "espectro limiar" (threshold spectrum). Isso é impossível com agregadores simples como soma ou média.
3. **Informação de qualidade**: A variância explicada $\text{VE}^{(m)}$ quantifica quão bem o score resume a informação da zona, permitindo ponderar a importância pela qualidade da representação.

---

## 6. Geração de Predicados por Quantis

### 6.1 Conceito

Um **predicado** é uma proposição lógica que particiona as amostras em dois grupos: as que satisfazem a condição e as que não satisfazem. No SMX, os predicados são gerados a partir dos valores agregados das zonas espectrais usando quantis como limiares.

### 6.2 Definição Formal

Dado o DataFrame de scores agregados $\mathbf{T} \in \mathbb{R}^{n \times M}$ e um conjunto de quantis $\mathcal{Q} = \{q_1, q_2, \ldots, q_K\}$ (e.g., $\{0.2, 0.4, 0.6, 0.8\}$), para cada zona $m$ e cada quantil $q_k$:

1. Calcula-se o valor do quantil:

$$
\tau_{m,k} = Q_{q_k}(t_1^{(m)}, t_2^{(m)}, \ldots, t_n^{(m)})
$$

onde $Q_{q_k}(\cdot)$ é a função quantil empírica de nível $q_k$.

2. Geram-se **dois predicados** complementares:

$$
P_{m,k}^{\leq}: \quad t_i^{(m)} \leq \tau_{m,k}
$$

$$
P_{m,k}^{>}: \quad t_i^{(m)} > \tau_{m,k}
$$

### 6.3 Número Total de Predicados

Para $M$ zonas e $K$ quantis, o número máximo de predicados é:

$$
N_P = 2 \times M \times K
$$

Na prática, predicados duplicados (gerados quando diferentes zonas possuem o mesmo valor de quantil) são removidos, resultando em $N_P' \leq N_P$ predicados únicos.

### 6.4 Matriz Indicadora de Predicados

A matriz indicadora $\mathbf{I} \in \{0, 1\}^{n \times N_P'}$ é construída onde:

$$
I_{i,j} = \begin{cases} 1 & \text{se a amostra } i \text{ satisfaz o predicado } j \\ 0 & \text{caso contrário} \end{cases}
$$

Essa matriz indica, para cada par (amostra, predicado), se a condição lógica é cumprida.

### 6.5 Exemplo Concreto

Suponha a zona "Ca ka" com 4 quantis $\{0.2, 0.4, 0.6, 0.8\}$:

| Quantil | $\tau$ | Predicado $\leq$ | Predicado $>$ |
|---------|--------|-------------------|---------------|
| 0.2 | -3.42 | Ca ka $\leq$ -3.42 | Ca ka $>$ -3.42 |
| 0.4 | -1.15 | Ca ka $\leq$ -1.15 | Ca ka $>$ -1.15 |
| 0.6 | 0.87 | Ca ka $\leq$ 0.87 | Ca ka $>$ 0.87 |
| 0.8 | 2.56 | Ca ka $\leq$ 2.56 | Ca ka $>$ 2.56 |

Cada predicado define um subconjunto de amostras. Por exemplo, "Ca ka $\leq$ -3.42" seleciona as ~20% amostras com menor valor agregado na zona do Cálcio.

---

## 7. Bagging de Predicados

### 7.1 Motivação

O bagging (Bootstrap AGGregatING) é utilizado para:
- **Reduzir a variância** do ranking de importância dos predicados
- **Aumentar a robustez** contra outliers e flutuações amostrais
- **Criar diversidade** na avaliação dos predicados

### 7.2 Estratégia de Amostragem

O bagging no SMX opera em **duas dimensões** independentes:

1. **Amostragem de linhas (amostras)**: Controlada por `sample_bagging`
2. **Amostragem de colunas (predicados)**: Controlada por `predicate_bagging`

Nos experimentos do SMX, a configuração padrão é:
- `sample_bagging=True`: Sorteia-se um subconjunto de amostras para cada bag
- `predicate_bagging=False`: Todos os predicados são usados em todos os bags

### 7.3 Algoritmo

Para cada bag $b = 1, \ldots, B$:

1. **Seleção de amostras**: Sorteia-se um subconjunto $\mathcal{S}_b \subset \{1, \ldots, n\}$ de tamanho $n_b$ (tipicamente 80% do total), **sem reposição** (subamostragem).

2. **Seleção de predicados**: Utiliza-se todos os $N_P'$ predicados (quando `predicate_bagging=False`).

3. **Filtragem**: Para cada predicado $P_j$, filtra-se os índices de amostras em $\mathcal{S}_b$ que satisfazem $P_j$:

$$
\mathcal{S}_b^{(j)} = \{i \in \mathcal{S}_b : I_{i,j} = 1\}
$$

4. **Validação**: Se $|\mathcal{S}_b^{(j)}| < n_{\min}$ (e.g., 20% do total), o predicado é descartado deste bag por cobertura insuficiente.

5. **Armazenamento**: Para cada predicado válido, armazena-se um DataFrame com:
   - `Zone_Sum`: valor agregado da zona para as amostras satisfeitas
   - `Predicted_Y`: predição do modelo para essas amostras
   - `Sample_Index`: índices originais das amostras

### 7.4 Multi-Semente

O procedimento completo de bagging é repetido com diferentes sementes aleatórias (tipicamente $\{0, 1, 2, 3\}$), e os resultados são agregados posteriormente. Isso confere robustez adicional ao ranking final.

### 7.5 Parâmetros Típicos

| Parâmetro | Valor Padrão | Descrição |
|-----------|-------------|-----------|
| `n_bags` | 10 | Número de bags por semente |
| `n_samples_per_bag` | $0.8 \times n$ | Amostras por bag |
| `min_samples_per_predicate` | $0.2 \times n$ | Mínimo de amostras para validação |
| `replace` | `False` | Subamostragem (sem reposição) |
| `sample_bagging` | `True` | Amostragem de linhas ativada |
| `predicate_bagging` | `False` | Todos os predicados em todos os bags |

---

## 8. Perturbação Espectral — O Coração do Método

### 8.1 Filosofia da Perturbação

A ideia central da perturbação espectral é simples e poderosa:

> **Se uma zona espectral é importante para a previsão do modelo, então "destruir" (perturbar) a informação nessa zona deve causar uma mudança significativa nas previsões.**

Diferentemente da covariância ou informação mútua (que medem associação estatística entre o valor agregado da zona e a previsão), a perturbação mede diretamente o **impacto causal** da informação espectral sobre o modelo. Essa abordagem é análoga ao conceito de *feature importance by permutation*, mas adaptada ao contexto espectral com duas diferenças fundamentais:

1. **Perturbação por substituição** ao invés de permutação: os valores da zona são substituídos por um valor fixo ou uma estatística (mediana, média, etc.), em vez de serem permutados entre amostras.
2. **Granularidade de predicado**: a importância é calculada não para a zona inteira, mas para cada predicado — ou seja, considerando apenas o subconjunto de amostras que satisfaz a condição do predicado.

### 8.2 Modos de Perturbação

O SMX oferece cinco modos de perturbação, controlados pelo parâmetro `perturbation_mode`:

#### 8.2.1 Modo `constant` (Constante)

Todas as variáveis da zona são substituídas por um valor fixo $c$:

$$
\tilde{x}_{i,j} = c, \quad \forall j \in Z_m
$$

Tipicamente $c = 0$, o que efetivamente "zera" a informação da zona.

#### 8.2.2 Modo `mean` (Média)

Cada variável da zona é substituída pela sua média:

$$
\tilde{x}_{i,j} = \bar{x}_j = \frac{1}{n_{\text{src}}}\sum_{k \in \mathcal{S}_{\text{src}}} x_{k,j}, \quad \forall j \in Z_m
$$

onde $\mathcal{S}_{\text{src}}$ é o conjunto fonte para cálculo da estatística (ver Seção 8.2.6).

#### 8.2.3 Modo `median` (Mediana) ⭐ *Recomendado*

Cada variável da zona é substituída pela sua mediana:

$$
\tilde{x}_{i,j} = \text{med}_j = \text{mediana}\{x_{k,j} : k \in \mathcal{S}_{\text{src}}\}, \quad \forall j \in Z_m
$$

Este é o modo utilizado em **todos os experimentos** do SMX. A mediana é preferida à média por ser mais robusta a outliers.

#### 8.2.4 Modo `min` (Mínimo)

$$
\tilde{x}_{i,j} = \min_{k \in \mathcal{S}_{\text{src}}} x_{k,j}, \quad \forall j \in Z_m
$$

#### 8.2.5 Modo `max` (Máximo)

$$
\tilde{x}_{i,j} = \max_{k \in \mathcal{S}_{\text{src}}} x_{k,j}, \quad \forall j \in Z_m
$$

#### 8.2.6 Fonte das Estatísticas (`stats_source`)

Quando o modo de perturbação é baseado em uma estatística (média, mediana, min, max), é necessário definir **qual conjunto de amostras** é usado para calcular essa estatística:

- `stats_source='full'` ⭐: Usa **todo o conjunto de calibração** $\mathbf{X}^{\text{prep}}$ para calcular a estatística. Esse é o padrão e a abordagem usada nos experimentos.
- `stats_source='predicate'`: Usa apenas as **amostras que satisfazem o predicado** atual.

A diferença é sutil, porém importante. Com `'full'`, a perturbação substitui os valores da zona pela estatística da população geral — uma perturbação em direção ao "comportamento médio". Com `'predicate'`, a perturbação é mais localizada.

### 8.3 Quais Amostras São Envolvidas?

Um ponto crucial da perturbação no SMX é que ela **não opera sobre todas as amostras do dataset** de uma vez. O processo envolve várias camadas de filtragem:

#### Camada 1 — Bagging (Seleção de Amostras por Bag)

Para cada bag $b$, um subconjunto $\mathcal{S}_b$ de amostras é selecionado (80% do total, sem reposição).

#### Camada 2 — Predicado (Seleção por Condição Lógica)

Dentro do bag $b$, para cada predicado $P_j$, apenas as amostras que **satisfazem** o predicado são utilizadas:

$$
\mathcal{S}_{b}^{(j)} = \{i \in \mathcal{S}_b : t_i^{(m)} \leq \tau \text{ (ou } > \tau \text{)}\}
$$

#### Camada 3 — Perturbação e Previsão

A perturbação é aplicada **apenas às colunas espectrais da zona relevante**, mantendo intactas todas as demais variáveis:

$$
\tilde{\mathbf{x}}_i = \begin{cases}
\text{estatística}(x_{\cdot,j}) & \text{se } j \in Z_m \\
x_{i,j} & \text{se } j \notin Z_m
\end{cases}
$$

Isso garante que estamos medindo o impacto da zona $Z_m$ de forma isolada.

#### Resumo Visual

```
Todo o dataset de calibração (n amostras)
    │
    ├── Bag 1: subconjunto S₁ (~80% das amostras)
    │   │
    │   ├── Predicado "Ca ka ≤ -1.15": amostras de S₁ satisfazendo a condição
    │   │   └── Perturbação: zerar/mediana a zona "Ca ka" → prever → medir impacto
    │   │
    │   ├── Predicado "Fe ka > 3.22": amostras de S₁ satisfazendo a condição
    │   │   └── Perturbação: zerar/mediana a zona "Fe ka" → prever → medir impacto
    │   │
    │   └── ...
    │
    ├── Bag 2: subconjunto S₂ (~80% das amostras)
    │   └── ...
    │
    └── ...
```

### 8.4 Métricas de Impacto — Regressão

Quando o modelo é tratado como regressor (e.g., PLS-DA com saída contínua entre 0 e 1, ou PLS-R), utilizam-se métricas contínuas para quantificar o impacto da perturbação. Seja $\hat{y}_i$ a previsão original e $\hat{y}_i^{\text{pert}}$ a previsão após perturbação da zona.

#### 8.4.1 `mean_abs_diff` — Diferença Absoluta Média ⭐ *Padrão para regressão*

$$
\text{Imp}(P_j) = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} |\hat{y}_i - \hat{y}_i^{\text{pert}}|
$$

**Interpretação**: Mede a magnitude média da mudança na previsão. Valores altos indicam que a zona é importante para o modelo. Essa métrica ignora a direção da mudança.

**Propriedades**:
- Sempre $\geq 0$
- Insensível à direção da perturbação
- Escala: mesma unidade da variável resposta

#### 8.4.2 `mean_diff` — Diferença Média (com sinal)

$$
\text{Imp}(P_j) = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} (\hat{y}_i - \hat{y}_i^{\text{pert}})
$$

**Interpretação**: Se positivo, a perturbação tende a **reduzir** as previsões; se negativo, tende a **aumentá-las**. Permite identificar a **direção** do impacto.

**Nota**: Para fins de ranking, utiliza-se $|\text{Imp}(P_j)|$.

#### 8.4.3 `mean_relative_dev` — Desvio Relativo Médio

$$
\text{Imp}(P_j) = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} \frac{\hat{y}_i^{\text{pert}} - \hat{y}_i}{\hat{y}_i}
$$

**Interpretação**: Expressa a mudança como fração da previsão original. Útil quando a escala absoluta das previsões varia significativamente.

**Cuidado**: Pode gerar instabilidades quando $\hat{y}_i \approx 0$. Valores nulos são tratados como `NaN` e excluídos do cálculo.

### 8.5 Métricas de Impacto — Classificação

Quando o modelo é utilizado como classificador (e.g., MLP, SVM), métricas específicas para classificação estão disponíveis.

#### 8.5.1 `prediction_change_rate` — Taxa de Mudança de Predição

$$
\text{Imp}(P_j) = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} \mathbb{1}[\hat{c}_i \neq \hat{c}_i^{\text{pert}}]
$$

onde $\hat{c}_i$ e $\hat{c}_i^{\text{pert}}$ são as classes preditas antes e após a perturbação, e $\mathbb{1}[\cdot]$ é a função indicadora.

**Interpretação**: Proporção de amostras que mudaram de classe. Varia de 0 (nenhuma mudou) a 1 (todas mudaram). **Não requer rótulos verdadeiros.**

#### 8.5.2 `accuracy_drop` — Queda de Acurácia

$$
\text{Imp}(P_j) = \text{Acc}(\hat{\mathbf{c}}, \mathbf{y}) - \text{Acc}(\hat{\mathbf{c}}^{\text{pert}}, \mathbf{y})
$$

**Interpretação**: Diferença na acurácia antes e após perturbação. Valores positivos indicam degradação. **Requer rótulos verdadeiros ($\mathbf{y}$).**

#### 8.5.3 `f1_drop` — Queda do F1-Score

$$
\text{Imp}(P_j) = F_1(\hat{\mathbf{c}}, \mathbf{y}) - F_1(\hat{\mathbf{c}}^{\text{pert}}, \mathbf{y})
$$

**Interpretação**: Análogo à queda de acurácia, mas usando F1-score (média ponderada para suporte de classes desbalanceadas). **Requer rótulos verdadeiros.**

#### 8.5.4 `probability_shift` — Deslocamento de Probabilidade ⭐ *Padrão para classificação*

$$
\text{Imp}(P_j) = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} \frac{1}{2}\sum_{c=1}^{C} |p_{i,c} - p_{i,c}^{\text{pert}}|
$$

onde $p_{i,c} = P(\hat{c}_i = c | \mathbf{x}_i)$ são as probabilidades preditas pelo modelo e $C$ é o número de classes.

**Interpretação**: Mede quanto as probabilidades de classe mudam em média após a perturbação. A divisão por 2 normaliza para o fato de que, em classificação binária, se a probabilidade de uma classe aumenta em $\delta$, a da outra diminui em $\delta$.

**Requer**: Modelo com método `predict_proba()` (e.g., SVC com `probability=True`, MLP, RandomForest).

**Propriedades**:
- Varia de 0 a 1
- Mais sensível que `prediction_change_rate` (detecta mudanças graduais nas probabilidades, mesmo que a classe predita não mude)
- Não requer rótulos verdadeiros

#### 8.5.5 `decision_function_shift` — Deslocamento da Função de Decisão

$$
\text{Imp}(P_j) = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} |d(\mathbf{x}_i) - d(\tilde{\mathbf{x}}_i)|
$$

onde $d(\mathbf{x})$ é a função de decisão do modelo (e.g., distância ao hiperplano para SVM).

**Requer**: Modelo com método `decision_function()` (e.g., SVC, LinearSVC, LogisticRegression).

### 8.6 Algoritmo Completo: `calculate_predicate_perturbation`

O algoritmo completo da perturbação espectral por predicado é:

```
ALGORITMO: Perturbação Espectral por Predicado
═══════════════════════════════════════════════

ENTRADA:
  - estimator: modelo treinado
  - X_prep: dados pré-processados (n × p)
  - folds_struct: estrutura de bags {Bag_k: {regra: DataFrame}}
  - predicates_df: DataFrame de predicados
  - spectral_cuts: definição das zonas [(nome, início, fim)]
  - aim: 'regression' ou 'classification'
  - perturbation_mode: 'constant', 'mean', 'median', 'min', 'max'
  - metric: métrica de impacto selecionada

SAÍDA:
  - metrics_dict: {Bag_k: DataFrame(Predicate, Perturbation)}

PARA CADA bag (fold) na estrutura:
  PARA CADA predicado P_j no bag:

    1. Extrair índices das amostras que satisfazem P_j
       S_j ← {i : amostra i satisfaz P_j no bag}

    2. Identificar a zona espectral Z_m associada a P_j
       zone_cols ← colunas de X_prep pertencentes a Z_m

    3. Construir subconjunto de dados original
       X_eval ← X_prep[S_j, :]  (apenas amostras do predicado)

    4. Aplicar perturbação
       X_pert ← X_eval.copy()
       SE perturbation_mode == 'constant':
         X_pert[:, zone_cols] ← perturbation_value
       SENÃO:
         stats ← calcular_estatística(X_fonte[:, zone_cols])
         X_pert[:, zone_cols] ← stats

    5. Obter previsões originais e perturbadas
       y_orig ← estimator.predict(X_eval)
       y_pert ← estimator.predict(X_pert)

    6. Calcular importância segundo a métrica selecionada
       imp ← calcular_métrica(y_orig, y_pert, y_true, metric)

    7. Armazenar importância
       fold_metrics[P_j] ← imp

  FIM PARA
  metrics_dict[bag] ← DataFrame(fold_metrics)
FIM PARA

RETORNAR metrics_dict
```

### 8.7 Compatibilidade de Métricas com Modelos

| Modelo | `prediction_change_rate` | `accuracy_drop` | `probability_shift` | `decision_function_shift` |
|--------|:------------------------:|:----------------:|:--------------------:|:------------------------:|
| SVC | ✓ | ✓ | ✓ (`probability=True`) | ✓ |
| LinearSVC | ✓ | ✓ | ✗ | ✓ |
| RandomForest | ✓ | ✓ | ✓ | ✗ |
| LogisticRegression | ✓ | ✓ | ✓ | ✓ |
| KNeighbors | ✓ | ✓ | ✓ | ✗ |
| PLSRegression | regression metrics | regression metrics | ✗ | ✗ |
| MLP | ✓ | ✓ | ✓ | ✗ |

---

## 9. Construção do Grafo de Predicados

### 9.1 Motivação

As importâncias calculadas pela perturbação (ou covariância/informação mútua) são **locais** — atribuídas a cada predicado individualmente dentro de cada bag. Para obter uma visão **global** da importância, o SMX constrói um **grafo dirigido** que codifica as relações entre predicados e as conecta a nós terminais de classe.

### 9.2 Estrutura do Grafo

O grafo $G = (V, E)$ é um **dígrafo ponderado** com:

- **Nós terminais**: `Class_A` e `Class_B` (as classes do problema)
- **Nós de predicado**: cada predicado válido que aparece em pelo menos um bag
- **Arestas dirigidas**: conexões entre predicados consecutivos, ponderadas pela importância

### 9.3 Algoritmo de Construção

Para cada bag $b$:

1. **Ordenação**: Os predicados do bag são ordenados em ordem **decrescente** de importância (valor da métrica de perturbação).

2. **Arestas entre consecutivos**: São criadas arestas direcionadas entre predicados consecutivos na ordenação:

$$
P_1^{(b)} \xrightarrow{w_1} P_2^{(b)} \xrightarrow{w_2} P_3^{(b)} \xrightarrow{w_3} \ldots
$$

onde $w_i$ é o valor da métrica de perturbação do predicado **de origem** (source).

3. **Ponderação pela Variância Explicada** (opcional, mas ativado por padrão): Os pesos das arestas são multiplicados pela variância explicada pelo PC1 da zona correspondente ao predicado de origem:

$$
w_i^{\text{ponderado}} = w_i \times \text{VE}^{(m_i)}
$$

Isso faz com que zonas cujo PC1 captura menor parcela da variância tenham seus pesos reduzidos — uma penalização pela qualidade da representação.

4. **Conexão terminal**: O último predicado da sequência é conectado ao nó terminal correspondente à **classe majoritária** entre as amostras que satisfazem aquele predicado:

$$
P_{\text{last}}^{(b)} \xrightarrow{w_{\text{last}}} \text{Class}_{c^*}
$$

onde $c^* = \arg\max_c \sum_{i \in \mathcal{S}} \mathbb{1}[\hat{c}_i = c]$.

5. **Acumulação**: Se uma aresta $(u, v)$ já existe no grafo (de bags anteriores), o peso é **acumulado** (somado):

$$
w(u, v) \leftarrow w(u, v) + w_{\text{novo}}
$$

### 9.4 Resolução de Arestas Bidirecionais

Após processar todos os bags, podem existir arestas bidirecionais: $u \rightarrow v$ e $v \rightarrow u$. Essas são resolvidas mantendo apenas a direção com **maior peso acumulado**:

$$
\text{Se } w(u \rightarrow v) > w(v \rightarrow u): \text{remover } v \rightarrow u
$$

Em caso de empate ($w(u \rightarrow v) = w(v \rightarrow u)$), a decisão é aleatória (controlada por `random_state`).

### 9.5 Resultado

O grafo final é um `nx.DiGraph` (NetworkX) com:
- **Nós**: predicados + dois nós terminais de classe
- **Arestas**: ponderadas pela importância acumulada ao longo de todos os bags
- **Topologia**: reflete a frequência e consistência com que certos predicados aparecem juntos e se conectam a certas classes

---

## 10. Centralidade Local de Alcance (LRC)

### 10.1 Conceito

A **Local Reaching Centrality (LRC)** quantifica a importância de um nó com base na sua capacidade de alcançar outros nós no grafo, levando em conta os pesos das arestas. Nós com alto LRC são "hubs" — predicados centrais que conectam diferentes partes do grafo.

### 10.2 Definição Formal

Para um grafo dirigido $G = (V, E)$ com pesos $w: E \rightarrow \mathbb{R}^+$, a LRC do nó $v$ é definida como:

$$
\text{LRC}(v) = \frac{1}{|V| - 1} \sum_{u \in V \setminus \{v\}} \frac{d(v, u)}{D}
$$

onde:
- $d(v, u)$ é a distância (caminho mais curto ponderado) de $v$ a $u$
- $D$ é o diâmetro ponderado do grafo (maior distância entre quaisquer dois nós alcançáveis)

Na implementação do NetworkX, `nx.local_reaching_centrality(G, v, weight='weight')` calcula essa medida considerando os pesos das arestas.

### 10.3 Interpretação no Contexto SMX

- **Alto LRC**: O predicado está bem posicionado no grafo, conectando-se a muitos outros predicados com arestas de alto peso. Indica alta importância global.
- **Baixo LRC**: O predicado é periférico — conecta-se a poucos nós ou com baixo peso. Menor importância global.

### 10.4 Extração do Ranking

O resultado é um DataFrame ordenado por LRC (decrescente):

| Node | Local_Reaching_Centrality | Zone | Threshold | Operator |
|------|--------------------------|------|-----------|----------|
| Fe ka $>$ 1.23 | 0.847 | Fe ka | 1.23 | $>$ |
| Ca ka $\leq$ -0.56 | 0.723 | Ca ka | -0.56 | $\leq$ |
| ... | ... | ... | ... | ... |

---

## 11. Agregação Multi-Semente

### 11.1 Procedimento

O pipeline completo (bagging → perturbação → grafo → LRC) é repetido para múltiplas sementes aleatórias (tipicamente $\{0, 1, 2, 3\}$). Os valores de LRC de todas as sementes são então **agregados pela média**:

$$
\overline{\text{LRC}}(v) = \frac{1}{|\mathcal{R}|}\sum_{r \in \mathcal{R}} \text{LRC}_r(v)
$$

onde $\mathcal{R}$ é o conjunto de sementes.

### 11.2 Ranking por Zona

Para o ranking final, seleciona-se o **predicado de maior LRC médio por zona**, eliminando duplicidades:

$$
P_m^* = \arg\max_{P_j : \text{zona}(P_j) = Z_m} \overline{\text{LRC}}(P_j)
$$

Isso produz um ranking de zonas espectrais pelo seu predicado mais relevante.

---

## 12. Mapeamento de Thresholds para o Espaço Natural

### 12.1 O Problema

Os predicados são gerados no **espaço pré-processado** (Poisson + MC). Os limiares $\tau$ nesse espaço não têm interpretação direta para o especialista. É necessário mapeá-los de volta para o **espaço natural** (escala original dos dados).

### 12.2 Método: Amostra Mais Próxima

A função `map_thresholds_to_natural` utiliza uma abordagem de **mapeamento por amostra mais próxima**:

1. **PCA é ajustada no espaço natural**: Os dados originais (sem pré-processamento) passam pela mesma extração de zonas e agregação PCA, produzindo scores no espaço natural $t_i^{(m,\text{nat})}$.

2. **Para cada predicado com limiar $\tau^{\text{prep}}$**: Encontra-se a amostra cujo score no espaço pré-processado é mais próximo do limiar:

$$
i^* = \arg\min_i |t_i^{(m,\text{prep})} - \tau^{\text{prep}}|
$$

3. **O score natural dessa amostra é o limiar natural**:

$$
\tau^{\text{nat}} = t_{i^*}^{(m,\text{nat})}
$$

4. **Erro de aproximação**: Também é registrado o erro $\epsilon = |t_{i^*}^{(m,\text{prep})} - \tau^{\text{prep}}|$.

### 12.3 Limitações

- A qualidade do mapeamento depende da densidade de amostras ao redor do limiar
- Para amostras esparsas, o erro de aproximação pode ser significativo
- O método assume que a **ordenação relativa** das amostras é preservada entre os espaços pré-processado e natural (para limiares baseados em quantis, isso é geralmente verdadeiro)

---

## 13. Reconstrução do Threshold Multivariado — Do Score ao Espectro

### 13.1 O Conceito de Threshold Multivariado

Esta é uma das contribuições mais inovadoras do SMX com agregação PCA. Enquanto um predicado como "Fe ka $> 1.23$" define um limiar sobre um **valor escalar** (o score PC1), o threshold multivariado reconstrói esse limiar no **espaço espectral original**, gerando um **espectro limiar** — um vetor em $\mathbb{R}^{d_m}$ que define a fronteira do predicado na dimensão espectral completa da zona.

### 13.2 Fundamentação Matemática

#### 13.2.1 Da Projeção à Reconstrução

Lembre que a PCA projeta os dados centrados sobre o vetor de pesos:

$$
t_i^{(m)} = (\mathbf{x}_i^{(m)} - \bar{\mathbf{x}}_{Z_m})^T \mathbf{w}_1^{(m)}
$$

Para reconstruir um score $q$ (no caso, o threshold $\tau$) de volta ao espaço original, invertemos a projeção. Na PCA com 1 componente, a reconstrução exata é:

$$
\hat{\mathbf{x}}^{(m)} = \bar{\mathbf{x}}_{Z_m} + q \cdot \mathbf{w}_1^{(m)}
$$

#### 13.2.2 O Espectro Limiar (Threshold Spectrum)

Substituindo $q$ pelo valor do threshold $\tau$, obtemos o **espectro limiar**:

$$
\boldsymbol{\tau}^{\text{espectro}} = \bar{\mathbf{x}}_{Z_m} + \tau \cdot \mathbf{w}_1^{(m)}
$$

onde:
- $\bar{\mathbf{x}}_{Z_m} \in \mathbb{R}^{d_m}$ é o vetor de médias da zona (centro dos dados)
- $\tau \in \mathbb{R}$ é o valor do threshold (escalar, no espaço de scores)
- $\mathbf{w}_1^{(m)} \in \mathbb{R}^{d_m}$ é o vetor de loadings do PC1 (direção de máxima variância)
- $\boldsymbol{\tau}^{\text{espectro}} \in \mathbb{R}^{d_m}$ é o espectro resultante

#### 13.2.3 Interpretação Geométrica

O espectro limiar é o **ponto no espaço espectral que está exatamente sobre a fronteira do predicado**. Geometricamente:

- $\bar{\mathbf{x}}_{Z_m}$ é o centroide da nuvem de pontos (amostras) no espaço da zona
- $\mathbf{w}_1^{(m)}$ define a direção principal de variação
- $\tau \cdot \mathbf{w}_1^{(m)}$ é o deslocamento ao longo dessa direção até o threshold
- O espectro limiar é, portanto, o centroide deslocado na direção do PC1

```
                    ↑ w₁
                    │         Amostras
                    │    ○      ○
  ←────────────●────│────○──●───○──────→ PC1
         τ<0   │    │      centro   τ>0
               │    │   (média)
    Threshold  │
    espectro   │
```

#### 13.2.4 Para o Espaço Natural

Para obter o threshold multivariado no espaço natural (não pré-processado), utiliza-se a PCA ajustada nos dados originais:

$$
\boldsymbol{\tau}_{\text{nat}}^{\text{espectro}} = \bar{\mathbf{x}}_{Z_m}^{\text{nat}} + \tau^{\text{nat}} \cdot \mathbf{w}_{1,\text{nat}}^{(m)}
$$

onde $\bar{\mathbf{x}}_{Z_m}^{\text{nat}}$ e $\mathbf{w}_{1,\text{nat}}^{(m)}$ são a média e os loadings da PCA ajustada no espaço natural.

### 13.3 Significado Prático

O espectro limiar pode ser sobreposto aos espectros individuais das amostras em uma visualização:

- **Amostras acima do espectro limiar**: tendem a satisfazer o predicado "$> \tau$"
- **Amostras abaixo do espectro limiar**: tendem a satisfazer o predicado "$\leq \tau$"

Para um especialista em XRF, por exemplo, o espectro limiar da zona "Fe ka" mostra a **intensidade mínima de fluorescência do Ferro** que diferencia os dois grupos de amostras. Isso traduz uma regra estatística abstrata em uma informação espectroscópica concreta e interpretável.

### 13.4 Vantagem sobre Agregadores Simples

Com agregadores como soma ou média, o threshold é um número ($\tau = 25.3$) sem correspondência espectral. Não há como saber **como** aquele valor se distribui entre as variáveis da zona. Com PCA:

| Aspecto | Soma/Média | PCA |
|---------|-----------|-----|
| Threshold | Escalar único ($\tau = 25.3$) | Vetor espectral ($\boldsymbol{\tau} \in \mathbb{R}^{d_m}$) |
| Interpretação | "A soma dos valores é 25.3" | "O espectro limiar tem esta forma" |
| Visualização | Linha horizontal | Espectro sobreponível aos dados |
| Ponderação | Todas variáveis iguas | Ponderação ótima (PC1 loadings) |
| Informação de qualidade | Não | Variância explicada |

---

## 14. Exemplo Completo: Do Espectro à Explicação

Para consolidar toda a teoria, vamos acompanhar o pipeline completo para um dataset real de XRF de cédulas bancárias (*bank notes*).

### Passo 1 — Dados

```python
# 150 amostras (75 autênticas + 75 falsas), 997 variáveis espectrais (2.74 - 22.71 keV)
data = pd.read_csv('real_datasets/xrf/bank_notes.csv', sep=';')
```

### Passo 2 — Pré-processamento Poisson + MC

$$
\mathbf{X}^{\text{prep}} = \frac{\mathbf{X}}{\sqrt{\bar{\mathbf{x}}}} - \overline{\left(\frac{\mathbf{X}}{\sqrt{\bar{\mathbf{x}}}}\right)}
$$

### Passo 3 — Modelo PLS-DA

```python
pls_model = pls_optimized(X_prep, y, LVmax=4, aim='classification')
y_pred_cont = plsda_results[5].iloc[:, -1]  # saída contínua [0, 1]
```

### Passo 4 — Definição de 15 zonas espectrais

```python
spectral_cuts = [
    ('Ar ka + Ag L', 2.76, 3.47),
    ('Ca ka', 3.50, 3.91),
    ('Fe ka', 6.15, 6.76),
    # ... 15 zonas no total
]
```

### Passo 5 — Extração e Agregação PCA

```python
zones = extract_spectral_zones(X_prep, spectral_cuts)      # 15 sub-DataFrames
scores, pca_info = aggregate_spectral_zones_pca(zones)      # 15 scores PC1
```

Resultado: DataFrame $\mathbf{T}$ com $n$ linhas e 15 colunas (uma por zona).

### Passo 6 — Geração de Predicados

```python
pred_df, indicator_df, cooccurrence = predicates_by_quantiles(scores, [0.2, 0.4, 0.6, 0.8])
```

Resultado: Até $2 \times 15 \times 4 = 120$ predicados (após remoção de duplicatas).

### Passo 7 — Bagging (para cada semente)

```python
bags = bagging_predicates(scores, y_pred_cont, pred_df,
    n_bags=10, n_samples_per_bag=int(0.8*n), replace=False,
    sample_bagging=True, predicate_bagging=False, random_seed=seed)
```

### Passo 8 — Perturbação

```python
pert_results = calculate_predicate_perturbation(
    estimator=pls_model,
    Xcalclass_prep=X_prep,
    folds_struct=bags,
    predicates_df=pred_df,
    spectral_cuts=spectral_cuts,
    perturbation_mode='median',    # ◄— mediana
    stats_source='full',           # ◄— todo o dataset
    aim='regression',
    metric='mean_abs_diff'         # ◄— |y_orig - y_pert|
)
```

**O que acontece internamente para o predicado "Ca ka ≤ -1.15" no Bag_3:**

1. Identifica ~18 amostras no Bag_3 que satisfazem "Ca ka ≤ -1.15"
2. Localiza as 21 colunas espectrais da zona Ca ka (3.50 - 3.91 keV)
3. Cria cópia dos dados das 18 amostras
4. Substitui as 21 colunas pela **mediana** (calculada sobre todas as $n$ amostras)
5. Prediz com `pls_model.predict()` antes e depois
6. Calcula $\frac{1}{18}\sum_{i=1}^{18}|\hat{y}_i - \hat{y}_i^{\text{pert}}| = 0.0342$

### Passo 9 — Grafo

```python
DG = build_predicate_graph(bags, pert_results,
    metric_column='Perturbation', var_exp=True, pca_info_dict=pca_info)
```

### Passo 10 — LRC

```python
lrc_df = calculate_lrc_single_graph(DG, pred_df)
```

### Passo 11 — Agregação e Mapeamento Natural

```python
# Agregar LRC de 4 sementes
lrc_avg = lrc_all.groupby('Node')['Local_Reaching_Centrality'].mean()

# Mapear para espaço natural
zones_nat = extract_spectral_zones(X_original, spectral_cuts)
scores_nat, pca_nat = aggregate_spectral_zones_pca(zones_nat)
lrc_natural = map_thresholds_to_natural(lrc_df, scores, scores_nat)
```

### Passo 12 — Threshold Multivariado

```python
# Para o predicado mais importante (e.g., "Fe ka > 1.23")
threshold_spectrum = reconstruct_threshold_to_spectrum(
    threshold_value=1.23_natural,   # threshold no espaço natural
    zone_name='Fe ka',
    pca_info_dict=pca_nat
)
# threshold_spectrum é um vetor de ~30 valores (um para cada energia na zona Fe ka)
```

### Passo 13 — Visualização

O espectro limiar (linha tracejada vermelha) é sobreposto aos espectros das amostras:
- Espectros de cédulas autênticas (Classe A) em dourado
- Espectros de cédulas falsas (Classe B) em azul
- O espectro limiar separa visualmente os dois grupos na zona do Ferro

---

## 15. Comparação com Outros Métodos de Explicabilidade

O SMX é comparado com métodos tradicionais de explicabilidade de modelos espectrais por meio do **RBO (Rank-Biased Overlap)**, que quantifica a similaridade entre rankings de zonas produzidos por diferentes métodos.

| Método | Tipo | Dependência do Modelo | Granularidade |
|--------|------|----------------------|---------------|
| **SMX-Perturbação** | Model-agnostic (requer `predict`) | Indireta (via previsões) | Predicado (zona + limiar) |
| **SMX-Covariância** | Model-agnostic | Indireta (associação estatística) | Predicado |
| **SHAP** | Model-agnostic | Indireta | Variável individual |
| **VIP Scores** | Model-specific (PLS) | Direta (pesos do modelo) | Variável individual |
| **Coeficientes de Regressão** | Model-specific (PLS) | Direta | Variável individual |
| **Permutation Importance** | Model-agnostic | Indireta | Variável individual |

### Vantagens do SMX-Perturbação

1. **Interpretabilidade química**: Rankings em nível de zona (com significado químico), não de variáveis isoladas.
2. **Threshold multivariado**: Possibilidade única de reconstruir o limiar como espectro, viabilizada pela agregação PCA.
3. **Robustez via bagging e multi-semente**: Reduz a variância do ranking.
4. **Flexibilidade**: Funciona com qualquer modelo (PLS, SVM, MLP, etc.) e em contextos de regressão e classificação.
5. **Causalidade direta**: A perturbação mede o impacto causal real sobre as previsões, ao contrário da covariância que mede apenas associação.

---

## 16. Parâmetros Recomendados

Com base nos experimentos realizados em 8 datasets de XRF com 3 tipos de modelo:

| Parâmetro | Valor Recomendado | Justificativa |
|-----------|------------------|---------------|
| Quantis | $\{0.2, 0.4, 0.6, 0.8\}$ | Boa cobertura da distribuição |
| `n_bags` | 10 | Equilíbrio entre robustez e custo |
| `n_samples_per_bag` | 80% de $n$ | Representatividade sem excesso |
| `min_samples_per_predicate` | 20% de $n$ | Garantia de cobertura mínima |
| `replace` | `False` | Subamostragem preferida a bootstrap |
| `sample_bagging` | `True` | Ativado para diversidade |
| `predicate_bagging` | `False` | Todos os predicados avaliados em cada bag |
| Sementes aleatórias | $\{0, 1, 2, 3\}$ | 4 repetições para robustez |
| `perturbation_mode` | `'median'` | Robusta a outliers |
| `stats_source` | `'full'` | Estatísticas da população inteira |
| `metric` (regressão) | `'mean_abs_diff'` | Magnitude do impacto, sem viés direcional |
| `metric` (classificação) | `'probability_shift'` | Sensível a mudanças graduais |
| `var_exp` | `True` | Ponderar pela qualidade da representação PCA |
| Threshold covariance | $0.01$ | Filtro mínimo para covariância |
| Agregação LRC | Média entre sementes | Estável e representativa |
| RBO | $p=0.7, k=10$ | Foco no topo do ranking |

---

## 17. Conclusão

O método SMX com perturbação espectral oferece um framework completo para a explicabilidade de modelos espectrais. Suas principais contribuições são:

1. **Uma nova filosofia de explicabilidade**: Combinar zonas espectrais com significado físico-químico, discretização por quantis, e teoria de grafos para produzir explicações que respeitam a semântica do domínio.

2. **Perturbação como medida de importância**: Substituir informação espectral por o valor da mediana (ou outro estatístico) e medir o impacto nas previsões do modelo proporciona uma medida quasi-causal da relevância de cada zona espectral.

3. **O threshold multivariado**: Através da agregação PCA, o limiar escalar de um predicado pode ser reconstruído como um **espectro limiar** no espaço original — uma capacidade única que transforma regras abstratas em informação espectroscópica concreta e visualizável.

4. **Robustez estatística**: O uso de bagging, múltiplas sementes, e agregação via LRC em grafos dirigidos confere estabilidade ao ranking de importância, reduzindo a sensibilidade a ruído e a flutuações amostrais.

5. **Flexibilidade**: O método é model-agnostic (funciona com PLS, SVM, MLP e qualquer modelo com `predict()`) e suporta tanto regressão quanto classificação com métricas apropriadas.

O SMX conecta, portanto, a análise exploratória de dados espectrais, a modelagem preditiva, e a interpretabilidade de modelos em um pipeline unificado, gerando explicações que são tanto estatisticamente rigorosas quanto quimicamente significativas.

---

*Documento gerado como referência teórica e didática do método SMX (Spectral Model eXplainability) com foco na abordagem de perturbação espectral. Baseado na análise do repositório SMX e seus experimentos com 8 datasets de XRF utilizando modelos PLS-DA, SVM e MLP.*
