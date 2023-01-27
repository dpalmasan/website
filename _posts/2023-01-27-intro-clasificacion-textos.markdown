---
layout: post
title:  "Introducción a la Clasificación de Textos"
date:   2023-01-27 11:10:03 -0400
categories: python algorithms classification machine-learning
---

En el [último post hasta la fecha]({{ site.baseurl }}{% link _posts/2023-01-21-intro-recuperacion-informacion.markdown %}), hablé sobre el índice invertido y motores de búsqueda. En otro post previo hablé sobre [probabilidad e inferencia]({{ site.baseurl }}{% link _posts/2023-01-20-sobre-probabilidades.markdown %}), y cómo a partir de una evidencia (observación), se puede calcular la probabilidad condicionada a las observaciones. En esta entrada, mezclo ambos mundos, y explico cómo hacer un clasificador de textos, con un ejemplo de clasificación de reseñas de [IMDB](https://www.imdb.com/).

## ¿Por qué clasificar textos?

Existen varias razones para clasificar textos:

* Clasificar reseñas sobre productos (positivas, negativas, neutrales), para ver puntos débiles y mejorarlos
* Hacer búsquedas específicas en un motor de búsqueda, por ejemplo si buscamos el término "Ciencias de la Computación" y queremos sólo los textos de Chile, necesitamos antes clasificar qué textos son chilenos.
* Detección de spam, por ejemplo en correos electrónicos.
* etc.

## Modelado de Clasificación de Textos como un Problema Probabilístico

Un texto (documento) es en esencia una secuencia de $n$ términos $d = <t_1, t_2, \ldots, t_n>$. El problema de clasificación de textos, es encontrar una función $f:d \longrightarrow c$, donde $c$ es una variable discreta que representa la categoría del texto (ej. deportes, ciencia, política, etc). La probabilidad de generar un documento, dada una categoría es $P(d|c)$.

### Regla de Bayes

La regla de bayes se puede inferir utilizando la _probabilidad condicional_. Por ejemplo, sabemos:

$$P(A|B) = \displaystyle \frac {P(A, B)}{P(B)} \Rightarrow P(A, B) = P(A|B)P(B)$$

También, por definición:

$$P(B|A) = \displaystyle \frac {P(A, B)}{P(A)} \Rightarrow P(A, B) = P(B|A)P(A)$$

Finalmente, utilizando $P(A, B)$:

$$P(B|A)P(A) = P(A|B)P(B) \Rightarrow P(B|A) = \displaystyle \frac{P(A|B)P(B)}{P(A)}$$

Esto se conoce como regla de Bayes, y a pesar de que es un concepto simple, es la base de muchos modelos probabilísticos de Machine Learning.

### Estimando la Probabilidad de una Categoría dado un Documento

Volviendo a la ecuación anterior, podemos calcular la probabilidad de una categoría dado un documento, como:

$$P(c|d) =  \displaystyle \frac{P(c)P(d|c)}{P(d)} \  \alpha \  P(c)P(d|c)$$

La probabilidad del documento no cambia, por lo que podemos eliminarla y considerarla un factor de normalización (para cumplir con el axioma de que las probabilidades se encuentran entre `[0, 1]`). $P(c)$ es la probabilidad a priori de una clase $c$, esta se puede estimar utilizando estimación de máxima verosimilitud:

$$P(c) = \displaystyle \frac{N_c}{N}$$

Donde, si observamos $N$ documentos, $N_c$ son los documentos que pertenecen a la clase $c$. Ahora es cuando podemos construir un clasificador de Bayes ingenuo (Naïve Bayes). En este clasificador, asumimos que los términos del documento son independientes entre sí, por lo que podemos calcular $P(c|d)$:

$$P(c|d) \ \alpha \  \displaystyle P(c)\prod_{t_i \in V} P(t_i|c)$$

Donde V es el vocabulario de términos a considerar, y $P(t_i, c)$ representa cuánto contribuye $t_i$ en que el documento sea de la clase $c$. Si utilizamos una estimación de máxima probabilidad aposteriori (MAP), entonces podemos determinar la clase del documento:

$$c_{map} = \underset{c \in \mathbb{C}}{\mathrm{argmax}} \hat{P}(c|d) = \underset{c \in \mathbb{C}}{\mathrm{argmax}} \ \hat{P}(c) \prod_{t_i \in V} \hat{P}(t_i|c)$$

Para estimar $\hat{P}(t|c)$ podemos utilizar la frecuencia relativa de $t$ dado un documento de clase $c$:

$$\hat{P}(t|c) = \displaystyle \frac {Count_c(t)}{\sum_{t' \in V} Count_c(t')}$$

El lector que esté atento, se hará la pregunta ¿Qué ocurre si al momento de la predicción se encuentra un término $t \notin V$? En este caso podemos utilizar alguna técnica de suavizado (_smoothing_), una forma simple es utilizar _Laplace smoothing_:

$$\hat{P}(t|c) = \displaystyle  \frac {Count_c(t) + 1}{\sum_{t' \in V} Count_c(t') + 1} = \frac {Count_c(t) + 1}{\left(\sum_{t' \in V} Count_c(t')\right) + |V|}$$

### Predecir Sentimientos de Reseñas en IMDB

Utilizaremos el dataset de [reseñas del repositorio de ML de UCI](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences). Por otro lado, eliminaremos la palabras funcionales (_stopwords_) y también aplicaremos lematización como describí en el post del índice invertido:

* [Lexicon de lemas](https://github.com/michmech/lemmatization-lists)
* [Lista de stopwords](https://countwordsfree.com/stopwords)

Creamos un tokenizador:

```python
from pathlib import Path
from typing import Dict
from typing import Iterable
import re


class LemmaTokenizer:
    def __init__(self, lemmas: Dict[str, str], threshold: int = 3):
        self._lemmas = lemmas
        self._re = re.compile("[^0-9a-zA-Z']+")
        self._threshold = threshold

    def tokenize(self, text: str) -> Iterable[str]:
        return [
            self._lemmas.get(word.lower(), word.lower())
            for word in self._re.sub(" ", text).split()
            if len(word) >= self._threshold
        ]

    @classmethod
    def load_from_file(cls, filepath: Path):
        lexicon = {}
        with open(filepath, "r") as fp:
            for line in fp:
                lemma, word = line.split()
                lexicon[word.strip()] = lemma.strip()

        return cls(lexicon)
```

Lo cargamos con la lista de lemas:

```python
RESOURCE_PATH = Path("./resources")

tokenizer = LemmaTokenizer.load_from_file(RESOURCE_PATH / "lemmatization-en.txt")
```

El formato del conjunto de datos es `texto \t etiqueta`, para cargar el conjunto de datos:

```python
reviews, labels = [], []
with open(RESOURCE_PATH / "imdb_labelled.txt") as fp:
    for line in fp:
        line = line.strip()
        if line:
            review, label = line.split("\t")
            reviews.append(review)
            labels.append(label)
```

Dividimos el conjunto en dos subconjuntos, uno de entrenamiento y uno de prueba:

```python
from random import shuffle

n = len(reviews)
idx = [i for i in range(n)]
shuffle(idx)
train_reviews = []
train_labels = []
for i in idx[0:int(0.8 * n)]:
    train_reviews.append(reviews[i])
    train_labels.append(labels[i])

test_reviews = []
test_labels = []
for i in idx[int(0.8 * n):]:
    test_reviews.append(reviews[i])
    test_labels.append(labels[i])
```

Calculamos las probabilidades a priori en el conjunto de entrenamiento:

```python
from collections import Counter
from itertools import chain

stopwords = set()

with open(RESOURCE_PATH / "stop_words_english.txt") as fp:
    for line in fp:
        stopwords.add(line.strip().lower())


label_count = Counter(train_labels)
label_priors = {
    label: count / sum(label_count.values()) for label, count in label_count.items()
}
print(label_priors)
```

Cuya salida es:

```
{'1': 0.50875, '0': 0.49125}
```

Se espera que estén cerca de `0.5` ya que el conjunto original contiene mitad clase positiva y mitad clase negativa. Calculamos la frecuencia de ocurrencia de términos en el conjunto de entrenamiento y añadimos un término `UNK` para palabras no vistas en el conjunto de entrenamiento:

```
cond_count = {
    label: Counter(
    filter(lambda word: word not in stopwords,
           chain(*[tokenizer.tokenize(review)
                   for i, review in enumerate(train_reviews) if train_labels[i] == label])))
       for label in train_labels
}

for label in label_priors:
    cond_count[label]["UNK"] = 0
```

Luego calculamos las probabilidades $P(t|c)$:

```python
cond_dist = {
    label: {
        word: (count + 1) / (sum(cond_count[label].values())
                             + len(cond_count[label]))
        for word, count in cond_count[label].items()
    }
    for label in train_labels
}
```

Por ejemplo:

```python
from random import sample

print(sample(cond_dist['0'].items(), 4))
```

Da como salida:

```
[('hill', 0.0006985679357317499),
 ('impact', 0.0006985679357317499),
 ('speak', 0.0013971358714634998),
 ('lead', 0.001047851903597625)]
```

La probabilidad de `P(UNK|c)`:

```python
print(cond_dist['0']['UNK'])
```

```
0.00034928396786587494
```

Se observa que las probabilidades son valores cercanos a 0, por lo tanto su multiplicación puede producir _underflow_ (quedar en 0). Para solucionar esto, en lugar de multiplicar, aplicamos logarithmo:

$$$$

## Problema del día

Para ir con la temática, el problema de hoy estará relacionado con [búsqueda de palabras](https://github.com/dpalmasan/code-challenges/issues/16).

$$c_{map} = \underset{c \in \mathbb{C}}{\mathrm{argmax}} \left(\ log \ \hat{P}(c) + \sum_{t_i \in V} log \ \hat{P}(t_i|c)\right)$$

```python
from math import log


predictions = []
for review in test_reviews:
    probs = {}
    for label, prior in label_priors.items():
        prob = log(prior) + sum(
            log(cond_dist[label][
                word if word in cond_dist[label] else "UNK"])
            for word in tokenizer.tokenize(review) if word not in stopwords)
        probs[label] = prob
    label = max(probs, key= lambda label: probs[label])
    predictions.append(label)
```

Finalmente, calculamos la métrica _accuracy_ del modelo y lo comparamos con una línea base que sería predecir siempre la clase con mayor probabilidad a priori:

```
most_common = max(label_priors, key=lambda label: label_priors[label])
baseline = sum(most_common == label for label in test_labels) / len(test_labels)
accuracy = sum(pred == label for pred, label in zip(predictions, test_labels)) / len(test_labels)
print(f"Baseline {baseline} | Accuracy: {accuracy}")
```

```
Baseline 0.465 | Accuracy: 0.77
```

El resultado es bastante decente considerando la simplicidad del modelo. Se pueden aplicar mejoras, pero eso prefiero dejarlo para otro artículo. Dado que este modelo es de tipo generativo, también podemos generar textos dada la categoría.

## Conclusiones

* En algunos casos se puede clasificar textos dado su contenido, se pueden hacer simples estimaciones como utilizar la frecuencia de ocurrencia de cada término.
* El vocabulario a utilizar se define en el conjunto de entrenamiento, por lo que se debe tener algún mecanismo para lidiar con palabras fuera del vocabulario.
* Muchos problemas de clasificación se pueden modelar como una estimación MAP de $P(c|X)$, donde luego el problema es definir las probabilidades apriori $P(c)$ y la distribución condicional $P(X|c)$
