---
layout: post
title:  "Redes Neuronales y Variables Multi-Categoricas"
date:   2024-04-28 17:22:00 -0400
categories: probability algorithms ai
---

# Introducción

En algunos problemas donde nos interesa poder "predecir" una cierta categoría para una entrada (por ejemplo usuario, negocio, entidad genérica), tenemos variables categóricas. Una variable categórica es una variable discreta que puede tomar ciertos valores, que no necesariamente consideran un orden específico (no hay noción de desigualdad). Generalmente los modelos basados en árboles de decisión no tienen problemas para lidiar con este tipo de variables, sin embargo los modelos numéricos necesitan un tratamiento especial para poder manipular este tipo de variables.

Por otro lado, las variables categóricas por lo general pueden tomar un sólo valor (categoría). Sin embargo, existen problemas más interesantes. Por ejemplo, supongamos que tenemos IDs de páginas visitadas como una variable. El lector podrá notar, que esta no es una variable ordinal, y que este tipo de variables es *Multi-Categrórica* ya que una entrada puede tener más de una cateogoría a la vez.

En este artículo, comentaré cómo lidiar con este tipo de variables, en particular, utilizando un modelo basado en redes neuronales. También mencionaré representaciones eficientes de variables multi-categóricas y cómo en la práctica se procesan este tipo de variables.


# Variables Categóricas

Por lo general, los enfoques clásicos para procesar este tipo de variables, consisten en seguir un esquema de _One Hot Encoding_, que consiste en codificar la variable como una máscara (vector de `0s` y `1s`), donde la dimensión de este vector será `N_categorias - 1`. Por ejemplo, si tuvieramos la variable `color` que pudiese tomar los valores `[celeste, gris, café]`, se necesitaría un vector binario de dimensión 2:

* $\text{celeste} = \left(1, 0\right)$
* $\text{gris} = \left(0, 1\right)$
* $\text{café} = \left(0, 0\right)$

Se asume que un registro puede pertenecer a una sola categoría, por lo que los vectores posibles consideran todas las combinaciones para dicha dimensión vectorial. El vector $\left(1, 1\right)$ no representa algún dato, debido al supuesto mencionado.

Sin embargo, en la naturaleza y en la práctica existen otro tipo de variables. Estas son variables _Multi-Categóricas_, que consisten también en variables discretas, pero que pueden tomar múltiples valores a la vez. Un ejemplo simple es un sistema de recomendación de películas. Por ejemplo, una persona puede haber visto ciertas películas, las cuales al considerarlas como un todo, permiten aproximar, por ejemplo, los gustos de dicha persona. En este caso, la variable podrían ser los títulos de las películas vistas (ej. una lista de películas).

# Problema de Juguete

Supongamos que estamos en un universo, en el cual un usuario puede haber visitado una serie de páginas webs, y nos interesa, a partir de dichas visitas, estimar las preferencias del usuario, para por ejemplo, recomendar productos/servicios basados en el contexto actual de dicho usuario. En este universo, cada usuario $u$ se representa como un vector de páginas visitadas (por simplicidad, IDs enteras). Supongamos que tenemos las siguientes definiciones:

* Si $u$ visitó $\left(1, 5, 11, 19\right)$ entonces $u$ tiene preferencias de contenido acerca de comida.
* Si $u$ visitó $\left(2, 4, 8, 16\right)$ entonces $u$ tiene preferencias de contenido acerca de películas.
* En cualquier otro caso, no sabemos con certeza las preferencias de $u$.

Supongamos que existen un total de 200 posibles páginas web a visitar. ¿Cómo podríamos representar a cada usuario $u$ para poder definir un modelo predictivo?

Una forma ingenua de hacerlo, es utilizando un vector máscara, similar a lo que se hace para variables categóricas simples. Por ejemplo, si un usuario $u_i$ visitó las páginas `[1, 3, 5]` entonces su representación sería:

* $u_i = \left(0, 1, 0, 1, 0, 1, 0, 0\ldots 0 \right)$

El lector atento ya puede ver cuál es el problema, pero para hacerlo más claro, se obtiene un vector muy disperso (gran porcentaje de ceros), lo que por lo general trae problemas de estabilidad numérica (arrastre de error), además de uso ineficiente de memoria. ¿Qué tal si representaramos cada categoría en un espacio vectorial de dimensión $K$ y además tuviésemos una lista $N$ donde $N$ es el número de `IDs` posibles para las páginas visitadas? Este tipo de representación se conoce como _embeddings_.

Por otro lado, el problema no es una simple reducción de dimensionalidad, tenemos que encontrar una forma más inteligente de representar a los usuarios. Supongamos que tenemos el siguiente _batch_ de usuarios:

* `u1 = [1, 4, 9]`
* `u2 = [3, 2]`
* `u3 = [11, 17, 12, 19]`

No necesitamos utilizar todas las categorías posibles para representar un usuario, en especial si posteriormente queremos agregar (aplicar _pooling_) la representación vectorial de las páginas. En este caso, podríamos representar el _batch_ como dos tensores, uno de _offsets_ y otro de valores:

* `values = [1, 4, 9, 3, 2, 11, 17, 12, 19]`
* `offsets = [0, 3, 5]`

Los valores, son simplemente todos los valores de los usuarios en el _batch_ concatenados, y los _offsets_ representan la posición donde comienza cada registro (en este caso usuario). Esta es una representación más eficiente que sólo requiere la información a procesar. Finalmente, aplicando algún mecanismo de _pooling_ podemos calcular el centro de masa, la suma vectorial, etc. del conjunto de vectores que representa al usuario, tal y como se muestra en la figura 1.


<div align="center">

![user_emb](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/4f2bc2edf17bd2522ba7ccd21b08ed92e108feb9/pooling.png)

_Fig 1: Representación de usuario y páginas en espacio vectorial._

</div>

## Implementando un Clasificador de Usuarios basado en sus páginas visitadas

La red neuronal a ajustar será una simple red neuronal de dos capas, y tendrá una Bolsa de Embeddings (_Embedding bag_) para transformar la representación del usuario un una representación vectorial. Esta arquitectura se muestra en la figura 2.

<div align="center">

![user_emb](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/5c93d818a78602489880bec56eff1567bb430d34/categorical_classifier.png)

_Fig 2: Arquitectura de Clasificador para universo de problema multi-categórico._

</div>

Este universo lo simularemos en `Python` con `Pytorch`. Comencemos importando las bibliotecas a utilizar:

```py
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
from enum import Enum
```

Definamos un conjunto de constantes:

```py
SAMPLES = 50000
NUM_OF_CATEGORIES = 200
TRAINING_SAMPLES = int(SAMPLES * 0.8)
TEST_SAMPLES = int(SAMPLES * 0.2)
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50
EMB_DIM = 8
HIDDEN_DIM = 16
```

Muestreo de datos en este universo:

```py
def generate_sparse_features(samples, max_categories):
    assert max_categories > 4
    rows = []
    universe = list(range(max_categories))
    for _ in range(samples):
        offset = random.randint(max_categories // 4, max_categories // 2)
        random.shuffle(universe)
        row = []
        for i in universe[:offset]:
            row.append(i)
        rows.append(row)

    return rows

training_sparse_features = generate_sparse_features(TRAINING_SAMPLES, NUM_OF_CATEGORIES)
```

Definamos las clases a las que el usuario podría pertenecer:

```py
class Preferences(Enum):
    NEUTRAL = 0
    LIKES_MOVIE_CONTENT = 1
    LIKES_FOOD_CONTENT = 2
```

Definamos la función oculta a aproximar:

```py
def exact_function_to_approximate(x_set: List[int]) -> int:
    if set({1, 5, 11, 19}).issubset(set(x_set)):
        return Preferences.LIKES_FOOD_CONTENT

    if set({2, 4, 8, 16}).issubset(set(x_set)):
        return Preferences.LIKES_MOVIE_CONTENT

    return Preferences.NEUTRAL
```

Estadísticas:

```py
training_labels_sparse = [exact_function_to_approximate(row) for row in training_sparse_features]
Counter(training_labels_sparse)
```

```
Counter({<Preferences.NEUTRAL: 0>: 38167,
         <Preferences.LIKES_FOOD_CONTENT: 2>: 941,
         <Preferences.LIKES_MOVIE_CONTENT: 1>: 892})
```

Convertir etiquetas a enteros (para luego transformar a tensores).

```py
training_labels_sparse = [x.value for x in training_labels_sparse]
```

Definamos el modelo en `Pytorch`:

```py
class SimpleMultiCategoricalClassifier(torch.nn.Module):
    def __init__(self, num_embeddings: int,
                 embed_dim: int,
                 hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings,
                                         embed_dim,
                                         mode="sum")
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 3)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        offsets: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        embed = self.embedding(values, offsets)
        return self.net(embed)

    def train_batch(
        self,
        offsets: torch.Tensor,
        values: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        out = self.forward(offsets, values)
        loss = self.criterion(out, labels)
        loss.backward()
        return loss.item()
```

Prueba inicial de funcionamiento:

```py
model = SimpleMultiCategoricalClassifier(num_embeddings=NUM_OF_CATEGORIES,
                                         embed_dim=16,
                                         hidden_dim=8)
example_values = torch.tensor(training_sparse_features[1])
example_offsets = torch.tensor([0, 1])
model(example_offsets, example_offsets)
```

```
tensor([[ 0.1533, -0.3507, -0.0683],
        [ 0.2590, -0.4411, -0.1981]], grad_fn=<AddmmBackward0>)
```

Definamos el _Dataloader_:

```py
class SimpleSparseDenseDataLoader(Dataset):
    def __init__(self, sparse_features, labels):
        self.sparse_features = sparse_features
        self.labels = labels

    def __len__(self):
        return len(self.sparse_features)

    def __getitem__(self, idx):
        return self.sparse_features[idx], self.labels[idx]
```

Debemos transformar los datos de manera de obtener los tensores de _offset_ y _values_, por lo que definimos una función personalizada para procesar los _batches_ de datos:

```py
def collate_batch(batch):
    label_list, data_list, offsets = [], [], [0]
    for data, label in batch:
        label_list.append(label)
        data_list.extend(data)
        offsets.append(len(data))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    values_list = torch.tensor(data_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    assert len(offsets) == len(label_list),\
        f"Offsets numel {len(offsets)} != labels numel {len(label_list)}"
    return label_list, offsets, values_list
```

Para evitar tener problemas con los pesos de la red, definimos funciones para inicializar los pesos de la red:

```py
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
    if classname.find("EmbeddingBag") != -1:
        initrange = 0.5
        m.weight.data.uniform_(-initrange, initrange)
```

Definimos el modelo:

```py
criterion = nn.CrossEntropyLoss()
model = SimpleMultiCategoricalClassifier(num_embeddings=NUM_OF_CATEGORIES,
                                         embed_dim=EMB_DIM,
                                         hidden_dim=HIDDEN_DIM)
model.apply(weights_init_uniform_rule)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_data = SimpleSparseDenseDataLoader(list(training_sparse_features),
                                         list(training_labels_sparse))
dataloader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
```

Creamos datos nuevos para validar la precisión del modelo:

```py
test_sparse_features = generate_sparse_features(TEST_SAMPLES, NUM_OF_CATEGORIES)
test_labels_sparse = [exact_function_to_approximate(row).value for row in test_sparse_features]
test_data = SimpleSparseDenseDataLoader(list(test_sparse_features), list(test_labels_sparse))
test_dataloader = DataLoader(
    test_data, batch_size=TEST_SAMPLES, shuffle=True, collate_fn=collate_batch
)
```

Creamos una función para evaluar la precisión del modelo y evaluamos la precisión previa al entrenamiento:

```py
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, offsets, values) in enumerate(dataloader):
            predicted_label = model(offsets, values)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

evaluate(test_dataloader)
```

```
0.2313
```

Entrenamos la red neuronal:

```py
train_loss = []
model.train()
for epoch in range(1, EPOCHS + 1):
    avg_loss = 0
    for idx, (label, offsets, values) in enumerate(dataloader):
        optimizer.zero_grad()
        loss_item = model.train_batch(offsets, values, label)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        avg_loss += loss_item

    avg_loss = avg_loss / BATCH_SIZE
    train_loss.append(avg_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}|{EPOCHS} Avg Loss: {avg_loss}")
```

Graficamos la función de pérdida en cada epoch:

```py
plt.plot(range(1, len(train_loss) + 1), train_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
```

<div align="center">

![multicat_lc](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/82535439ed579a98a62c825c9448aaf98835fd01/multicat-learning-curve.png)

_Fig 3: Curva de aprendizaje de modelo para universo de usuario-páginas._

</div>

Finalmente evaluamos el modelo ya entrenado:

```py
evaluate(test_dataloader)
```

```
0.9999
```

Observamos que la precisión aumentó de `~0.2` a `~0.99`. Estos números pueden variar de ejecución en ejecución (debido a la aleatoriedad de la inicialización de los pesos y de las pasadas por los _batches_).

# Conclusiones y Reflexiones

* Existen problemas en los que tenemos variables discretas donde cada variable puede tomar 0 o más valores (ejemplo: Lista de `IDs`).
* Representar variables como máscaras de vectores se vuelve intratable con el aumento de categorías, por lo que se necesitan representaciones más eficientes para evitar procesar datos dispersos.
* Una estrategia para representar multi-categorías es utilizar dos tensores, uno para posiciones de cada registro y otro para los valores. Las categorías pueden tener una representación latente en forma de vectores (_embeddings_).
* Se puede representar un registro como una combinación interna de pesos de una red neuronal y con ello se puede crear un modelo para un problema de clasificación dado.

El notebook jupyter se encuentra disponible en Colab: [Ejemplo de problema multi-categórico](https://colab.research.google.com/drive/1JdIDklTFa_82BS7d_hxLxt20ldwRc8zB?usp=sharing)
