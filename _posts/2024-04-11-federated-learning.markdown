---
layout: post
title:  "Introducción al Aprendizaje Federado"
date:   2024-04-11 11:09:00 -0400
categories: probability algorithms ai
---

# Introducción

Un post para hablar de un tema interesante en aprendizaje automático (a.k.a _Machine Learning_). Según la literatura, el aprendizaje federado (también llamado aprendizaje colaborativo) es una técnica de aprendizaje automático que entrena un algoritmo a través de una arquitectura descentralizada formada por múltiples dispositivos los cuales contienen sus propios datos locales y privados. En esencia es una técnica para entrenar datos sin la necesidad de enviar los datos a un lugar central (por ejemplos servidores), con el fin de cumplir ciertas restricciones de privacidad.

En este post, simplemente daré un ejemplo de simulación de esta técnica, y lo abordaremos con un ejemplo simple.

# Aprendizaje Federado

En la figura 1, se muestra una simplificación sobre qué es el aprendizaje federado. En esencia, dado un universo de $N$ clientes (ej. dispositivos android), se escoge un conjunto de $K$ clientes. De esos $K$ clientes, se extraen datos locales y se ejecuta entrenamiento de ML. Luego, mediante algún canal se envía la infomación relevante a un agregador, en general los gradientes de cada paso de entrenamiento, y finalmente en el servidor, se ejecuta un paso de optimización.


<div align="center">

![fl_high_level](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/848b5f71dcca6f04032886f0255132dc85383cf7/Screenshot%25202024-04-11%2520at%252010.50.33%25E2%2580%25AFAM.png)

_Fig 1: Ilustración del proceso de aprendizaje federado._

</div>

## Simulando Aprendizaje Federado

Simularemos una situación en que queremos ajustar un modelo para un problema de clasificación binaria. El modelo a considerar será una red neuronal de dos capas. Primero importamos los datos:

```python
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from typing import Tuple, Dict
import random
import copy
```

Generamos un espacio sintético, donde ya conocemos la función $f$ que determina las clases en base a dos características:

```py
N = 1000

# Real weights
w = [0.3, 0.2, 0.8]

# Generate fake data between -1 and 1
x = np.random.rand(N, 2)*2 - 1
x = np.concatenate([np.ones((N, 1)), x], axis=1)

# Labels based on the weights
labels = (np.sum(w * x, axis=1) > 0).astype(np.int8)
```

Graficamos los datos y las clases, para asegurarnos que estamos definiendo el problema de forma correcta:

```py
plt.scatter(x[:, 1], x[:, 2], c=labels)

xbar = np.array([-1, 1])
ybar = -(xbar * w[1] + w[0]) / w[2]
plt.plot(xbar, ybar, "--k")
```

<div align="center">

![ex_fl_class](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/f009d90093741af7d76ac37b4ba2ecf92d8c6da0/fl_simple_2f_model.png)

_Fig 2: Ilustración del problema de clasificación del ejemplo._

</div>

Definimos la función de hipótesis y modelo a ajustar:

```py
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid(),
        )
        self.criterion = nn.BCELoss()

    def forward(
        self, input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x = self.net(input)
        return {"prediction": x}

    def train_batch(
        self,
        input: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        out = self.forward(input)["prediction"]
        loss = self.criterion(out, labels)
        loss.backward()

        return {"loss": float(loss.item())}

    def eval_batch(
        self,
        input: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        out = self.forward(input)["prediction"]
        loss = self.criterion(out, labels)

        return {"loss": float(loss.item())}
```

Para simular los clientes, creamos una función que carga datos sintéticos, los cuales no son visibles para el servidor simulado:

```py
def get_data_batches(w, N):
    n_batches = np.random.randint(5, 10)
    batches = []
    for _ in range(n_batches):
        x_test = np.random.rand(N, 2) * 2 - 1
        x_test = np.concatenate([np.ones((N, 1)), x_test], axis=1)

        # Labels based on the weights
        labels_test = (np.sum(w * x_test, axis=1) > 0).astype(np.int8)
        x_test = torch.tensor(x_test[:, 1:]).float()
        batches.append((x_test, labels_test))

    return batches
```

Creamos una función que ejecutará los pasos de entrenamiento dado un _batch_ de datos:

```py
def run_client_batch(model, w, N, device):
    batches = get_data_batches(w, N)
    n = len(batches)
    for x, labels in batches:
        labels = torch.tensor(labels).float().view(-1, 1)
        model.train_batch(x.to(device), labels.to(device))["loss"]
    return model
```

En cada iteración para actualizar el server, se debe ejecutar una ronda de entrenamiento que considera los $K$ clientes. En dicha ronda, se actualizan los modelos locales

```py
def run_round(model, device, K):
    models = []
    for _ in range(K):
        new_model = SimpleModel()
        new_model.load_state_dict(copy.deepcopy(model.state_dict()))
        new_model = run_client_batch(new_model, w, N, device)
        models.append(new_model)
    return models
```

Posteriormente, necesitamos una función que agregue los gradientes de los modelos locales:

```py
def aggregate_gradients(models):
    grads = []
    for m in models[0].parameters():
        if m.requires_grad:
            grads.append(m.grad)

    for model in models[1:]:
        for i, m in enumerate(model.parameters()):
            if m.requires_grad:
                grads[i] += m.grad
    return grads
```

Ahora definamos el modelo y sus parámetros. También definiremos el optimizador en el lado del servidor, para actualizar el modelo global:

```py
model = SimpleModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=1)
```

Ahora ejecutamos `100` rondas de entrenamiento, donde en cada ronda, se obtendrán datos de cada cliente (dispositivo simulado) se entrenarán los modelos locales, se enviarán los gradientes al servidor, estos se agregarán y se ejecutará un paso de entrenamiento del modelo global:

```py
loss = []
accs = []
rounds = 100
for r in range(rounds):
    K = np.random.randint(20, 30)
    models = run_round(model, device, K)
    grads = aggregate_gradients(models)
    x_test = np.random.rand(N, 2) * 2 - 1
    x_test = np.concatenate([np.ones((N, 1)), x_test], axis=1)

    labels_test = (np.sum(w * x_test, axis=1) > 0).astype(np.int8)
    x_test = torch.tensor(x_test[:, 1:]).float()
    labels_test = torch.tensor(labels_test).float().view(-1, 1)
    with torch.no_grad():
        y_pred = model.net(x_test)
        accs.append(((y_pred.round().view(-1)).int() == labels_test.view(-1)).sum() / len(labels_test))

    optimizer.zero_grad()
    # Update global model's gradients
    for i, m in enumerate(model.parameters()):
        if m.requires_grad:
            m.grad = grads[i]

    # Eval current model performance
    loss_item = model.eval_batch(x_test, labels_test)["loss"]
    loss.append(loss_item)
    optimizer.step()
```

Finalmente, graficamos las funciones de pérdida y la precisión del modelo en cada ronda:

```py
plt.figure()
plt.plot(range(1, len(loss) + 1), loss)
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Loss after each clients round")
plt.figure()
plt.plot(range(1, len(accs) + 1), accs)
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Accuracy after each clients round")
```

<div align="center">

![loss_fl](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/30d78bda8814ee4806283ab828959362d5627f52/loss.png)

_Fig 3: Pérdida en cada round de entrenamiento del ejemplo de FL._

</div>

<div align="center">

![loss_fl](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/30d78bda8814ee4806283ab828959362d5627f52/acc.png)

_Fig 4: "Accuracy" en cada round de entrenamiento del ejemplo de FL._

</div>

# Proyecto Opensource

Dejo como dato un simulador de aprendizaje federado que es de código abierto y mucho más robusto que este ejemplo de juguete: [FLSim](https://github.com/facebookresearch/FLSim).

# Final

Espero que haya gustado este mini-post. La idea es ser informativo de algunas técnicas que pueden ser desconocidas para algunas personas. Abrazos...
