---
layout: post
title:  "Árboles de Decisión, lo que probablemente no sabías"
date:   2023-02-18 11:10:03 -0400
categories: python algorithms classification machine-learning
---

Probablemente, si trabajas con _Machine Learning_, has notado que uno de los modelos más usados es _Extreme Gradient Boosting Decision Trees_ o _XGBoost_, o algún modelo similar. La práctica que funciona en general, es tomar los datos, insertarlos en la juguera y probablemente obtener un resultado. Las _API_ y frameworks disponibles hacen que la tarea no sea complicada. Si bien, en general en la práctica, el problema generalmente se resuelve teniendo los datos correctos, existen algunos casos en que incluso teniendo una gran disponibilidad de datos a mano, los modelos no tengan buen desempeño. En este caso, el problema puede deberse a múltiples fuentes, sin embargo cuando hay que _"entrar a picar"_, a veces el problema real está en no entender los modelos ni sus fundamentos.

En esta entrada, como dice el título, explicaré uno de los modelos de ML más utilizados en la práctica, e incluso, con toda humildad pienso que voy a sorprender al lector promedio y espero aportar mi granito de arena explicando en detalle cómo funciona este modelo y algunas intuiciones.

## Aprendizaje Supervisado e Intuiciones

Hace un tiempo escribí un artículo en detalle sobre como funcionan las [máquinas de soporte vectorial (SVM)]({{ site.baseurl }}{% link _posts/2023-01-30-svm-con-manzanas.markdown %}). En dicha entrada también hablé sobre aprendizaje automático en general y el problema de aprendizaje. En esta sección daré algunas intuiciones sobre este tema y el por qué se requieren heurísticas para resolver este problema.

El aprendizaje supervisado consiste en, dada un **conjunto de entrenamiento** de $N$ pares entrada-salida:

$$(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)$$

Donde cada $y_j$ se generó desde una función desconocida $y = f(x)$, descubrir una función $h$ tal que se aproxima a la función $f$.

En este caso la función $h$ es una **hipótesis**. El problema de aprendizaje es básicamente encontrar, en un espacio de hipótesis $\mathbb{H}$, la hipótesis que tenga el mejor desempeño, incluso en datos fuera del conjunto de entrenamiento. Cuando la variable dependiente tiene un conjunto de valores finito, se dice que el problema es un problema de **clasificación** (por ejemplo valores como soleado, nublado o lluvia). Por otro lado, si la variable es numérica, se dice que el problema es un problema de **regresión** (por ejemplo la temperatura en los próximos días).

<div align="center">

![noisy-pol](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/1f32e483b23335f5ea1c409dab8126cc67fca1d0/noisy-pol.png)

_Fig. 1: Ajuste de datos con ruido._

</div>

En el caso de la figura 1, tenemos dos hipótesis, una es un polinomio de grado 2, que no ajusta perfectamente los datos y un polinomio de grado 14 que ajusta perfectamente los datos. Como dato _freak_, el polinomio de grado 14 ajusta todos los datos porque se tienen 15 puntos, y por teorema de existencia y unicidad del polinomio de ajuste, existe un único polinomio $gr(p) \leq N - 1$, tal que el polinomio ajusta perfectamente los puntos. Sin embargo, se observan peaks (_Fenómeno de Runge_), por lo que la intuición dice que en muestras fuera del conjunto de entrenamiento, el rendimiento estará lejos de ser perfecto.

En general, la mejor hipótesis $h^*$ se puede definir como:

$$h^* = \underset{h \in \mathbb{H}}{\mathrm{argmax}} \ P(h|datos)$$

O por ley de Bayes:

$$h^* = \underset{h \in \mathbb{H}}{\mathrm{argmax}} \ P(datos|h)P(h)$$

Considerando las hipóetsis del ejemplo anterior, ¿cuál intuitivamente tendría una mayor probabilidad $P(h)$?

## Árboles de Decisión

Un árbol de decisión representa una función que toma como entrada un vector de valores (para distintos atributos) y retorna como "decisión" un valor de salida. Los valores de entrada y salida pueden ser discretos o continuos, para propósitos de este artículo nos concentraremos en valores discretos.

Para llegar a una decisión, se debe recorrer el árbol y se llevan a cabo una secuencia de pruebas. Cada prueba, toma un valor de un atributo dado, y se sigue una ramificación dado este valor. En la figura 2 se muestra un ejemplo de árbol de decisión.

<div align="center">

![noisy-pol](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6931c42a7ca027082b122bebb64443b1a8efd39f/arbol-ejemplo.svg)

_Fig. 2: Ejemplos de hipótesis para ajustar datos._

</div>

En este caso las variables son:

1. _Alternate_: Si es que hay un restorán alternativo en las cercanías
2. _Bar_: Si es que el restorán tiene un área cómoda para esperar (bar)
3. _Fri / Sat_: Verdadero si es Viernes o Sábado
4. _Hungry_: ¿Estamos hambrientos?
5. _Patrons_: ¿Cuánta gente hay en el restorán? (Nadie _None_, Poca gente _Some_, Lleno _Full_)
6. _Price_: Rango de precios del restorán (\\$, \\$\\$, \\$\\$\\$)
7. _Raining_: Si es que está lloviendo afuera
8. _Reservation_: ¿Tenemos reserva?
9. _Type_: Tipo de restorán (Francés, Italiano, Thai, Hamburguesas)
10. _WaitEstimate_: Tiempo estimado de espera (0-10 minutos, 10-30, 30-60, `>60`)

El **predicado objetivo** en este caso es _WillWait_, que representa la decisión de esperar o no.

### Expresividad en los árboles de decisión

En entradas previas hablé sobre [lógica proposicional]({{ site.baseurl }}{% link _posts/2022-12-29-logic.markdown %}) y [lógica de primer orden]({{ site.baseurl }}{% link _posts/2023-01-05-fallo-consistencia.markdown %}). Resulta que un árbol de decisión Booleano es equivalente a decir que el atributo objetivo es verdadero sí y sólo sí los atributos de entrada satisfacen un camino que llegue a una hoja cuyo valor sea *verdadero*. En este caso, escrito de forma proposicional, tenemos:

$$Objetivo \iff  \left( Camino_1 \lor Camino_2 \lor \ldots \right)$$

En el ejemplo de la figura 2, tenemos que el siguiente camino lleva al objetivo _true_:

$$Camino = \left( Patrons=Full \land WaitEstimate=0-10\right)$$

Para una gran gamma de problemas, el formato de árbol de decisión lleva a un resultado conciso y fácil de interpretar. Sin embargo, algunas funciones no se pueden representar de forma concisa. Por ejemplo, si tenemos una función que retorne *verdadero* cuando la mitad de los atributos son verdaderos, se requiere un árbol exponencialmente grande. En otras palabras, los árboles de decisión son una buena representación para algunas funciones y mala para otras. El lector puede hacerse la pregunta ¿Existe una representación que sea eficienet para todos los tipos de funciones? Lamentablemente la respuesta es no. Esto se puede demostrar de forma general. Consideremos el conjunto de funciones Booleanas de $n$ atributos. En este conjunto, las funciones son el número de distintas tablas de verdad que podemos escribir. Una tabla de verdad de $n$ atributos tiene $2^n$ filas. Podemos considerar la columna de "respuesta" como un número de $2^n$ bits que define a la función. Esto significa que existen $2^{2^n}$ diferentes funciones (y probablemente hay muchos más árboles, ya que una función se puede describir con múltiples árboles distintos). Esto es un número elevado, por ejemplo en el caso del problema del restorán tenemos 10 atributos, por lo tanto $2^{1024}$ o aproximadamente $10^{308}$ funciones diferentes que escoger. Por lo tanto, para buscar una solución en este espacio de hipótesis, se requieren algoritmos ingenosos.

### Inducir Árbol de Decisión a partir de ejemplos

Supongamos que recolectamos los ejemplos mostrados en la tabla 1. La pregunta es cómo inducir un árbol de decisión a partir de los datos, el árbol idealmente debe ser lo más pequeño posible y debe ser consistente. Como mencionamos anteriormente, no hay forma eficiente de encontrar dicho árbol, pues existen $2^{2^N}$ posibles modelos, lo que hace que este problema sea intratable.

<div><center>Tabla 1. Conjunto de Entrenamiento problema del restorán</center></div>

| Example | Alt | Bar | Fri | Hun | Pat  | Price | Rain | Res | Type   | Est   | WillWait |
| --------| --- | --- | --- | --- | ---- | ----- | ---- | --- | ------ | ----- | -------- |
| $x_1$   | Yes | No  | No  | Yes | Some | $$$   | No   | Yes | French | 0-10  | Yes      |
| $x_2$   | Yes | No  | No  | Yes | Full | $     | No   | No  | Thai   | 30-60 | No       |
| $x_3$   | No  | Yes | No  | No  | Some | $     | No   | No  | Burger | 0-10  | Yes      |
| $x_4$   | Yes | No  | Yes | Yes | Full | $     | Yes  | No  | Thai   | 10-30 | Yes      |
| $x_5$   | Yes | No  | Yes | No  | Full | $$$   | No   | Yes | French | `>60` | No       |
| $x_6$   | No  | Yes | No  | Yes | Some | $$    | Yes  | Yes | Italian| 0-10  | Yes      |
| $x_7$   | No  | Yes | No  | No  | None | $     | Yes  | No  | Burger | 0-10  | No       |
| $x_8$   | No  | No  | No  | Yes | Some | $$    | Yes  | Yes | Thai   | 0-10  | Yes      |
| $x_9$   | No  | Yes | Yes | No  | Full | $     | Yes  | No  | Burger | `>60` | No       |
| $x_{10}$| Yes | Yes | Yes | Yes | Full | $$$   | No   | Yes | Italian| 10-30 | No       |
| $x_{11}$| No  | No  | No  | No  | None | $     | No   | No  | Thai   | 0-10  | No       |
| $x_{12}$| Yes | Yes | Yes | Yes | Full | $     | No   | No  | Burger | 30-60 | Yes      |


Sin embargo, con algunas heurísticas simples, podemos encontrar un árbol simple (no el más pequeño) que sea consistente. Para esto se utiliza un _algoritmo voraz_ (_greedy_) y un enfoque "_divide y vencerás_" (_divide and conquer_), de tal forma que dividimos el problema en subproblemas y resolvemos de forma recursiva. El algoritmo sigue los siguientes pasos:

1. Si todos los ejemplos son de una misma clase, el algoritmo termina ya que podemos contestar a la pregunta ¿a qué clase pertenece esta muestra?
2. Si hay muestras positivas y negativas, escogemos el "mejor" atributo y dividimos las muestras.
3. Si no hay ejemplos disponibles, entonces significa que no se han observado muestras que tengan esta combinación atributo-valor, por lo que retornamos un valor por defecto (la clase más común)
4. Si no quedan atributos, pero existen ejemplos de ambas clases, significa que estos ejemplos tienen la misma descripción pero una clasificación distinta. Esto quiere decir que puede haber un error o **ruido** en los datos; ya sea porque el dominio es no deterministico o porque no podemos observar un atributo que permita dividir estas muestras en diferentes clases.

#### ¿El mejor atributo?

El algoritmo descrito anteriormente, está diseñado para reducir la profundidad del árbol de decisión. Un componente importante en este algoritmo es tener una medida de _importancia_ de atributos, de manera de separar lo antes posible las clases y de esta forma minimizar la profundidad del árbol. Utilizaremos la noción de ganancia de información, la cual es definida en términos de **entropía**.

La entropía es una medida de la incertidumbre de una variable aleatoria. La adquisición de información corresponde a una reducción en la entropía. Por ejemplo, supongamos que tenemos una moneda que al lanzarla, siempre termina en cara. La entropía de esta moneda es 0, pues la variable aleatoria sólo tiene un valor. Por otro lado, si tuviesemos una moneda cuya probabilidad de salir cara o sello sea la misma (0 o 1) se tendrá 1 bit de entropía. En general, la entropía de una variable aleatoria $V$ con valores $v_k$ con probabilidades $P(v_k)$ se define como:

$$H(V) = \displaystyle \sum_k P(v_k) \text{log}_2 \ \frac{1}{P(v_k)} = - \sum_k P(v_k)\text{log}_2 \ P(v_k)$$

Por ejemplo, la entropía del lanzamiento de una moneda (cuya probabilidad de salir cara es igual a salir sello):

$$H(moneda) = -(0.5 \ \text{log}_2 \ 0.5 + 0.5\ \text{log}_2 \ 0.5) = 1$$

Si consideramos una variable Booleana con probabilidad $q = P(v = \text{verdadero})$:

$$B(q) = - (q\ \text{log}_2 \ q + (1 - q)\ \text{log}_2 \ (1 - q))$$

Si un conjunto de entrenamiento consiste en $p$ ejemplos de la clase positiva y $n$ ejemplos de la negativa, entonces la entropía del objetivo del conjunto completo es:

$$H(\text{Objetivo}) = \displaystyle B\left(\frac{p}{p + n}\right)$$

Un atributo $A$ con $d$ valores distintos divide el conjunto de datos $E$ en los subconjuntos $E_1, E_2, \ldots E_d$. Cada subconjunto $E_k$ tiene $p_k$ ejemplos de la clase positiva y $n_k$ ejemplos de la clase negativa. Por lo que, si tomamos dicha rama en el árbol de decisión, necesitaremos $B(p_k/(p_k + n_k))$ bits de información para responder a la pregunta (objetivo). Un ejemplo del conjunto de entrenamiento escogido de forma aleatoria que tenga el valor $k$ del atributo, tiene una probabilidad $(p_k + n_k) / (p + n)$, por lo tanto la entropía remanente luego de probar el atributo $A$ es:

$$\text{Remanente}(A) = \sum_{k=1}^{d} \frac{p_k + n_k}{p + n} B\left(\frac{p_k}{p_k + n_k}\right)$$

La **Ganancia de información** del atributo $A$ es la reducción de entropía esperada:

$$Ganancia(A) = B\left( \frac{p}{p + n}\right) - Remanente(A)$$

Podemos utilizar esta función para implementar la importancia de un atributo. Por ejemplo, en los datos de la tabla 1, podemos calcular las siguientes ganancias:

$$Ganancia(Patrons) = 1 - \left[\frac{2}{12}B\left(\frac{0}{2}\right) + \frac{4}{12}B\left(\frac{4}{4}\right) + \frac{6}{12}B\left(\frac{2}{6}\right)\right] \approx 0.541 \ \text{bits}$$

$$Ganancia(Type) = 1 - \left[\frac{2}{12}B\left(\frac{1}{2}\right) + \frac{2}{12}B\left(\frac{1}{2}\right) + \frac{4}{12}B\left(\frac{2}{4}\right) + \frac{4}{12}B\left(\frac{2}{4}\right)\right] = 0 \ \text{bits}$$

En este caso, la intuición dice que $Patrons$ es un mejor atributo para dividir las muestras que $Type$.

El código en `python` para implementar estas funciones:

```python
from collections import Counter

import numpy as np

def boolean_entropy(q):
    return 0 if q in {0, 1} else -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

def gain(var, y, pos="Yes"):
    p = sum(1 if yk == pos else 0 for yk in y)
    b = boolean_entropy(p / len(y))
    r = 0
    counts = Counter(var)
    for vk in counts:
        pk = sum(y == pos and v == vk for v, y in zip(var, y))
        r += counts[vk] / len(y) * boolean_entropy(pk / counts[vk])

    return b - r
```

#### Implementación árbol de decisión

Ahora tenemos las herramientas para implementar un árbol de decisión, primero necesitamos algunas funciones:

```python
from collections import defaultdict
from typing import Dict, List, Union


def plurality_value(y):
    return Counter(y).most_common(1)[0][0]

def all_same_class(examples, out):
    y = examples[out][0]
    for y_ in examples[out]:
        if y_ != y:
            return False
    return True


assert all_same_class({"x": [1, 2, 3], "y": [1, 1, 1]}, "y")
assert not all_same_class({"x": [1, 2, 3], "y": [2, 1, 1]}, "y")
```

Por otro lado también necesitamos representar el árbol de decisión:

```python
class DecisionTree:
    def __init__(self, label):
        self._label = label
        self._children = {}

    def add_children(self, edge, children: "DecisionTree") -> None:
        self._children[edge] = children

    def add_childrens(self, children: Dict[str, Union[str, "DecisionTree"]]):
        for edge, child in children.items():
            self.add_children(edge, child)


    # Hack para hacernos la vida más simple
    def is_terminal_node(tree) -> bool:
        return isinstance(tree, DecisionTree) == 0
```

Finalmente podemos implementar la clase `DecisionTreeClassifier` que tendrá los métodos `fit` y `predict` para entrenar y predecir respectivamente. El método que implementa el algoritmo descrito previamente es `_decision_tree_learning`, en donde seguimos un algoritmo voraz, escogiendo en cada paso el atributo con ganancia máxima y diviendo el conjunto de datos en base a los valores de este atributo.

Para implementar el método `predict`, necesitamos tener el árbol de decisión y recorrer las ramas hasta llegar a un nodo terminal que tendrá la clase correspondiente a la nueva muestra.

```python
class DecisionTreeClassifier:
    def __init__(self):
        self._is_trained = False

    def _decision_tree_learning(
        self, examples, attributes, parent_examples, out
    ) -> DecisionTree:
        if len(examples) == 0:
            return plurality_value(parent_examples[out])

        if all_same_class(examples, out):
            return examples[out][0]

        if len(attributes) == 0:
            return plurality_value(examples[out])

        a = attributes[
            np.argmax([gain(examples[a], examples[out]) for a in attributes])
        ]
        tree = DecisionTree(a)
        for vk in set(parent_examples[a]):
            exs = defaultdict(list)
            idx = [i for i, v in enumerate(examples[a]) if v == vk]
            new_attributes = [attribute for attribute in attributes if attribute != a]
            for attribute in new_attributes + [out]:
                for i in idx:
                    exs[attribute].append(examples[attribute][i])
            subtree = decision_tree_learning(exs, new_attributes, examples, out)
            tree.add_children(vk, subtree)
        return tree

    def fit(self, x_train, y_train) -> None:
        attributes = [a for a in x_train]
        if len(y_train) != 1:
            raise Exception(f"y_train should be a class list {y_train}")
        out_name = list(y_train.keys())[0]
        if isinstance(x_train, dict):
            examples = x_train.copy()
            examples.update(y_train)

        self._y_default = plurality_value(y_train[out_name])
        self._dt = self._decision_tree_learning(
            examples, attributes, examples, out_name
        )
        self._is_trained = True

    def predict(self, x_test):
        try:
            tree = self._dt
            if not (isinstance(tree, dict) or isinstance(tree, DecisionTree)):
                return tree
            val = x_test[tree._label]
            output = tree._children[val]
            return output if is_terminal_node(output) else predict(output, x_test)
        # Caso en que observamos un valor no visto en entrenamiento
        except KeyError:
            return self._y_default

    @classmethod
    def load_from_tree(cls, tree) -> "DecisionTreeClassifier":
        new_clf = cls()
        clf._is_trained = True
        clf._dt = tree
        return clf
```

Si entrenamos el modelo con los datos de la tabla 1, los cuales pondré a continuación en el formato en que los recibe la implementación:

<details><summary>Click para ver conjunto de datos</summary>

```python
x = {
    "Alternate": [
        "Yes",
        "Yes",
        "No",
        "Yes",
        "Yes",
        "No",
        "No",
        "No",
        "No",
        "Yes",
        "No",
        "Yes",
    ],
    "Bar": [
        "No",
        "No",
        "Yes",
        "No",
        "No",
        "Yes",
        "Yes",
        "No",
        "Yes",
        "Yes",
        "No",
        "Yes",
    ],
    "FriSat": [
        "No",
        "No",
        "No",
        "Yes",
        "Yes",
        "No",
        "No",
        "No",
        "Yes",
        "Yes",
        "No",
        "Yes",
    ],
    "Hungry": [
        "Yes",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
    ],
    "Patrons": [
        "Some",
        "Full",
        "Some",
        "Full",
        "Full",
        "Some",
        "None",
        "Some",
        "Full",
        "Full",
        "None",
        "Full",
    ],
    "Price": [
        "$$$",
        "$",
        "$",
        "$",
        "$$$",
        "$$",
        "$",
        "$$",
        "$",
        "$$$",
        "$",
        "$",
    ],
    "Rain": [
        "No",
        "No",
        "No",
        "Yes",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "No",
        "No",
        "No",
    ],
    "Reservation": [
        "Yes",
        "No",
        "No",
        "No",
        "Yes",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "No",
    ],
    "Type": [
        "French",
        "Thai",
        "Burger",
        "Thai",
        "French",
        "Italian",
        "Burger",
        "Thai",
        "Burger",
        "Italian",
        "Thai",
        "Burger",
    ],
    "WaitEstimate": [
        "0-10",
        "30-60",
        "0-10",
        "10-30",
        ">60",
        "0-10",
        "0-10",
        "0-10",
        ">60",
        "10-30",
        "0-10",
        "30-60",
    ],
    "WillWait": [
        "Yes",
        "No",
        "Yes",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "No",
        "No",
        "Yes",
    ],
}
```
</details>

Obtenemos el siguiente árbol de decisión:

<div align="center">

![fit-tree](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d4fecb0cc148416b9be244a3c55c63b595bccd14/arbol-entrenado.svg)

_Fig. 3: Árbol de decisión inducido de los ejemplos de la tabla 1._

</div>

Se observa un comportamiento curioso, el algoritmo encontró un patrón utilizando la variable $Type$, siendo que sabemos que el árbol generador de los datos no tiene una rama utilizando este atributo. Por ello, se debe tener cuidado al interpretar un árbol de decisión, pues la estructura del mismo dependerá totalmente del conjunto de datos utilizado para generarlo.

Para evaluar el desempeño de un algoritmo de aprendizaje, podemos utilizar una **curva de aprendizaje**. En la figura 4, se muestra la curva de aprendizaje teniendo 100 muestras y variando el tamaño del conjunto de entrenamiento desde 1 a 99:

<div align="center">

![learning-curve](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/a0b4ea1aab6f702dfa2ce6a15078297c53440908/learning-curve.png)

_Fig. 4: Curva de aprendizaje en 100 muestras generadas aleatoriamente. Cada punto se obtuvo del promedio de 20 experimentos._

</div>

Para los curiosos, el código para generar la curva se muestra a continuación:

<details><summary>Click para ver código generador de curva de aprendizaje</summary>

```python
def generate_random_sample():
    x = {}
    x["Alternate"] = random.choice(["Yes", "No"])
    x["Hungry"] = random.choice(["Yes", "No"])
    x["Bar"] = random.choice(["Yes", "No"])
    x["FriSat"] = random.choice(["Yes", "No"])
    x["Patrons"] = random.choice(["None", "Some", "Full"])
    x["Price"] = random.choice(["$", "$$", "$$$"])
    x["Rain"] = random.choice(["Yes", "No"])
    x["Reservation"] = random.choice(["Yes", "No"])
    x["Type"] = random.choice(["French", "Italian", "Thai", "Burger"])
    x["WaitEstimate"] = random.choice(["0-10", "10-30", "30-60", ">60"])
    # Bottom Up
    bar = DecisionTree("Bar")
    bar.add_childrens({"No": "No", "Yes": "Yes"})

    reservation = DecisionTree("Reservation")
    reservation.add_childrens({"No": bar, "Yes": "Yes"})

    fri_sat = DecisionTree("FriSat")
    fri_sat.add_childrens(
        {
            "No": "No",
            "Yes": "Yes",
        }
    )

    alternate = DecisionTree("Alternate")
    alternate.add_childrens(
        {
            "No": reservation,
            "Yes": fri_sat,
        }
    )

    raining = DecisionTree("Rain")
    raining.add_childrens(
        {
            "No": "No",
            "Yes": "Yes",
        }
    )

    alternate_2 = DecisionTree("Alternate")
    alternate_2.add_childrens(
        {
            "No": "Yes",
            "Yes": raining,
        }
    )

    hungry = DecisionTree("Hungry")
    hungry.add_childrens({"No": "Yes", "Yes": alternate_2})

    wait_estimate = DecisionTree("WaitEstimate")
    wait_estimate.add_childrens(
        {">60": "No", "30-60": alternate, "10-30": hungry, "0-10": "Yes"}
    )

    patrons = DecisionTree("Patrons")
    patrons.add_childrens(
        {
            "None": "No",
            "Some": "Yes",
            "Full": wait_estimate,
        }
    )
    clf = DecisionTreeClassifier.load_from_tree(patrons)
    x["WillWait"] = clf.predict(x)
    return x


def generate_samples(n=100):
    x = []
    for i in range(n):
        xs = generate_random_sample()
        x.append(xs)
    return x


def generate_accuracy_data(n_trials=20):
    training_size = []
    accuracies = []
    for train_size in range(1, 99):
        training_size.append(train_size)
        avg_accuracy = 0
        for _ in range(n_trials):
            dataset = generate_samples()
            train = dataset[0:train_size]
            test = dataset[train_size:]
            x_train = defaultdict(list)
            y_train = {"WillWait": []}
            for sample in train:
                for col in sample:
                    if col != "WillWait":
                        x_train[col].append(sample[col])
                    else:
                        y_train[col].append(sample[col])
            clf = DecisionTreeClassifier()
            clf.fit(x_train, y_train)
            hits = 0
            for data in test:
                y_test = data.pop("WillWait")
                hits += 1 if y_test == clf.predict(data) else 0
            avg_accuracy += hits / len(test)
        accuracies.append(avg_accuracy / n_trials)
    return training_size, accuracies


train_size, accuracies = generate_accuracy_data()
```

Para graficar:

```python
import matplotlib.pyplot as plt

plt.plot(train_size, accuracies, "-b")
plt.xlabel("Training size")
plt.ylabel("Avg Accuracy")
plt.title("Curva de aprendizaje 100 muestras y 20 ejecuciones")
```
</details>

#### Generalización y sobreajuste

Como se observó anteriormente, el algoritmo de aprendizaje puede aprender patrones inexistentes en los datos, dependiendo de las muestras que se entreguen en la fase de entrenamiento. Esto puede generar caminos que no tienen sentido para predecir valores. Este problema se conoce como **sobreajuste**.

Para evitar esto, se deben aplicar técnicas como **podado del árbol**, en la cual eliminamos nodos que no son relevantes. Esto se hace luego de armar completamente el árbol.

La pregunta es ¿cómo determinar qué nodos son relevantes? Para este caso se pueden hacer **pruebas de significancia estadística**. Un método común para podar árboles de decisión es el [podado $\chi^2$](https://web.cs.ucdavis.edu/~vemuri/classes/ecs271/Decision%20Tree%20Rules%20&%20Pruning.htm). En este caso, como el lector podrá deducir, se utiliza un test $\chi^2$ para determinar la relevancia de un nodo.

## Ejemplo práctico: Predecir lluvia en Seattle

Como al momento de escribir esta entrrada, me encuentro viviendo en Seattle, qué mejor que probar nuestro algoritmo en el conjunto [_Did it rain in Seattle?_ de Kaggle](https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017).

Este conjunto de datos tiene variables continuas, pero para transformarlas en variables discretas, debemos generar particiones del atributo (por ejemplo `Temperatura <= 90`). Para encontrar de forma eficiente estas particiones, lo que hacemos es:

* Ordenamos los datos respecto al atributo de interés
* Mantenemos un conteo de las clases positivas y negativas en cada lado de la partición
* Si encontramos un punto a particionar, calculamos la ganancia de información
* Escogemos el punto de máxima ganancia

Para cargar el conjunto de datos:

```python
import csv
from pathlib import Path

RESOURCES_PATH = Path("./resources")

dataset = []
with open(RESOURCES_PATH / "seattleWeather_1948-2017.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        some_na = False
        for key, val in row.items():
            if val == "NA":
                some_na = True
        if not some_na:
            dataset.append(row)
```

Dividir en conjunto de entrenamiento y de prueba. Por ahora "asumiremos" la mentira de que la lluvia no depende de la estacionalidad (sólo por simplicidad):

```python
import random

random.shuffle(dataset)

train_idx = int(len(dataset) * 0.8)
train = dataset[0:train_idx]
test = dataset[train_idx:]
```

```python
print("Training set size:", len(train))
print("Test set size:", len(test))
print(train[0])
```

```
Training set size: 20438
Test set size: 5110
{'DATE': '1955-11-30', 'PRCP': '0.38', 'TMAX': '47', 'TMIN': '40', 'RAIN': 'TRUE'}
```

Encontramos las particiones y calculamos los nuevos atributos para el conjunto de entrenamiento:

```python
def get_train_split_point(train, col):
    total_pos = 0
    total_neg = 0
    for data in train:
        if data["RAIN"] == "TRUE":
            total_pos += 1
        else:
            total_neg += 1
    b = boolean_entropy(total_pos / (total_pos + total_neg))
    pos_running_sum = 0
    neg_running_sum = 0
    sorted_train = sorted(train, key=lambda x: float(x[col]))
    split_points = {}
    for i, data in enumerate(sorted_train):
        if i + 1 == len(train):
            break
        if data["RAIN"] == "TRUE":
            pos_running_sum += 1
        else:
            neg_running_sum += 1

        lo = float(data[col])
        hi = float(sorted_train[i + 1][col])
        if lo < hi:
            split_point = round(lo + hi / 2, 2)
            right_pos = total_pos - pos_running_sum
            right_neg = total_neg - neg_running_sum
            r = (pos_running_sum + neg_running_sum) / len(train) * boolean_entropy(
                pos_running_sum / (pos_running_sum + neg_running_sum)
            ) + (right_pos + right_neg) / len(train) * boolean_entropy(
                right_pos / (right_pos + right_neg)
            )
            split_points[split_point] = b - r
    return max(split_points, key=split_points.get)


split_points = {}
for col in ["PRCP", "TMAX", "TMIN"]:
    split_points[col] = get_train_split_point(train, col)

x_train = defaultdict(list)
y_train = {"RAIN": []}
for data in train:
    for col in split_points:
        x_train[f"{col} <= {split_points[col]}"].append(
            float(data[col]) <= split_points[col]
        )
    y_train["RAIN"].append(data["RAIN"])
```

Entrenamos un clasificador:

```python
clf_seattle = DecisionTreeClassifier()
clf_seattle.fit(x_train, y_train)
```

Calculamos _accuracy_:

```python
hits = 0
for data in test:
    y_test = data["RAIN"]
    x_test = {}
    for col in split_points:
        x_test[f"{col} <= {split_points[col]}"] = float(data[col]) <= split_points[col]

    hits += 1 if y_test == clf_seattle.predict(x_test) else 0
```

```python
print(
    "Baseline Acc:",
    Counter(list(map(lambda x: x["RAIN"], test))).most_common(1)[0][1] / len(test),
)
print("Decision Tree Acc:", hits / len(test))
```

```
Baseline Acc: 0.5720156555772994
Decision Tree Acc: 0.9655577299412916
```

Se observa que el modelo es simple pero obtiene un mejor resultado de accuracy, significativamente mayor que la línea base. Sólo de curiosidad, graficamos el árbol:

<div align="center">

![seattle-ejemplo](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d4fecb0cc148416b9be244a3c55c63b595bccd14/arbol-ejemplo-seattle.svg)

_Fig. 5: Árbol de decisión para ejemplo de predecir lluvia en Seattle._

</div>

Notar que si utilizamos la regla obtenida por el árbol de decisión:

```python
hits = 0
for data in test:
    y_test = "TRUE" if float(data["PRCP"]) > 0.01 else "FALSE"
    hits += 1 if y_test == data["RAIN"] else 0

print("Accuracy PRCP > 0.01:", hits / len(test))
```

```
Accuracy PRCP > 0.01: 0.9655577299412916
```

Lo que tiene sentido dentro de los cálculos y la intuición. Finalmente, destacar que la parte más costosa en entrenar árboles de decisión, es el procesamiento de variables numéricas y encontrar los puntos de corte.

Un punto gracioso, es que en el conjunto de datos, si `PRCP >= 0.01` obtenemos la clasificación perfecta:

```python
hits = 0
for data in test:
    y_test = "TRUE" if float(data["PRCP"]) >= 0.01 else "FALSE"
    hits += 1 if y_test == data["RAIN"] else 0

print("Accuracy PRCP >= 0.01:", hits / len(test))
```

```
Accuracy PRCP >= 0.01: 1.0
```

Finalmente, para los curiosos que quieran saber cómo grafiqué los árboles, creé un método que recorre los árboles de forma recursiva y los pasa a formato `dot` para graficarlos con `graphviz`:

```python
def to_dot(tree) -> str:
    def internal_tree(tree, i=0):
        if is_terminal_node(tree):
            return ""

        color = "black:white"
        node_name = tree._label.split()[0]
        s = f'{node_name}_{i} [shape=square, color="{color}", peripheries=1 label="{tree._label}"];\n'
        non_terminal_child_val = None
        for val, child in tree._children.items():
            if not is_terminal_node(child):
                child_name = child._label.split()[0]
                s += f'{node_name}_{i} -> {child_name}_{i + 1} [label="{val}"];\n'
                s += internal_tree(child, i + 1)
            else:
                child_name = child.split()[0]
                s += f'{child_name}_{i} [style=filled, peripheries=1, label="{child}"];\n'
                s += f'{node_name}_{i} -> {child_name}_{i} [label="{val}"];\n'

        return s

    # Se puede hacer con f-strings pero no se renderiza bien en el artículo
    # Si lo hago así
    return r"digraph G {" + internal_tree(tree) + r"}"
```

## Conclusiones

* Los árboles de decisión pueden verse como la búsqueda de un objetivo dado un conjunto de caminos posibles.
* Encontrar el árbol perfecto (consistente y más pequeño) es un problema intratable, por lo que se requieren heurísticas.
* Los árboles de decisión pueden aprender patrones que no necesariamente corresponden a la realidad en la generación de los datos.
* Se debe tener un mecanismo de "podado" o reducción de profunidad/anchura del árbol para evitar sobre-ajuste
* La operación más costosa en la inducción un árbol de decisión es procesar datos numéricos, por lo que se deben encontrar formas eficientes de lograr esta partición

El ejercicio del día, para seguir con la temática, será [Crear punteros hacia la derecha en un árbol binario](https://github.com/dpalmasan/code-challenges/issues/50).
