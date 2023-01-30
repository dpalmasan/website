---
layout: post
title:  "SVM explicado"
date:   2023-01-28 11:10:03 -0400
categories: python algorithms classification machine-learning
---

Esta entrada nuevamente tendrá un poco de clasificación de textos y también intentaré explicar en términos simples en qué consisten _máquinas de soporte vectorial_ (_Support Vector Machines_ o SVM).

Aprovechando la introducción, responderé un par de preguntas que me llegan recurrentemente, espero poder ayudar:

* ¿Cómo aprendiste sobre IA y ML?
* ¿Cómo te has movido en tantos roles (e.g. Data Engineering, Backend)
* ¿Seguiste alguna ruta en particular?

Las respuestas están al final de esta entrada.

## Máquinas de Soporte Vectorial

Primero se debe entender qué problema intentan resolver las máquinas de soporte vectorial. Volvamos a lo básico: en un *problema de clasificación* se tiene una función $h: \mathbb{R} \longrightarrow L$, donde $L$ es un conjunto de etiquetas o _clases_. En particular, en clasificación binaria se tienen dos clases, y para propósitos de este artículo $L = \\{-1, +1\\}$.

En general, en los cursos introductorios de _Machine Learning_ se ve revisa como primera unidad el _aprendizaje del perceptrón_, que en esencia, es una función que buscar un hiperplano que particione el espacio vectorial y separe las clases positivas de las negativas. Cuando es posible hacer esto, se habla de _separabilidad lineal_. En la figura 1, se muestra una animación del aprendizaje del perceptrón.

<div align="center">

![perceptron](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/54e9984d25b566dbfe8f54e8bcdce1c2dde17ca9/animacion-perceptron.gif)

_Fig 1: Aprendizaje del perceptrón en un conjunto de datos separable linealmente_

</div>

El lector atento, podrá notar que pueden existir infinitos hiperplanos que logren particionar un espacio logrando separar las clases. Sin embargo, debe existir un hiperplano que sea "mejor" que el resto bajo algún criterio. De aquí sale el concepto de _margen_. El margen es en esencia, la distancia de separación entre el hiperplano y los puntos más cercanos al hiperplano (por cada clase). Intuitivamente, el "mejor" hiperplano, podría ser el que maximice este margen, ya que estaría logrando la máxima separación lineal entre las clases. Las máquinas de soporte vectorial intentan encontrar este hiperplano.

<div align="center">

![svm-lin-sep](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/ee85e9acce5687f22b4419ca8b04f1d925cacd43/svm-data-points.png)

_Fig 2: Conjunto de datos separable linealmente ¿Cuál es el méjor hiperplano?_

</div>

### Intuición y entrenando un modelo SVM

Supongamos que tenemos un punto $x \in \mathbb{R}$. Consideremos la función:

$$f: \mathbb{R}^n \longrightarrow \mathbb{R}$$

$$x \longrightarrow \langle w, x \rangle + b$$

Donde $w$ y $b$ son parámetros de $f$. El hiperplano que separa ambas clases en el problema de clasificación binaria está dado por:

$$\\{ x \in \mathbb{R}: f(x) = 0\\}$$

Para calcular la clase a la que pertenece un dato nuevo $x_{test}$, se calcular el valor $f(x_{test})$ y se clasifica, por ejemplo con la clase $+1$ si $f(x_{test}) \geq 0$ o $-1$ en caso contrario.

Al momento de entrenar, se requiere que los ejemplos de la clase positiva se encuentren en el lado positivo del hiperplano, $\langle w, x_n \rangle + b \geq 0$ cuando $y_n = +1 $ y la clase negativa en el otro lado $\langle w, x_n \rangle + b < 0$ cuando $y_n = -1$ Generalmente se utiliza la función _signo_ para escribir de forma compact esto, es decir: $y_n = sign(\langle w, x_n \rangle + b)$

### Margen y problema de optimización

Supongamos que el punto $x_a$ es el punto más cercano al hiperplano $\langle w, x_a \rangle + b > 0$ ($w$ es un vector ortogonal al hiperplano). Consideremos tamnién una escala, tal que $\langle w, x_a \rangle + b = 1$, la razón es que si consideramos la escala en la que están los datos (escala de $x_n$), si cambiamos la unidad de medida o valores de $x$ (por ejemplo $10x_n$), la distancia al hiperplano también va a cambiar. Consideremos una escala tal que $\langle w, x_a \rangle + b = 1$.

<div align="center">

![svm-lin-sep](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/43fb5f53da0f7fe31e4110c998580388af70a16b/svm.png)

_Fig 3: Intuición margen máximo_

</div>

Como se muestra en la figura 3, la proyección $x_a'$ de $x_a$ en el plano, está dada por:

$$x_a = x_a' - \displaystyle r \frac{w}{||w||}$$

Donde $r$ es la distancia al hiperplano y consideramos la dirección del vector $w$. Ya que $x_a'$ se encuentra en el hiperplano, entonces $\langle w, x_a' \rangle + b = 0$, si resolvemos:

$$ \langle w, x_a - \displaystyle r \frac{w}{||w||} \rangle + b = 0$$

$$\langle w, x_a\rangle + b - \langle w, \displaystyle r \frac{w}{||w||} \rangle = 0$$

$$1 - r\displaystyle \frac{w^Tw}{||w||} = 1 - r\displaystyle \frac{||w||^2}{||w||} = 0$$

$$r = \displaystyle \frac{1}{||w||}$$

En esencia, para maximizar el margen $r$, debemos maximizar $\displaystyle \frac{1}{||w||}$, esto sería equivalente a minimizar $||w||$, luego:

$$
\begin{aligned}
\min_{w,b} \quad & \frac{1}{2}||w||^2 \\\\\\\\\\
\textrm{s.t.} \quad & y_n (\langle w, x_n \rangle + b) \geq 1\\\\
\end{aligned}
$$

Donde $\frac{1}{2}||w||^2$ no cambia al valor optimo de $w$, se escribe sólo por conveniencia (al calcuar la derivada parcial con respecto a $w$ para encontrar el óptimo, queda más simple de analizar), y las restricciones están relacionadas al ajuste de datos.

Este problema de optimización se resuelve con _programación cuadrática_. La forma estándar que consideran como entrada los algoritmos de programación cuadrática, es la siguiente:

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^TPx + q^Tx\\\\\\
\textrm{s.t.} \quad & Gx \leq h\\\\
& Ax = b
\end{aligned}
$$

Utilizando la biblioteca `cvxopt` en python, y escribiendo el programa en la forma estándar, tenemos el siguiente código en `Python` (asumiendo que $x$ tiene dos dimensiones)

```python
from cvxopt import matrix, solvers


P = matrix(np.eye(3))
q = matrix(np.zeros((3, 1)))
G = matrix(-y * np.concatenate([np.ones((len(y), 1)), x], axis=1))
h = matrix(-np.ones(len(y)))
sol = solvers.qp(P, q, G, h)
wsol = np.array(sol['x'])
```

Al graficar la solución, se obtiene:

<div align="center">

![svm-lin-sep-opt](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/43fb5f53da0f7fe31e4110c998580388af70a16b/svm-linear-best-margin.png)

_Fig 4: Hiperplano que optimiza el margen_

</div>

### Interpretación como cáscara convexa

Una interpretación del hiperplano que maximiza el margen, es considerar que cada clase pertence a una cáscara convexa (_convex hull_). Una cáscara convexa, es básicamente un polígono que cubre los límites de un conjunto de puntos (figura 5).

<div align="center">

![svm-convex-humm](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/43fb5f53da0f7fe31e4110c998580388af70a16b/svm-convex-hull.png)

_Fig 5: Cáscara convexa_

</div>

En esencia, una SVM, maximiza la distancia entre las cáscaras convexas de cada clase.

### Margen Duro vs Margen Suave

El problema de optimización previo, asume que las clases son separables linealmente, es decir, la solución no admite soluciones fuera del margen. Esto se conoce como _margen duro_. Una alternativa, es, en lugar de tener $y_n (\langle w, x_n\rangle) + b \geq 1$, se introduce una variable de flexibilidad $\xi_n$, tal que se permiten ciertos errores de clasificación $y_n (\langle w, x_n\rangle) + b \geq 1 - \xi_n$

$$
\begin{aligned}
\min_{w,b,\xi} \quad & \frac{1}{2}w^{t}w+C\sum_{n=1}^{N}{\xi_{n}} \\\\
\textrm{s.t.} \quad & y_{n}(\langle w, x_{n} \rangle+b)\geq 1 - \xi_{n}\\\\
  &\xi_n\geq0    \\\\
\end{aligned}
$$

### Derivación de la forma Dual

El problema de optimización original es maximizar el margen tal que ambas clases son separables. Hasta ahora hemos definido el problema como un problema de minimización. Sin embargo, podemos resolver el problema de optimización utilizando _multiplicadores de Lagrange_. Tomando la ecuación del margen suave, y utilizando los multiplicadores de lagrange, tenemos:

$$
\begin{aligned}
\min_{w,b,\xi, \alpha, \gamma} \quad & \frac{1}{2}w^{t}w \\\\
& -\sum_{n=1}^{N}\alpha_n(y_{i}(\langle w, x_{n} \rangle+b)+\xi_{i}-1)\\\\
  & -C\sum_{n=1}^{N} \gamma_n\xi_n \\\\
\end{aligned}
$$

Donde $\alpha_n$ y $\gamma_n$ son multiplicadores de Lagrange. Para encontrar el mínimo, debemos calcular gradiente, es decir las derivadas parciales de la función de Lagrange, con respecto a cada variable a optimizar e igualar a cero.

$$
\begin{aligned}
\displaystyle \frac{\partial \mathcal{L}}{\partial w} = w^T - \sum_{n=1}^{N} \alpha_n y_n x_n^T = 0 \Rightarrow w = \sum_{n=1}^{N} \alpha_n y_n x_n\\\\
\displaystyle \frac{\partial \mathcal{L}}{\partial b} = \sum_{n=0}^{N} \alpha_n y_n = 0\\\\
\displaystyle \frac{\partial \mathcal{L}}{\partial \xi_n} = C - \alpha_n - \gamma_n = 0\\\\
\end{aligned}
$$

Trabajando las ecuaciones y considerando las restricciones, llegamos al siguiente problema de optimización:

$$
\begin{aligned}
\min_{\alpha} \quad & \frac{1}{2} \sum_{i=1}^{N}\sum_{j=1}^{N} y_iy_j\alpha_i\alpha_j\langle x_i, x_j\rangle - \sum_{n=1}^{N} \alpha_n \\\\
\textrm{s.t.} \quad & \sum_{n=1}^{N} \alpha_n y_n = 0 \\\\
  & 0 \leq \alpha_n \leq C \\\\
\end{aligned}
$$

En este caso los vectores $\alpha$ se les conoce vectores de soporte, ya que soportan el margen de máxima separabilidad. De aquí sale el nombre de _máquinas de soporte vectorial_.

### Separabilidad Lineal

Existen casos (me disculpo por el ejemplo clásico) en que no existe un hiperplano que separe ambas clases, como se muestra en la figura 6:

<div align="center">

![svm-convex-humm](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6a3cce4767da7ee9fc6e2050cf4929bf8401feed/svm-non-separable.png)

_Fig 6: Conjunto de datos no separable linealmente_

</div>

En este caso, ya que los datos fueron generados en este ejemplo, conocemos el líimite de decisión, el cual está marcado de color negro. En este caso, conocemos una transformación $\phi$ tal que $\phi (x) = z$. En la figura 7, se muestra el conjunto de datos transformado al nuevo sistema de coordenadas. Y luego podemos encontrar el máximo margen resolviendo el problema de optimización. El hiperplano se muestra en la figura 7.

<div align="center">

![svm-nonsep-trans](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6a3cce4767da7ee9fc6e2050cf4929bf8401feed/svm-transformed-space.png)

_Fig 7: Conjunto de datos no separable linealmente en un espacio $z$ donde es separable_

</div>

### El truco del kernel

Supongamos que tenemos un conjunto de datos donde cada registro consiste en dos características $x = (x_1, x_2)$. Lo que podemos hacer es por ejemplo considerar todas las coordenadas posibles (ejemplo: polinomio cuadrado) $z = (x_1, x_2, x_1^2, x_2^2, x_1x_2)$. Si quisieramos agregar más características, este problema se vuelve intratable (ejemplo: 50 características ¿cuántas posibles combinaciones hay para un polinomio cuadrado? ¿y para uno de grado $Q$?).

Si observamos la última derivación del problema de optimización, la solución depende de los vectores de soporte (cantidad de registros) y no de la cantidad de características. En el único momento en que utilizamos las características, es cuando calculamos $\langle x_i, x_j\rangle$. Esto quiere decir, si tuvieramos la función $\phi$, sólo nos interesa el producto interior $\langle \phi(x_i), \phi(x_j)\rangle$. En el caso estándar $\langle x_i, x_j\rangle$ es un kernel lineal. Podemos definir por ejemplo un kernel polinomial:

$$K(x_i, x_j) = (1 + x_i^Tx_j)^Q$$

Por ejemplo en la figura 8, se muestra el límite de decisión derivado utilizando un kernel polinomial con $Q=2$.

<div align="center">

![svm-nonsep-kernel](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6a3cce4767da7ee9fc6e2050cf4929bf8401feed/svm-kernel-trick.png)

_Fig 8: Límite de decisión utilizando kernel polinomial_

</div>

Puede notarse que el límite de decisión encontrado es el límite de decisión manualmente generado, con la diferencia que no tuvimos que aplicar directamente la transformación $\phi$ a $x$. La idea de un kernel es en esencia llevar los datos de un espacio a otro espacio de mayor dimensionalidad, donde las clases sean linealmente separables (idealmente).

### Realizando Predicciones



## Conclusiones

* En algunos casos se puede clasificar textos dado su contenido, se pueden hacer simples estimaciones como utilizar la frecuencia de ocurrencia de cada término.



## Respuestas a las preguntas de la introducción

<details><summary>Click para ver mi respuesta</summary>
<div align="center">

![meme](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6a3cce4767da7ee9fc6e2050cf4929bf8401feed/meme-wey.jpg)

</div>
Supongo que estudiando, he invertido mucho en libros &nbsp; :smile: &nbsp;, no tengo una respuesta ni una ruta lamentablemente.
</details>
