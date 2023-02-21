---
layout: post
title:  "Árboles de Decisión, lo que probablemente no sabías"
date:   2023-02-18 11:10:03 -0400
categories: python algorithms classification machine-learning
---

Probablemente, si trabajas con _Machine Learning_, haz notado que uno de los modelos más usados es _Extreme Gradient Boosting Decision Trees_ o _XGBoost_, o algún modelo similar. La práctica que funciona en general, es tomar los datos, insertarlos en la juguera y probablemente obtener un resultado. Las _API_ y frameworks disponibles hacen que la tarea no sea complicada. Si bien, en general en la práctica, el problema generalmente se resuelve teniendo los datos correctos, existen algunos casos en que incluso teniendo una gran disponibilidad de datos a mano, los modelos no tengan buen desempeño. En este caso, el problema puede deberse a múltiples fuentes, sin embargo cuando hay que _"entrar a picar"_, a veces el problema real está en no entender los modelos ni sus fundamentos.

En esta entrada, como dice el título, explicaré uno de los modelos de ML más utilizados en la práctica, e incluso, con toda humildad pienso que voy a sorprender al lector promedio y espero aportar mi granito de arena explicando en detalle cómo funciona este modelo y algunas intuiciones. 

## Aprendizaje Supervisado e Intuiciones

Hace un tiempo escribí un artículo en detalle sobre como funcionan las [máquinas de soporte vectorial (SVM)]({{ site.baseurl }}{% link _posts/2023-01-30-svm-con-manzanas.markdown %}). En dicha entrada también hablé sobre aprendizaje automático en general y el problema de aprendizaje. En esta sección daré algunas intuiciones sobre este tema y el por qué se requieren heurísticas para resolver este problema.

El aprendizaje supervisado consiste en, dada un **conjunto de entrenamiento** de $N$ pares entrada-salida:

$$(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)$$

Donde cada $y_j$ se generó desde una función desconocida $y = f(x)$, descubrir una función $h$ tal que se aproxima a la función $f$.

En este caso la función $h$ es una **hipótesis**. El problema de aprendizaje es básicamente encontrar, en un espacio de hipótesis $\mathbb{H}$, la hipótesis que tenga el mejor desempeño, incluso en datos fuera del conjunto de entrenamiento. Cuando la variable dependiente tiene un conjunto de valores finito, se dice que el problema es un problema de **clasificación** (por ejemplo valores como soleado, nublado o lluvia). Por otro lado, si la variable es numérica, se dice que el problema es un problema de **regresión** (por ejemplo la temperatura en los próximos días).

<div align="center">

![noisy-pol](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/1f32e483b23335f5ea1c409dab8126cc67fca1d0/noisy-pol.png)

_Fig 1: Ajuste de datos con ruido_

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

_Fig 2: Árbol de decisión para decidir si esperar o no una mesa en un restorán_

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
10. _WaitEstimate_: Tiempo estimado de espera (0-10 minutos, 10-30, 30-60, >60)

El **predicado objetivo** en este caso es _WillWait_, que representa la decisión de esperar o no.

### Expresividad en los árboles de decisión

En entradas previas hablé sobre [lógica proposicional]({{ site.baseurl }}{% link _posts/2022-12-29-logic.markdown %}) y [lógica de primer orden]({{ site.baseurl }}{% link _posts/2023-01-05-fallo-consistencia.markdown %}). Resulta que un árbol de decisión Booleano es equivalente a decir que el atributo objetivo es verdadero sí y sólo sí los atributos de entrada satisfacen un camino que llegue a una hoja cuyo valor sea *verdadero*. En este caso, escrito de forma proposicional, tenemos:

$$Objetivo \iff  \left( Camino_1 \lor Camino_2 \lor \ldots \right)$$

En el ejemplo de la figura 2, tenemos que el siguiente camino lleva al objetivo _true_:

$$Camino = \left( Patrons=Full \land WaitEstimate=0-10\right)$$

Para una gran gamma de problemas, el formato de árbol de decisión lleva a un resultado conciso y fácil de interpretar. Sin embargo, algunas funciones no se pueden representar de forma concisa. Por ejemplo, si tenemos una función que retorne *verdadero* cuando la mitad de los atributos son verdaderos, se requiere un árbol exponencialmente grande. En otras palabras, los árboles de decisión son una buena representación para algunas funciones y mala para otras. El lector puede hacerse la pregunta ¿Existe una representación que sea eficienet para todos los tipos de funciones? Lamentablemente la respuesta es no. Esto se puede demostrar de forma general. Consideremos el conjunto de funciones Booleanas de $n$ atributos. En este conjunto, las funciones son el número de distintas tablas de verdad que podemos escribir. Una tabla de verdad de $n$ atributos tiene $2^n$ filas. Podemos considerar la columna de "respuesta" como un número de $2^n$ bits que define a la función. Esto significa que existen $2^{2^n}$ diferentes funciones (y probablemente hay muchos más árboles, ya que una función se puede describir con múltiples árboles distintos). Esto es un número elevado, por ejemplo en el caso del problema del restorán tenemos 10 atributos, por lo tanto $2^{1024}$ o aproximadamente $10^{308}$ funciones diferentes que escoger. Por lo tanto, para buscar una solución en este espacio de hipótesis, se requieren algoritmos ingenosos.



