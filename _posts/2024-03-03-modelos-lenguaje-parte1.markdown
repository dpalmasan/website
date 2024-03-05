---
layout: post
title:  "Entendiendo los Modelos del Lenguaje (Parte 1)"
date:   2024-03-03 15:30:00 -0400
categories: probability algorithms ai
---

<div align="center">

![ai](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/78da1f25307f7487f5e6287bcd6640096de80b58/ai.png)

</div>

# Introducción

Iniciaré una serie de artículos, corta igual que la última que hice sobre _De vuelta a lo Básico_, donde expliqué probabilidades y conceptos básicos de modelos probabilísticos. Si quieres revisar dichos artículos, dejo los siguientes enlaces:

* [_De vuelta a lo básico (Parte 1)_]({{ site.baseurl }}{% link _posts/2024-02-25-de-vuelta-a-lo-basico.markdown %})
* [_De vuelta a lo básico (Parte 2)_]({{ site.baseurl }}{% link _posts/2024-02-26-variables-aleatorias.markdown %})
* [_De vuelta a lo básico (Parte 3)_]({{ site.baseurl }}{% link _posts/2024-02-27-hipotesis-correlacion.markdown %})

Esta nueva serie de artículos intentaré _aclarar_ y desmenuzar los modelos de lenguaje, entendiendo primero qué es un modelo de lenguaje, y algunos temas de lingüística que pueden ser interesantes. Todo esto, con el fin de transferir conocimiento y evitar _mal entendimiento_ de algunos conceptos.

# Modelos de Lenguaje

## Intuición

En términos simbólicos, definimos un vocabulario como un conjunto de símbolos, que típicamente representan letras, dígitos, fonemas e incluso palabras. Entonces, este conjunto que al menos tiene un elemento se definiría como:

$$\Sigma = \left\\{w_1, w_2, \ldots w_N\right\\}$$

En el caso de un lenguaje hablado, podría pensarse como que el i-ésimo elemento $w_i$ es una palabra del lenguaje. Para entender mejor esta idea, consideremos el siguiente vocabulario:

$$\Sigma = \left\\{\text{"."}, \text{"perro"}, \text{"come"}, \text{"El"}\right\\}$$

Por otro lado, supongamos que vivimos en un mundo donde las oraciones tienen únicamente $4$ palabras. En este caso, podríamos representar las oraciones como una secuencia $\left(w_1, w_2, w_3, w_4\right)$. Para este mundo ficticion, tenemos que existen $4^4 = 256$ posibles combinaciones:

```py
from itertools import product

vocab = ("El", "perro", "come", ".")
sample_space = list(product(vocab, repeat=len(vocab)))

for perm in sample_space:
    print(perm)
```

<details><summary>Ver posibles combinaciones</summary>


```
('El', 'El', 'El', 'El')
('El', 'El', 'El', 'perro')
('El', 'El', 'El', 'come')
('El', 'El', 'El', '.')
('El', 'El', 'perro', 'El')
('El', 'El', 'perro', 'perro')
('El', 'El', 'perro', 'come')
('El', 'El', 'perro', '.')
('El', 'El', 'come', 'El')
('El', 'El', 'come', 'perro')
('El', 'El', 'come', 'come')
('El', 'El', 'come', '.')
('El', 'El', '.', 'El')
('El', 'El', '.', 'perro')
('El', 'El', '.', 'come')
('El', 'El', '.', '.')
('El', 'perro', 'El', 'El')
('El', 'perro', 'El', 'perro')
('El', 'perro', 'El', 'come')
('El', 'perro', 'El', '.')
('El', 'perro', 'perro', 'El')
('El', 'perro', 'perro', 'perro')
('El', 'perro', 'perro', 'come')
('El', 'perro', 'perro', '.')
('El', 'perro', 'come', 'El')
('El', 'perro', 'come', 'perro')
('El', 'perro', 'come', 'come')
('El', 'perro', 'come', '.')
('El', 'perro', '.', 'El')
('El', 'perro', '.', 'perro')
('El', 'perro', '.', 'come')
('El', 'perro', '.', '.')
('El', 'come', 'El', 'El')
('El', 'come', 'El', 'perro')
('El', 'come', 'El', 'come')
('El', 'come', 'El', '.')
('El', 'come', 'perro', 'El')
('El', 'come', 'perro', 'perro')
('El', 'come', 'perro', 'come')
('El', 'come', 'perro', '.')
('El', 'come', 'come', 'El')
('El', 'come', 'come', 'perro')
('El', 'come', 'come', 'come')
('El', 'come', 'come', '.')
('El', 'come', '.', 'El')
('El', 'come', '.', 'perro')
('El', 'come', '.', 'come')
('El', 'come', '.', '.')
('El', '.', 'El', 'El')
('El', '.', 'El', 'perro')
('El', '.', 'El', 'come')
('El', '.', 'El', '.')
('El', '.', 'perro', 'El')
('El', '.', 'perro', 'perro')
('El', '.', 'perro', 'come')
('El', '.', 'perro', '.')
('El', '.', 'come', 'El')
('El', '.', 'come', 'perro')
('El', '.', 'come', 'come')
('El', '.', 'come', '.')
('El', '.', '.', 'El')
('El', '.', '.', 'perro')
('El', '.', '.', 'come')
('El', '.', '.', '.')
('perro', 'El', 'El', 'El')
('perro', 'El', 'El', 'perro')
('perro', 'El', 'El', 'come')
('perro', 'El', 'El', '.')
('perro', 'El', 'perro', 'El')
('perro', 'El', 'perro', 'perro')
('perro', 'El', 'perro', 'come')
('perro', 'El', 'perro', '.')
('perro', 'El', 'come', 'El')
('perro', 'El', 'come', 'perro')
('perro', 'El', 'come', 'come')
('perro', 'El', 'come', '.')
('perro', 'El', '.', 'El')
('perro', 'El', '.', 'perro')
('perro', 'El', '.', 'come')
('perro', 'El', '.', '.')
('perro', 'perro', 'El', 'El')
('perro', 'perro', 'El', 'perro')
('perro', 'perro', 'El', 'come')
('perro', 'perro', 'El', '.')
('perro', 'perro', 'perro', 'El')
('perro', 'perro', 'perro', 'perro')
('perro', 'perro', 'perro', 'come')
('perro', 'perro', 'perro', '.')
('perro', 'perro', 'come', 'El')
('perro', 'perro', 'come', 'perro')
('perro', 'perro', 'come', 'come')
('perro', 'perro', 'come', '.')
('perro', 'perro', '.', 'El')
('perro', 'perro', '.', 'perro')
('perro', 'perro', '.', 'come')
('perro', 'perro', '.', '.')
('perro', 'come', 'El', 'El')
('perro', 'come', 'El', 'perro')
('perro', 'come', 'El', 'come')
('perro', 'come', 'El', '.')
('perro', 'come', 'perro', 'El')
('perro', 'come', 'perro', 'perro')
('perro', 'come', 'perro', 'come')
('perro', 'come', 'perro', '.')
('perro', 'come', 'come', 'El')
('perro', 'come', 'come', 'perro')
('perro', 'come', 'come', 'come')
('perro', 'come', 'come', '.')
('perro', 'come', '.', 'El')
('perro', 'come', '.', 'perro')
('perro', 'come', '.', 'come')
('perro', 'come', '.', '.')
('perro', '.', 'El', 'El')
('perro', '.', 'El', 'perro')
('perro', '.', 'El', 'come')
('perro', '.', 'El', '.')
('perro', '.', 'perro', 'El')
('perro', '.', 'perro', 'perro')
('perro', '.', 'perro', 'come')
('perro', '.', 'perro', '.')
('perro', '.', 'come', 'El')
('perro', '.', 'come', 'perro')
('perro', '.', 'come', 'come')
('perro', '.', 'come', '.')
('perro', '.', '.', 'El')
('perro', '.', '.', 'perro')
('perro', '.', '.', 'come')
('perro', '.', '.', '.')
('come', 'El', 'El', 'El')
('come', 'El', 'El', 'perro')
('come', 'El', 'El', 'come')
('come', 'El', 'El', '.')
('come', 'El', 'perro', 'El')
('come', 'El', 'perro', 'perro')
('come', 'El', 'perro', 'come')
('come', 'El', 'perro', '.')
('come', 'El', 'come', 'El')
('come', 'El', 'come', 'perro')
('come', 'El', 'come', 'come')
('come', 'El', 'come', '.')
('come', 'El', '.', 'El')
('come', 'El', '.', 'perro')
('come', 'El', '.', 'come')
('come', 'El', '.', '.')
('come', 'perro', 'El', 'El')
('come', 'perro', 'El', 'perro')
('come', 'perro', 'El', 'come')
('come', 'perro', 'El', '.')
('come', 'perro', 'perro', 'El')
('come', 'perro', 'perro', 'perro')
('come', 'perro', 'perro', 'come')
('come', 'perro', 'perro', '.')
('come', 'perro', 'come', 'El')
('come', 'perro', 'come', 'perro')
('come', 'perro', 'come', 'come')
('come', 'perro', 'come', '.')
('come', 'perro', '.', 'El')
('come', 'perro', '.', 'perro')
('come', 'perro', '.', 'come')
('come', 'perro', '.', '.')
('come', 'come', 'El', 'El')
('come', 'come', 'El', 'perro')
('come', 'come', 'El', 'come')
('come', 'come', 'El', '.')
('come', 'come', 'perro', 'El')
('come', 'come', 'perro', 'perro')
('come', 'come', 'perro', 'come')
('come', 'come', 'perro', '.')
('come', 'come', 'come', 'El')
('come', 'come', 'come', 'perro')
('come', 'come', 'come', 'come')
('come', 'come', 'come', '.')
('come', 'come', '.', 'El')
('come', 'come', '.', 'perro')
('come', 'come', '.', 'come')
('come', 'come', '.', '.')
('come', '.', 'El', 'El')
('come', '.', 'El', 'perro')
('come', '.', 'El', 'come')
('come', '.', 'El', '.')
('come', '.', 'perro', 'El')
('come', '.', 'perro', 'perro')
('come', '.', 'perro', 'come')
('come', '.', 'perro', '.')
('come', '.', 'come', 'El')
('come', '.', 'come', 'perro')
('come', '.', 'come', 'come')
('come', '.', 'come', '.')
('come', '.', '.', 'El')
('come', '.', '.', 'perro')
('come', '.', '.', 'come')
('come', '.', '.', '.')
('.', 'El', 'El', 'El')
('.', 'El', 'El', 'perro')
('.', 'El', 'El', 'come')
('.', 'El', 'El', '.')
('.', 'El', 'perro', 'El')
('.', 'El', 'perro', 'perro')
('.', 'El', 'perro', 'come')
('.', 'El', 'perro', '.')
('.', 'El', 'come', 'El')
('.', 'El', 'come', 'perro')
('.', 'El', 'come', 'come')
('.', 'El', 'come', '.')
('.', 'El', '.', 'El')
('.', 'El', '.', 'perro')
('.', 'El', '.', 'come')
('.', 'El', '.', '.')
('.', 'perro', 'El', 'El')
('.', 'perro', 'El', 'perro')
('.', 'perro', 'El', 'come')
('.', 'perro', 'El', '.')
('.', 'perro', 'perro', 'El')
('.', 'perro', 'perro', 'perro')
('.', 'perro', 'perro', 'come')
('.', 'perro', 'perro', '.')
('.', 'perro', 'come', 'El')
('.', 'perro', 'come', 'perro')
('.', 'perro', 'come', 'come')
('.', 'perro', 'come', '.')
('.', 'perro', '.', 'El')
('.', 'perro', '.', 'perro')
('.', 'perro', '.', 'come')
('.', 'perro', '.', '.')
('.', 'come', 'El', 'El')
('.', 'come', 'El', 'perro')
('.', 'come', 'El', 'come')
('.', 'come', 'El', '.')
('.', 'come', 'perro', 'El')
('.', 'come', 'perro', 'perro')
('.', 'come', 'perro', 'come')
('.', 'come', 'perro', '.')
('.', 'come', 'come', 'El')
('.', 'come', 'come', 'perro')
('.', 'come', 'come', 'come')
('.', 'come', 'come', '.')
('.', 'come', '.', 'El')
('.', 'come', '.', 'perro')
('.', 'come', '.', 'come')
('.', 'come', '.', '.')
('.', '.', 'El', 'El')
('.', '.', 'El', 'perro')
('.', '.', 'El', 'come')
('.', '.', 'El', '.')
('.', '.', 'perro', 'El')
('.', '.', 'perro', 'perro')
('.', '.', 'perro', 'come')
('.', '.', 'perro', '.')
('.', '.', 'come', 'El')
('.', '.', 'come', 'perro')
('.', '.', 'come', 'come')
('.', '.', 'come', '.')
('.', '.', '.', 'El')
('.', '.', '.', 'perro')
('.', '.', '.', 'come')
('.', '.', '.', '.')
```
</details>

En este modelo del mundo, estas combinaciones definirían el **espacio muestral** de las oraciones a poder formar. Supongamos que queremos calcular la probabilidad de la oración $\text{"El perro come."}$. Tenemos 4 **variables aleatorias** $w_1, w_2, w_3, w_4$, sea $P_1$ tal que:

$$
\begin{align}
P_1 &= P\left(w\_1=\text{"El"}, w\_2=\text{"perro"}, w\_3=\text{"come"}, w\_4=\text{"."}\right) \\\\
&= \frac{1}{256} \\\\
& \approx 0.004
\end{align}
$$

Supongamos que muestreamos $P_2$ tal que:

$$
\begin{align}
P_2 &= P\left(w\_1=\text{"."}, w\_2=\text{"come"}, w\_3=\text{"perro"}, w\_4=\text{"El"}\right) \\\\
&= \frac{1}{256} \\\\
& \approx 0.004
\end{align}
$$

Esto ocurre, porque asumimos que $P(w_1, w_2, w_3, w_4)$ es una distribución uniforme. En el mundo real, si consideramos el lenguaje español, esperaríamos que $P_1 \gt P_2$, además, idealmente $P_2 = 0$, ya que según las reglas gramaticales, no sería parte del lenguaje. A este modelo estadístico de calcular la probabilidad de una secuencia de palabras se le conoce como **modelo de lenguaje**.

Finalmente, aclarar que el vocabulario $\Sigma$ en un lenguaje natural es mucho más grande que el del ejemplo donde la cardinalidad (cantidad de elementos) $|\Sigma| = 4$. Observamos que la cantidad de combinaciones posibles crece exponencialmente con el largo $N$ de texto a considerar y la cardinalidad $|\Sigma|$, que representa la cantidad de posibles asignaciones para las variables $w_1, w_2, \ldots, w_N$.

## ¿Predicción de la siguiente palabra?

La mayoría de la gente, cuando se habla de modelos del lenguaje habla sobre _predicción de la siguiente palabra_. Aclaremos esto, ya que recuerdo una vez que cierta persona (ingeniero de software), mencionaba los _LLM_ (_Large Language Models_) y hablaba sobre predicción de la siguiente palabra. Sin embargo, cuando le preguntaron el por qué, o la justificación, la persona no pudo responder y divagó por varias ramas sin responder a la pregunta.

Si recordamos la probabilidad condicional, y escribiendo considerando $P(A\cap B) = P(A, B)$ tenemos:

$$P(A|B) = \displaystyle \frac{P(A, B)}{P(B)}$$

Consideremos un lenguaje cuyas oraciones son de largo $N$, entonces, el modelo de lenguaje estaría dado por:

$$P(w_1, w_2, \ldots w_N)$$

Utilizando la definición de probabilidad condicional:


$$P(w_1, w_2, \ldots w_N) = P(w_N|w_1, w_2, \ldots w_{N-1})P(w_1, w_2, \ldots w_{N-1}) $$

Extendiendo:

$$
\begin{align}
P(w_1, w_2, \ldots w_N) &= P(w_N|w_1, w_2, \ldots w_{N-1})P(w_{N-1}|w_1, w_2, \ldots w_{N-2}) P(w_1, w_2, \ldots w_{N-2}) \\\\\\\\
&= \displaystyle \prod_{i=1}^{N} P(w_i|w_1 \ldots w_{i-1})
\end{align}
$$

En este caso, intentamos calcular la probabilidad conjunta $P(w_1, w_2, \ldots w_N)$ como una cadena de probabilidades condicionales. Se eligieron las probabilidades condicionales, de manera de considerar la variable $w_i$ dadas todas las variables $w_i, \ldots\ w_{i-1}$, de ahí que decimos que intentamos calcular la probabilidad $P(w_i|w_1 \ldots w_{i-1})$, que se interpreta como calcular la probabilidad de la palabra $w_i$ dadas todas las palabras previas.

## ¿Estimar $P(w_1, w_2, \ldots w_N)$?

Sabemos que $P(w_1, w_2, \ldots w_N)$ es desconocida, por lo tanto necesitamos estimarla. Para ello, lo que se hace en la práctica es construir un repositorio gigante de textos, y encontrar las probabilidades condicionales para poder calcular la probabilidad conjunta que representa al lenguaje. El lector podrá observar, ¿Qué me garantiza que podré recolectar un conjunto de oraciones que contenga todas las oraciones posibles?. No hay garantía, y de hecho, podría darse el caso de que el repositorio no contenga una oración válida y que a dicha oración se le asigne una probabilidad igual a cero. Para evitar esto, se hacen aproximaciones. Si definimos $w_1, w_2 \ldots w_{i-1}$ como el _contexto_ para la palabra $w_i$, entonces podemos escoger no utilizar el contexto completo y por ejemplo considerar únicamente las $n$ palabras previas. En este caso, estaríamos calculando $P(w_i|w_{i-n}, \ldots w_{i - 1})$. Este tipo de secuencias se conocen como _n-gramas_.

En mi artículo [_Un poco de NLP básico (no un tutorial de pytorch/tensor flow)_]({{ site.baseurl }}{% link _posts/2023-01-08-nlp-intro.markdown %}), explico cómo se construye un modelo de n-gramas simple, utilizando una _cadena de Markov_. Sin embargo, en dicho artículo describo una posible solución al problema. Actualmente, con los avances en semi-conductores y en las ciencias de la computación, ahora existen modelos que son capaces de considerar más contexto, e incluso ponderarlo, por ejemplo considerar ciertas palabras más relevantes que otras para el caso de predecir la siguiente palabra. Por ejemplo si consideramos la oración: "_El gato se sentó en_", podríamos decir que es probable que sea "_el sofá_". Sin embargo, si tuviesemos el contexto completo: "_Estabamos haciendo una parrillada en la terraza y vimos que el gato se sentó en_", en este caso es más probable decir "_el techo_" que "_el sofá_".

Finalmente, al estimar $P(w_1, w_2, \ldots w_N)$ se obtiene un _modelo generativo_ con el cual se pueden muestrear oraciones/texto que incluso no se vio en el conjunto de entrenamiento.

En siguientes artículos, explicaré de forma simple cómo funciona uno de los modelos que ha causado mayor sensacionalismo recientemente, hablo de GPT (_Generative Pre-Trained Transformers_), e idealmente estos artículos podrán aclarar algunos temas que lamentablemente ciertos _influencers_ viven publicando y que son engañosos.

# Conclusiones

* Un modelo de lenguaje es simplemente un modelo probabilístico que opera con secuencias.
* La probabilidad conjunta de una oración puede calcularse como un producto de probabilidades condicionales.
* El _predecir la siguiente palabra_ es simplemente una forma de estimar $P(w_1, w_2, \ldots w_N)$.
* Este modelo probabilístico es generativo, lo que permite muestrear de $P(w_1, w_2, \ldots w_N)$ incluso oraciones no vistas en el repositorio de datos utilizado para estimar dicha distribución de probabilidad.
* En la historia han existido distintos enfoques/trucos/técnicas para calcular $P(w_1, w_2, \ldots w_N)$. En los recientes desarrollos GPT es el estado del arte en este problema.
