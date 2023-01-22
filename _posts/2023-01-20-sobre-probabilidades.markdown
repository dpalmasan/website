---
layout: post
title:  "IA, Incertidumbre, razonamiento probabilístico"
date:   2023-01-21 17:15:03 -0400
categories: python algorithms
---
En un [post previo]({{ site.baseurl }}{% link _posts/2023-01-05-fallo-consistencia.markdown %}), hablé sobre lógica, en especial lógica de primer orden ya que en [otra entrada]({{ site.baseurl }}{% link _posts/2022-12-29-logic.markdown %}) hablé sobre razonamiento lógico utilizando lógica proposicional. También, mencioné mi biblioteca open-source [py-logic](https://github.com/dpalmasan/py-logic) que soporta lógica proposicional y un subconjunto de lógica de primer orden (cláusulas de Horn).

Sin embargo, hay escenarios como el mostrado en la figura 1, en que el razonamiento lógico sólo va a responder preguntas como: _"Es posible que haya un Wumpus en `(1, 2)`, `(2, 3)` y `(0, 1)`"_. Sin embargo, no podemos cuantificar que tan probable es que haya un Wumpus en cualquiera de las habitaciones a explorar. Por otro lado, intuitivamente la habitación más peligrosa es la `(1, 2)`, pues tiene adyacentes dos habitaciones con odor. Aquí es donde entra el razonamiento probabilístico. 

<div align="center">

![Uncertainity](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/9a570e812a9d055f1ca27beefafa7bb83a5652af/wumpus-evidence-example.png)

_Fig 1: Agente IA tomando una decisón._

</div>

# IA, Probabilidades y Probabilidad Condicional

¿Por qué mencionar IA? Para refrescar la memoria, un problema de IA típico es dado un objetivo, buscar la secuencia de acciones que llevan a ese objetivo. Un caso típico es un problema de clasificación, donde se tiene una variable $y \in Y$, donde $Y$ es un conjunto finito de valores, y estamos interesados en dado un conjunto de hipótesis $\mathbb{H}$ una hipótesis $h \in \mathbb{H}$, que dada una evidencia $e$ entregue un valor $y \in Y$. Dicha hipótesis es arbitraria, pero idealmente estamos interesados en una hipótesis que cumpla con algún objetivo (generalmente minimizar la esperanza del error fuera de muestra). En algunos casos nos interesa encontrar $P(y|e)$, ya que nos interesa estimar la certeza del valor dada la evidencia.

## Distribución conjunta y Probabilidad condicional

Como repaso de distribución conjunta, consideremos un mundo en el que tenemos 3 variables, $A: \\{\text{perro, gato}\\}$, $N: \\{1, 3, 5\\}$, $B: \\{\text{True, False}\\}$. La distribución conjunta, debiese asignar una probabilidad para todas las asignaciones de variables posibles en este mundo (posibles eventos):

```
{B: False, A: 'cat', N: 1}
{B: False, A: 'cat', N: 2}
{B: False, A: 'cat', N: 3}
{B: False, A: 'dog', N: 1}
{B: False, A: 'dog', N: 2}
{B: False, A: 'dog', N: 3}
{B: True, A: 'cat', N: 1}
{B: True, A: 'cat', N: 2}
{B: True, A: 'cat', N: 3}
{B: True, A: 'dog', N: 1}
{B: True, A: 'dog', N: 2}
{B: True, A: 'dog', N: 3}
```

El caso con variables continuas requiere otra representación (funciones de densidad de probabilidad), pero en esencia es la misma intuición. En general, no siempre estamos interesados en enumerar todos los posibles mundos. Por ejemplo, supongamos que observamos que la variable $A = dog$. En este caso, tenemos sub-mundos, y podemos enumerar nuevamente los posibles eventos en este sub-mundo:

```
{A: 'dog', B: False, N: 1}
{A: 'dog', B: False, N: 2}
{A: 'dog', B: False, N: 3}
{A: 'dog', B: True, N: 1}
{A: 'dog', B: True, N: 2}
{A: 'dog', B: True, N: 3}
```

En el segundo caso, tenemos la _evidencia_ de que $A = dog$. Se puede definir la _probabilidad condicional_ como la probabilidad de que un evento ocurra dada una evidencia. En este ejemplo, se escribiría como $P(B, N| A)$. Para dos eventos $A$, $B$, la probabilidad condicional se define como:

$$P(A|B) = \displaystyle \frac{P(A, B)}{P(B)}$$

De forma más general, supongamos que tenemos la variable $X$, las variables $E$ que representan la evidencia y $e$ los valores de estas variables, e $Y$ las variables restanets (o variables ocultas), entonces:

$$P(X|e) = \alpha P(X, e) = \alpha \sum_y P(X, e, y)$$

En este caso $P(X, e, y)$ es un subconjunto de probabilidades de la distribución conjunta.

Podemos estimar o hacer inferencias sobre ciertas variables en un mundo, dada una evidencia. Sin embargo, este problema es intratable. Como se puede observar, si consideramos $n$ variables Booleanas, se requiere calcular $2^n$ eventos, lo que se vuelve impráctico.

## Agente probabilístico y mundo del Wumpus

Volviendo al ejemplo de la figura 1, supongamos que queremos estimar la probabilidad de que la habitación en la coordenada `(0, 1)` tiene un Wumpus, dado que sabemos que hay odor en las habitaciones `(0, 2)` y `(1, 3)`. Sean: 

* $s = \neg s_{03} \land s_{02} \land s_{13}$
* $conocido = \neg w_{03} \land \neg w_{02} \land \neg w_{13}$
* $desconocido = $ el resto de las variables (y sus asignaciones posibles)

En este caso, lo que queremos calcular es:

$$P(P_{13} | conocido, s) = \alpha \sum_{desconocido} P(P_{13}, desconocido, conocido, s)$$

En esencia, necesitamos conocer todas las asignaciones posibles para $desconocido$. Si el mundo fuera de 12 habitaciones, tendríamos 12 variables, cada una con dos asignaciones, por lo tanto $2^{12} = 4096$ términos. El lector atento podrá observar que las habitaciones desconocidas no deberían incidir en el cálculo de la probabilidad, y además que los odores una vez conocidos, son independientes de todas las variables. Haciendo algunas simplificaciones, se puede llegar a una expresión simplificada y con menos términos.

Supongamos que tenemos el mundo mostrado en la figura 2

<div align="center">

![wsolv](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/9a570e812a9d055f1ca27beefafa7bb83a5652af/wumpus-prob-solvable.png)

_Fig 2: Configuración arbitraria para mundo del Wumpus._

</div>

El agente probabilístico tomaría el siguiente conjunto de acciones:


<div align="center">

![wsolv](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/f8ddbb114528741199a24aa73302411029cba820/wumpus-succ.gif)

_Fig 3: Agente probabilístico completa el puzzle._

</div>

Se observa que el agente intenta buscar la habitación "más segura", en este caso, la que tenga la mínima probabilidad de algún peligro (Wumpus o precipicio). El agente evita a toda costa visitar la habitación `(1, 2)`. Ahora veamos qué ocurre en el caso en que tenemos una configuración un poco más compleja:

<div align="center">

![wsolv](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/f8ddbb114528741199a24aa73302411029cba820/wumpus-nonprob-solvable.png)

_Fig 4: Configuración en que la certeza no cuadra con la realidad._

</div>

Como se puede observar en la figura 5, el agente muere:

<div align="center">

![wsolv](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/f8ddbb114528741199a24aa73302411029cba820/wumpus-fail.gif)

_Fig 5: Agente falla en resolver el laberinto._

</div>

En este caso el agente falla, debido a que desafortunadamente, la habitación segura era la que menos probabilidades tenía de contener alguna trampa. Básicamente se observó un __falso negativo__, en este caso un agente ficticio pierde en un juego, en la vida real, puede ser pérdida de mucho dinero.

Esto nos prueba que no hay una solución que resuelva todos los escenarios.

# Notas Finales

* Pensé incluir redes Bayesianas, pero eso lo dejaré para otro post.
* El código del agente lo pueden encontrar en mi github, en particular [aquí](https://github.com/dpalmasan/wumpus-ai/blob/main/player.py#L226) (Obs: No he documentado ni pulido el repo aún).
* Incluso en casos en que pareciera que el resultado es obvio, podrían darse escenarios/mundos en que no.

# Problema de Código

El problema de código de esta entrada es: [Llenado de Imagen](https://github.com/dpalmasan/code-challenges/issues/33). La solución se encuentra en el mismo repositorio.
