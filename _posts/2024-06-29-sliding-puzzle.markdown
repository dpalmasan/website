---
layout: post
title:  "Resolviendo puzzles con algoritmos"
date:   2024-06-29 18:00:00 -0400
categories: algorithms ai
---

# Introducción

Era el año 2014, y me había titulado de ingeniería civil eléctrica en la Universidad de Concepción (¡grande mi alma mater!). También estaba en conflicto pues venía saliendo de una práctica profesional que no me gustó para nada (área de mantención de una planta papelera). No fue una buena experiencia y estaba dudando si era eso a lo que quería dedicarme en mi vida profesional. Por otro lado, siempre me gustaron las clases teóricas en ingeniería, las simulaciones y meter algo de código; debo confesar que mi única experiencia programando fue un curso de un semestre en el lenguaje `C`, y el resto `MATLAB`.

En ese mismo año alguien publica en facebook que darán un curso "gratis" de electromagnetismo en la plataforma edX (que yo desconocía), así que me inscribí; lamentablemente dicho curso ya no está disponible, era el curso _8.02x - MIT Physics II: Electricity and Magnetism_ (pero si no saben la historia **funaron** al profesor Walter Lewin; por lo que el curso ya no está dispnible en edX; hay que reconocer que las clases eran espectaculares). Debo decir que disfruté bastante el curso y hasta me uní a grupos de facebook y participaba activamente en los foros de edX ayudando y respondiendo dudas de los problemas; ¿qué pasó? Me volví adicto a los cursos en línea y me puse a tomar varios cursos (más relacionados a la ingeniería eléctrica). Luego, gracias a una de las personas que estaba en uno de esos cursos de física donde yo participaba, encontré el mítico curso [CS50x: Introduction to Computer Science de Harvard](https://www.edx.org/learn/computer-science/harvard-university-cs50-s-introduction-to-computer-science). Fue dicho curso en donde aprendí lo introductorio de las Ciencias de la Computación y varios temas de programación (en especial en C) que desconocía. Con mi "experiencia" programando, no pensé que sería un curso desafiante, pero lo fue. Lo interesante es que cada tarea, tenía dos versiones: La tarea para aprobar y la versión "hacker". Yo logré resolver las versiones "hacker" de todos los problemas, excepto de uno: [The game of fifteen](https://docs.cs50.net/problems/fifteen/fifteen.html). No entraré en detalles, pero en esencia es un puzzle, la tarea era implementar el puzzle como un juego de terminal y la versión hacker era implementar un comando "GOD" que resolviera el puzzle de forma automática. Este post es mi redención e intento de solución de este problema.

# ¿$N^2 - 1$ Puzzle?

El puzzle de $N^2 - 1$ es una grilla de $N\times N$ que consiste en $N^2 - 1$ piezas que deben moverse a una configuración objetivo, que consiste en tener todas las piezas ordenadas de forma ascendente, como se muestra en la figura 1.

<div align="center">

![puzzle-def](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/1c42887ede70e37523fb601075a0e8cbf2d8a892/8puzzle-def.png)

_Fig 1: Ejemplo de puzzle de 8-piezas y su configuración objetivo._

</div>

Para mover las piezas, hay una restricción, estas deben deslizarse, por ello es que la grilla tiene un espacio libre como se muestra en la figura. Cabe destacar que no todas las configuraciones posibles tienen solución (no vamos a demostrar esto hoy jeje), por lo tanto, para lo que sigue del artículo, asumiremos que existe solución para la configuración inicial.

¿Cómo buscar una solución? Pensando que cada puzzle representa un **estado**, entonces al realizar cualquier **acción** habrá un cambio de estado. Considerando múltiples posibles acciones, construímos lo que es un **espacio de estados** que se muestra en la figura 2.

<div align="center">

![puzzle-search](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/1c42887ede70e37523fb601075a0e8cbf2d8a892/8puzzle-search.png)

_Fig 2: Ejemplo de espacio de estados para puzzle de 8-piezas dada una configuración inicial._

</div>

Y aquí viene la gran pregunta de todo problema de inteligencia artificial ¿Cómo explorar el espacio de estados para encontrar una solución? ¿Cómo encontrar la "mejor" solución? Existen varios algoritmos de búsqueda, y para no alargar tanto este artículo, prefiero no mencionar todos. Sin embargo, para tener una intuición, consideremos el problema de encontrar el camino menos costoso entre una ciudad y otra.

## Problemas de Búsqueda

En la figura 3, se muestra un diagrama de las ciudades `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`. Cada flecha indica la conexión entre las ciudades el valor en la flecha indica el costo de viajar entre dos ciudades. Si qusieramos encontrar el camino menos costoso entre `A` y `H`, tenemos que usar dicho costo para realizar la búsqueda.

<div align="center">

![puzzle-sol](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/b8153a6577ac00558bcf42e2f9580e2c3f3b4820/city-example.png)

_Fig 3: Ejemplo de camino menos costoso entre ciudades._

</div>

Comencemos:

1. Estoy en la ciudad `A`, puedo hacer `(A->B, 1)`, `(A->C, 3)`, `(A->E, 3)`
2. Me cuesta menos el primer camino, luego puedo hacer `(A->B->C, 2)`, `(A->B->H, 15)` `(A->C, 3)`, `(A->E, 3)`
3. Ahora podría elegir `(A->B->C, 2)`, luego tengo las opciones: `(A->B->C->D, 8)`, `(A->B->H, 15)` `(A->C, 3)`, `(A->E, 3)`
4. Ahora elijo `(A->C, 3)`, luego: `(A->C->D, 9)`, `(A->B->C->D, 8)`, `(A->B->H, 15)`, `(A->E, 3)`
5. Ahora elijo `(A->E, 3)`, luego: `(A->E->F, 4)`, `(A->C->D, 9)`, `(A->B->C->D, 8)`, `(A->B->H, 15)`
6. Elijo `(A->E->F, 4)`, luego: `(A->E->F->G, 5)`, `(A->C->D, 9)`, `(A->B->C->D, 8)`, `(A->B->H, 15)`
7. Elijo `(A->E->F->G, 5)`, luego: `(A->E->F->G->H, 6)`, `(A->C->D, 9)`, `(A->B->C->D, 8)`, `(A->B->H, 15)`
8. Finalmente llegué a `H` con el recorriendo el siguiente camino: `A->E->F->G->H`. Que es el menos costoso, cuyo coste es 6.

Como podemos observar, este es un enfoque completamente automatizable. De hecho, el algoritmo que acabamos de hacer se conoce como **búsqueda de costo uniforme**. Como se pudo observar, esta búsqueda fue no informada, ya que reaccionabamos al costo de las acciones y sólo utilizamos información vista (pasada). ¿Qué pasaría si tuviesemos información adicional? Por ejemplo ¿Qué pasaría si supieramos exactamente cuánto falta para llegar al destino? Supongamos que existe una función $f$, tal que $f(s) = n$, donde $s$ es una ciudad y $n$ es el costo mínimo que existe desde $s$ hasta `H`. Por ejemplo si $s = \text{B}$ entonces `f(B) = 15`. Intentemos nuevamente el algoritmo, con esta información:

1. Estoy en la ciudad `A`, puedo hacer `(A->B, 1 + 15)`, `(A->C, 3 + 14)`, `(A->E, 3 + 3)`
2. Tomo `(A->E, 3)`, puedo hacer: `(A->B, 1 + 15)`, `(A->C, 3 + 14)`, `(A->E->F, 4 + 2)`
3. Tomo `(A-E->F, 4)`, puedo hacer: `(A->B, 1 + 15)`, `(A->C, 3 + 14)`, `(A->E->F->G, 5 + 1)`
4. Tomo `(A->E->F->G, 5)` puedo hacer: `(A->B, 1 + 15)`, `(A->C, 3 + 14)`, `(A->E->F->G->H, 5)`
5. Llego a `H` tomando `(A->E->F->G->H, 6)`, camino menos costoso: 6.

La diferencia, es que teniendo información pude descartar soluciones y reducir el espacio de búsqueda. Este algoritmo se conoce como `A*` (o _A estrella_). Ahora, generalmente no conocemos esta función $f$. Generalmente, podemos tener una aproximación $h$. Si dicha aproximación es tal que $h$ nunca sobre-estima $f$ (es decir, $h(s) \leq f(s)$) entonces se puede garantizar que la solución a encontrar será óptima y la heurística se dice **admisible**. En caso contrario, no hay garantías. Sin embargo, en la mayoría de los casos en la práctica, un óptimo local es suficiente y si $h$ ayuda a reducir el espacio de búsqueda entonces no es necesaria la admisibilidad.

Finalmente, cabe destacar que la heurística trivial $h(s) = 0$ es admisible, pero no entrega información, por lo que es equivalente a hacer búsqueda no informada. En general, más cerca se encuentre $h$ de $f$, mayor es la cantidad de soluciones a descartar.

## Definiendo una heurística admisible para el problema del puzzle de 8 piezas

Una estrategia para definir heurísticas admisibles es "relajar" las restricciones del problema. Por ejemplo, en el caso del puzzle podemos considerar tomar cada pieza y ponerla en el lugar que le corresponda. En este caso, la heurística sería "número de piezas fuera de lugar", que sería equivalente a calcular la distancia Hamming. Otra heurística, podría ser deslizar las piezas por encima de las otras, en este caso la distancia sería el número de "deslizamientos de cada pieza", esta se conoce como distancia Manhattan, ya que dada una posición objetivo $(i_o, j_o)$ y una posición inicial $(i_i, j_i)$, entonces la distancia sería: $|i_o - i_i| + |j_o - j_i|$.

En este caso, la distancia Manhattan es una mejor Heurística que la distancia Hamming debido a que el costo estimado está más cerca al costo real de la solución, lo que permite podar más el espacio de búsqueda. Asumiendo que tenemos una clase `State` que representa el estado del puzzle, una posible definición para la distancia Manhattan:

```py
def manhattan_distance(s1: State, s2: State) -> int:
    """Manhattan distance heuristic.

    Given s1 and s2, return the Manhattan distance between them.
    In this case if i1, j1 are coordinates of s1 and i2, j2 are
    coordinates of s2, the manhattan distance is computed by
    `abs(i1 - i2) + abs(j1 - j2)`.
    """
    distance = 0
    for row in s1.puzzle.board:
        for tile in row:
            i1, j1 = s1.puzzle.tile_pos[tile]
            i2, j2 = s2.puzzle.tile_pos[tile]
            distance += abs(i1 - i2) + abs(j1 - j2)

    return distance
```

Y el algoritmo `A*` se vería como:

```py
def a_star_puzzle(init_state, goal_state) -> List[Direction]:
    """A* search algorithm for the 8-puzzle problem.

    In theory it supports any size of the board,
    but in practice it only works for 3x3 boards. The reason
    being that the search space increases exponentially with the size
    of the board.
    """
    queue = PriorityQueue()
    queue.push(
        init_state, manhattan_distance(init_state, goal_state) + init_state.depth
    )
    visited = set()
    plan: List[Direction] = []
    while not queue.isEmpty():
        state = queue.pop()
        visited.add(state)
        if state == goal_state:
            while state:
                if state.action:
                    plan.insert(0, state.action)
                state = state.parent
            break
        for neighbor in state.neighbors():
            if neighbor not in visited:
                queue.push(
                    neighbor, manhattan_distance(neighbor, goal_state) + neighbor.depth
                )
    return plan
```

La cola de prioridades o _Priority Queue_ es una estructura de datos que en base a una función de costo, permite obtener elementos de forma ordenada haciendo orden en las inserciones de forma eficiente. Esta estructura de datos está basada en un [Binary Heap](https://en.wikipedia.org/wiki/Binary_heap).

El plan de pasos para llegar a la solución dado el estado inicial de ejemplo en la figura 2, se muestra a continuación:

<div align="center">

![puzzle-sol](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/1c42887ede70e37523fb601075a0e8cbf2d8a892/8puzzle-sol.gif)

_Fig 4: Solución y pasos para ejemplo del puzzle de las 8 piezas._

</div>

## Buscando soluciones para puzzles más grandes

Hasta acá todo bien, ¿cuál es el problema? Al crecer $N$ el espacio de búsqueda crece en órdenes de magnitudes. Por ejemplo el puzzle de las 8 piezas tiene $9! / 2 = 181440$ estados alcanzables. El puzzle de las 15 piezas tiene 1.3 trilliones de estados posibles, el puzzle de las 24 piezas ($5 \times 5$) tiene $10^25$ estados posibles...

Si no nos importa esperar unas cuantas horas, o cientos de años para encontrar una solución, entonces no hay problema (y eso, asumiendo que se tiene la RAM disponible para hacerlo). En caso contrario, hay que buscar otro tipo de soluciones, aunque sean sub-óptimas. En algunos casos _tener una solución es mejor que no tener ninguna_.

En estos casos, se puede aplicar el principio de _Divide y vencerás_, en el cual un problema se puede descomponer en sub-problemas, y la solución de estos problemas se puede agregar de manera tal de resolver el problema inicial. El paper [A Real-Time Algorithm for the $(n^2 − 1)$-Puzzle](https://ianparberry.com/pubs/saml.pdf) implementa un enfoque como el descrito. El algortimo es simple:

1. **Procedimiento** $\text{Resolver Puzzle}(n)$
   1. Si $n = 3$ resolver por fuerza bruta
       * En caso contrario usar algoritmo voraz para poner las filas y columnas en su posición correcta
   2. $\text{Resolver Puzzle}(n - 1)$


<div align="center">

![puzzle-greedy-1](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/649b07262fc51591de77f06b4c2aa561fe317496/greedy-n8-1.png)


![puzzle-greedy-2](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/649b07262fc51591de77f06b4c2aa561fe317496/greedy-n8-2.png)

_Fig 5: Muestra del algoritmo voraz para poner las filas y columnas en su lugar._

</div>

Esto se hace recursivamente hasta llegar al punto en que $n = 3$. Luego, aplicamos el algoritmo `A*` y resolvemos el resto del puzzle.

# Reflexiones Finales

* Plantear un problema de búsqueda no es complicado, pero encontrar una solución lo es. Encontrar una solución óptima lo es mucho más, a veces imposible o impráctico.
* Una heurística es una aproximación de una función desconocida, que puede utilizarse para realizar **búsqueda informada** y de esta forma reducir el espacio de búsqueda
* Resolvimos en problema del $N^2 - 1$ puzzle de manera sub-óptima para $n > 3$ y óptima en el caso de $n = 3$.

Como nota aparte, en su tiempo tomé aproximadamente 60 _MOOCs_ (_Massive Open Online Courses_); más que todos los cursos que tomé en ingeniería... Si me preguntan cuáles fueron mis favoritos:

1. [CS50's Introduction to Computer Science](https://www.edx.org/learn/computer-science/harvard-university-cs50-s-introduction-to-computer-science)
2. [Learning from Data](https://work.caltech.edu/telecourse.html)
3. [Electricity and Magnetism](https://www.edx.org/learn/magnetism/rice-university-electricity-and-magnetism-part-1)
4. [Algorithms Part 1](https://online.princeton.edu/algorithms-part-i)

No soy mucho de estudiar algo directamente como por ejemplo frameworks, etc... prefiero los libros MIL veces para entender los fundamentos y leer la documentación y papers para entender el uso. Por ello, no me gustan las plataformas como platzi, Udemy, etc... Me gusta mucho más la teoría abierta y a veces la que al parecer no tiene mucho uso, hasta que realmente te toca resolver un problema que hay que pensar.
