---
layout: post
title:  "Tips para entrevistas SWE/MLE"
date:   2023-07-04 18:10:03 -0400
categories: python algorithms classification machine-learning
---

## Introducción

Muchos se preguntarán ¿Y éste pelagato qué tiene que dar tips/consejos? Yo me preguntaría lo mismo inicialmente, pero aconsejo seguir leyendo pues puede haber información útil. Un poco sobre mi:

* He trabajado en múltiples roles a lo largo de mi carrera (QA SWE, Data Scientist, Data Engineer, SWE backend, MLE)
* Llevo 1 año trabajando en USA (FAANG) y logré sacar el máximo "rating" posible en evaluación de desempeño en mi primer año (top 1%-3% de la empresa obtiene este ranking, víctima del [stack ranking](https://www.betterup.com/blog/stack-ranking) 😅)
* En mi estadía en USA, he tenido ofertas de 2 FAANG y una empresa non-FAANG pero tamnbién conocida (por lo que conozco el sistema y cómo pasar las entrevistas)
* He publicado múltiples papers en NLP/ML y también tengo proyectos open-source que han tenido al menos un poco de éxito (estrellas en Github)

Lo anterior puede sonar __d*** measurement__, pero ya luego de haber tenido buen desempeño en múltiples roles, puedo ya pensar que no ha sido coincidencia y liberarme un poco del __síndrome del impostor__ en este blog.

El mercado actual está muy competitivo debido a despidos masivos por el momento económico actual, por lo tanto hay muchos talentos disponibles en el mercado. Por lo tanto, es importante conocer la mecánica de los procesos de selección. En esta entrada, comentaré algunos detalles a tener en cuenta y algunos tips para las entrevistas de código.

## 1. Currículum Vitae (CV)

Un error común cuando me toca revisar currículums es ver experiencias como __"Programé X"__, __"Seguí buenas prácticas de código"__, __"Lideré Y"__. En caso de MLE __"Creé un modelo para Z con precisión 90%"__. Lo relevante es ver el impacto que tuvo la experiencia. Por ejemplo, se puede re-escribir una experiencia como __"Implementé X, lo que incrementó los usuarios activos diarios en un 30%"__, __"Escalé servicio Z, que redujo la latencia p95 de 200ms a 50ms"__, __"Implementé un modelo de ML para asignación de recursos que redujo costos mensuales en $2M"__. La experiencia debe describir el impacto/dirección que se tuvo.

Otro error común es mencionar __tecnologías internas__ o __proyectos internos__. Por ejemplo, supongamos que la persona desarrolló el proyecto __"Cooperación de Agentes para Calibración Agnóstica"__ e internamente se conoce como proyecto `CACA`. Poner como experiencia __"Trabajé en el proyecto `CACA`"__ no significa nada a menos que se indique el contexto. En este caso, se debe explicar la experiencia de forma que sea autocontenida y que la persona que lea el CV no necesite contexto.

Finalmente, hay que tener en cuenta que muchos CVs se filtran automáticamente, por palabras claves. Por lo tanto, si la descripción de un cargo menciona que se requiere conocimiento tecnología `A1`, `A2`, intentar agregar dichas tecnologías al CV, e intentar agregar cómo se usó dicha tecnología en la experiencia.

## 2. Entrevista Técnica (código)

Es importante tener conocimiento de estructuras de datos básicas y algoritmos, por ejemplo:

* Manipuación de Arrays
* Algoritmos de ordenamiento
* Árboles Binarios
* Recursividad
* Depth-First-Search/Breadth-First-Search
* Grafos
* Complejidad en espacio y tiempo de ejecución

### Manipulación de Arrays

Un ejemplo práctico:

__Dado un array de citas bibliográficas de un autor, donde `citas[i]` es el número de citas que un investigador recibió por su paper `i`, retornar el índice `h` del investigador. El índice `h` se define como el máximo valor de `h` tal que el autor ha publicado al menos `h` papers que han sido citados al menos `h` veces.__

Ejemplos:

```
Entrada: [3, 0, 6, 1, 5]
Salida: 3
Explicación: El investigador tiene 5 papers en total. Dado que el investigador tiene 3 papers que han sido citados al menos 3 veces, su índice `h` es 3.
```

```
Entrada: [1, 3, 1]
Salida: 1
Explicación: El investigador tiene 3 papers. El investigador tiene 2 papers que han sido citados al menos 1 vez, cero papers que hayan sido citados 2 veces y 1 sólo paper que ha sido citado 3 veces.
```

```
Entrada: [100]
Salida: 1
Explicación: El investigador tiene 1 paper. El paper ha sido citado al menos 100 veces, por lo tanto el valor máximo de `h` es 1.
```

Restricciones:

`1 <= n <= 5000`
`0 <= citas[i] <= 1000`

Una solución simple es hacer fuerza bruta, es decir, iterar sobre todos los valores posibles de `h` y encontrar el máximo `h` satisfaga la condición mencionada. Por ejemplo en `python`:

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        max_seen = 0
        hmax = 0
        for h in range(len(citations)):
            count = 0
            for citation in citations:
                if citation >= h + 1:
                    count += 1
            if count >= h + 1 and count >= max_seen:
                hmax = h + 1
        return hmax
```

Dicha solución tiene una complejidad de $O(n^2)$ en tiempo de ejecución y $O(1)$ en memoria, ya que no se utiliza memoria extra. En este caso, el entrevistador podrá preguntar, ¿se puede obtener una solución mejor en tiempo de ejecución?

En este caso se podría ordenar el array ($O(n\log(n))$) y recorrer cada paper hasta encontrar el paper tal que `citas[h] < h`, como el arreglo está ordenado, el invariante del ciclo es que siempre tendré el máximo valor posible de `h` entre el índice `[0, h]`, ya que el array está ordenado:

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort(reverse=True)
        h = 0
        while h < len(citations) and citations[h] >= h + 1:
            h += 1
        return h
```

Complejidad asintótica $O(n \log n)$ en tiempo de ejecución, y $O(1)$ en memoria. Luego el entrevistador podría preguntar nuevamente ¿Se puede hacer mejor? En este caso, el cuello de botella es el algoritmo de ordenamiento, por lo que se puede mencionar que dadas las retricciones, se podría implementar __count sort__ que sería constante en memoria ($O(k)$) y $O(n)$ en ordenamiento, reduciendo la complejidad en tiempo de ejecución a $O(n)$.

### Dos Punteros

Máximo número de de pares que suman `k`:

__Dado un array `nums` y un entero `k`, en una operación, se pueden tomar dos números que sumen `k` y removerlos del array. Retornar el **máximo número de operaciones que se pueden aplicar a este array**__.

```
Entrada: nums = [1,2,3,4], k = 5
Salida: 2
Explicación:
- Remover los números 1 y 4, luego nums = [2,3]
- Remover 2 y 3, luego numes = []
No hay más pares, por lo que la salida es 2
```

```
Entrada: nums = [3,1,3,4,3], k = 6
Salida: 1
Explicación:
- Remover los primeros 3 = [1,4,3]
No hay más pares que sumen 6, por lo que el resultado es 1
```

En este caso, se puede ordenar el array de entrada. Luego se pueden tener dos punteros, uno a la izquierda del array y otro a la derecha. Si `nums[left] + num[right] < k`, entonces mover `left` hacia adelante. Si la suma es mayor que `k`, mover `right` hacia atras. Si la suma es igual a `k` incrementar el conteo de operaciones y mover ambos punteros (hacia la izquierda y derecha respectivamente). Continuar hasta que `left == right`.

```python
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        total = 0
        start = 0
        end = len(nums) - 1
        nums.sort()
        while start < end:
            if nums[start] + nums[end] > k:
                end -= 1
            elif nums[start] + nums[end] < k:
                start += 1
            else:
                total += 1
                start += 1
                end -= 1
        return total
```

Complejidad en tiempo de ejecución $O(n \log n)$, complejidad en memoria $O(1)$. Este truco también se puede aplicar en listas enlazadas.

### Ventana Deslizante (Sliding Window)

Suma de sub-array de tamaño mínimo:

__Dado un array de enteros positivos `nums` y un número positivo `target`, retornar el largo mínimo de un subarray cuya suma es mayor o igual que `target`. Si no existe dicho subarray, retornar 0__:

```
Entrada: target = 7, nums = [2,3,1,2,4,3]
Salida: 2
Explicación: El subarray [4,3] tiene el largo mínimo dadas las restricciones
```

```
Entrada: target = 4, nums = [1,4,4]
Salida: 1
```

```
Entrada: target = 11, nums = [1,1,1,1,1,1,1,1]
Salida: 0
```

En este caso, la solución simple sería probar todos los subarray posibles. En este caso la complejidad en tiempo de ejecución sería $O(n^2)$. Otra solución sería utilizar una __ventana deslizante__, es decir, un puntero inicial y un puntero final hacia elementos del array, tal que la suma de los elementos entre los punteros cumpla la restricción. El invariante del ciclo sería que en cada iteración, se tiene la mínima ventana encontrada que satisfaga las restricciones. El algoritmo sería:

* Inicializar puntero inicial `i = 0` y `min_length = inf` (también se puede usar `min_length = n + 1` donde `n` es el largo de `nums`)
* Inicializar la suma como `0`
* Para cada puntero final (iniciando desde 0), agregar `nums[j]` a la suma total
* Mientras la suma total sea mayor o igual a la suma objetivo (`target`), mover `i` hacia la derecha
* Actualizar largo mínimo (`min_length = (min_length, j - i + 1)`)
* Retornar `0` si `min_length == inf` o `min_length` en caso contrario

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        min_length = float('inf')

        i = 0
        total_sum = 0
        for j, n in enumerate(nums):
            total_sum += n
            while total_sum >= target:
                min_length = min(min_length, j - i + 1)
                total_sum -= nums[i]
                i += 1

        return 0 if min_length == float('inf') else min_length
```

Notar que aunque la solución tiene dos ciclos anidados, la complejidad asintótica es $O(n)$, por lo que no recomiendo mecanizar el análisis de complejdidad (por ejemplo decir __Si tiene $k$ ciclos anidados entonces complejidad es $O(n^k)$__).

### Árboles Binarios

Algunos consejos para este tipo de problemas:

* Intentar la solución simple inicialmente (por ejemplo utilizando recursividad)
* Tener claro lo que significa recorrido en __preorder__, __postorder__, __inorder__
* Diferencia entre recorrer por niveles (Usando Breadth First Search) o recorrer por profunidad (Depth First Search)

Un ejemplo simple, invertir un árbol binario:

__Dada la raíz de un árbol binario (`root`), invertir el árbol y retornar su raíz__:

<div align="center">

![tree-1](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/a6b49289782e60ceb2d40fb8682f2821514e4e9c/invert1-tree.jpg)

_Fig. 1: Ejemplo 1._

</div>

<div align="center">

![tree-2](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/a6b49289782e60ceb2d40fb8682f2821514e4e9c/invert2-tree.jpg)

_Fig. 2: Ejemplo 2._

</div>

En este caso, una solución simple es utilizar recursividad, por ejemplo:

* Dado un nodo, intercambiar el nodo izquierdo con el derecho, repetir con el nodo izquierdo y el derecho
* Si el nodo es nulo, retornar
* Retornar la raíz

En este caso, al terminar de recorrer el árbol, todos los nodos habrán sido invertidos.

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def invert_childs(node):
            if node is None:
                return
            tmp = node.left
            node.left, node.right = node.right, node.left
            invert_childs(node.left)
            invert_childs(node.right)

        if root is None:
            return root

        if root.left is None and root.right is None:
            return root

        invert_childs(root)
        return root
```

En este caso la complejidad es $O(n)$ donde $n$ es el número de nodos (pues recorremos todos los nodos del árbol una sola vez).

### Grafos

Problemas que involucren grafos aparecen frecuentemente en este tipo de entrevistas (y en varios problemas reales). Lo que recomiendo es saber cómo recorrer un grafo, detección de ciclos, cómo representar un grafo (ejemplo: matriz de adyacencia), y cómo medir la complejidad en un problema de grafos (¿proporcional a los vértices, a las aristas?).

Ejemplo de problema, Horario Cursos:

__Existe un total de cursos `numCourses` que tienes que tomar, etiquetados desde `0` a `numCourses - 1`. Se te da un array `prerequisites` donde `prerequisites[i] = [ai, bi]` indica que debes tomar el curso `bi` si quieres tomar `ai`. Por ejemplo, el par `[0, 1]` indica que para tomar el curso `0`, se debe haber tomado el curso `1`. Crear una función que retorne `true` si se pueden completar todos los cursos y `false` en caso contrario.__

```
Entrada: numCourses = 2, prerequisites = [[1,0]]
Salida: true
Explicación: Hay dos cursos a tomar.
Para tomar el curso 1, se debe tomar el curso 0, por lo que es posible tomar ambos cursos.
```

```
Entrada: numCourses = 2, prerequisites = [[1,0],[0,1]]
Salida: false
Explicación: Hay dos cursos a tomar
Para tomar el curso 0, se debe tomar el curso 1 y para tomar el curso 1 se debe tomar el curso 0, por lo que es imposible tomar todos los cursos
```

En este caso, el problema se puede representar como un grafo. En esencia, el problema se puede simplificar a encontrar un ciclo dentro del grafo (pues esto implicaría que hay dependencias circulares entre cursos). Un algoritmo podría ser:

* Construir el grafo a partir de los pares de cursos (ejemplo: Matriz de adyacencia)
* Para cada curso, recorrer el grafo en busca de algún ciclo, por ejemplo utilizando __Depth-First-Search__
* Si se encuentra un ciclo retornar `false`
* En caso de que se recorrieron todos los nodos y caminos, retornar `true`

```python
from collections import defaultdict

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        edges = defaultdict(list)
        for prereq in prerequisites:
            edges[prereq[1]].append(prereq[0])

        for course in range(numCourses):
            stack = [course]
            visited = set()
            while stack:
                node = stack.pop()
                visited.add(node)
                for neighbor in edges[node]:
                    if neighbor == course:
                        return False
                    if neighbor not in visited:
                        stack.append(neighbor)

        return True
```

En este caso, la complejidad asintótica en tiempo de ejecución es $O(|V||E|)$ ya que para cada vértice se deben recorrer todas las aristas. La complejidad en espacio es $O(V)$, pues a lo más se guardan $|V|$ nodos en el conjunto de nodos visitados.

## 3. Entrevista Técnica (Diseño de Sistemas)

En este caso depende mucho de la empresa. Este tipo de entrevistas son bastante abiertas, y por lo general se utilizan para medir el __Seniority__ del postulante. En FAANG las preguntas son más abiertas y no están ligadas a un framework específico. En entrevistas para empresas convencionales, pueden aparecer preguntas sobre ciertos frameworks, por ejemplo `kubernetes`, manejo de despliegues, entre otros detalles un poco más ligados al framework. Lo que es cierto para ambos casos es que conviene saber conceptos como:

* Escalabilidad (ejemplo: horizontal vs vertical)
* Latencia y uso de CPU (qué significa p95, p99, etc.); SLA, SLO
* Uso de Caching, invalidación, TTL
* Bases de datos (ejemplo: SQL vs NoSQL, OLAP vs OLTP, B-Trees vs Log-Structured Merge Tree)
* Servicios, serialización de datos, protocolos para transmisión de datos (ejemplo: TCP vs UDP)
* Sharding de bases de datos, procesamiento distribuído (ejemplo: Round Robin vs Consistent Hashing)
* Etc

Ejemplos de problemas:

* Diseñar un acortador de URLs
* Diseñar un motor de búsqueda exacta de textos

En Machine Learning:

* Flujo completo y mantención de modelos (entrenamiento y despliegue, comunicación con servicios, escalamiento, monitoreo de rendimiento)
* Algoritmos comunes y sus ventajas y desventajas (ejemplo: En qué consiste un árbol de decisión, qué diferencias existen entre Random Forests y XGB y cuándo se prefiere uno del otro)
* Qué diferencia hay entre __similarity aware__ embeddings vs los que no lo son
* Diferencia entre tener alto volumen de datos vs bajo y qué modelos utilizar

Ejemplo de problemas de diseño de sistemas en MLE:

* Diseñar un detector de armas en anuncios
* Diseñar un sistema para __Federated Machine Learning__

La idea en este tipo de entrevistas es iterar con el entrevistador, aclarar supuestos, definir el problema. Análisis de diferentes alternativas u ventajas/desventajas (ej. escalamiento vertical vs horizontal).

## Conclusiones

Ya cerrando esta entrada, en resumen:

* El CV debe ser conciso, ir al grano y mostrar impacto en las tareas desempeñadas; idealmente apuntar al cargo
* Se deben tener conocimientos básicos de algoritmos y estructuras de datos. Hay ciertas técnicas encontrar soluciones mejores en términos de complejidad asintótica
* La experiencia y buenos fundamentos teóricos son la clave para arrasar en una entrevista de diseño de sistemas

Finalmente, a los que estén buscando empleo, les deseo suerte y espero que estos tips les sirvan.

Saludos!
