---
layout: post
title:  "Consistencia y consejos para entrevistas de programación"
date:   2023-01-05 19:27:03 -0400
categories: entrevistas ti
---

Primera entrada de este 2023, y como siempre escribo entre un poco aburrido y un poco para aportar conocimiento. Antes solía escribir artículos en linkedin, pero la verdad, tiene pocas opciones en cuanto a renderizado de contenido y la verdad no me interesa hacer una marca personal ni nada por el estilo. Por eso no pongo contador de visitas en este sitio, porque la verdad no espero muchas lecturas.

Sin más preámbulos, en esta entrada hablaré un poco de las entrevistas TI, de cómo proyectos personales pueden ayudar a mejorar tus habilidades de algoritmos y estructuras de datos, y también cómo abordar problemas de programación en entrevistas.

## Los "rants" de linkedin ##

He visto mucha gente en linkedin quejándose de los famosos __code challenges__, algunos porque dicen que se ponen nerviosos, o usan el término (mal utilizado) introvertido, entre otras cosas. Yo me considero una persona introvertida, y en general mis relaciones con la gente son banales, precisamente porque no me gusta compartir temas internos/emociones. Eso no significa que sea tímido y la verdad, hace rato dejé de ponerme nervioso en entrevistas de programación. De hecho, las disfruto, porque me entretengo con resolver un problema de algoritmos. Otra cosa es ser tímido, y eso es un tema que se trabaja poco a poco.

Estas personas que se quejan de que las entrevistas de programación no evalúan nada, generalmente quieren que les den el trabajo por CV. Eso sería bastante complejo, porque en papel todos son geniales, hay mucha gente con experiencia. Por lo tanto, imagino que las empresas deben tener un filtro. Algunas envían una "tarea" que se debe desarrollar en un plazo de `X` días. En lo personal, no me gusta este sistema, ya que me están __chupando__ vida y tiempo innecesariamente. Por ello, como el lector podrá imaginar, prefiero los __code challenges__.

## Proyectos Personales ##

Aquí hay un tema bastante complejo, no todos los proyectos personales dan experiencia para crecer como ingeniero de software, es decir:

* Tener miles de sitios web, con una arquitectura similar no ayuda más que tener 1.
* Tener notebooks de Kaggle, haciendo análisis de los sobrevivientes del Titánic tampoco es buen proyecto, es algo para hacer una vez.

En lugar de lo planteado anteriormente, lo que serían projectos interesantes son:

* Sitio web que resolviera algún problema (e.g recomendación de contenido en base a un catálogo, análisis de texto, uno que otro truco con la cámara web). Sitios de e-commerce hay millones y es hasta pega automatizable.
* Crear un algoritmo nuevo de Machine Learning
* Implementar un parser
* Inventar tu propio lenguaje y hacer un compilador
* Hacer análisis de árboles de derivación (e.g. contar nodos hijos de una frase nominal)
* Implementar servicios y correrlos en kubernetes (con API gateway)
* El clásico y trillado [acortador de urls]({% post_url 2022-07-05-acortador-url %}),
* etc.

Los proyectos mencionados al inicio, usan libs y frameworks que ya están hechos. Son habilidades buenas para tener, pero repetir lo mismo mecánicamente no te va a hacer utilizar el cerebro al máximo potencial. El segundo conjunto de proyectos, te hace pensar sobre estructuras de datos, algoritmos, trucos para hacer los algoritmos escalables, etc.

### Ejemplos de proyectos personales ###

#### TRUNAJOD ####

El proyecto open-source del que estoy más orgulloso es [TRUNAJOD](https://github.com/dpalmasan/TRUNAJOD2.0), me ha dado de comer (proyectos de investigación), he logrado publicar papers, y mejorar mis conocimientos de lingúística y lingúística computacional (así es, no todo es ejecutar un código copiado y pegado de pytorch). También me ha ayudado a entender mejor algunos problemas lingúísticos, como evaluación de coherencia, segementación de cláusulas, etc.

#### PyLogic ####

El nuevo proyecto en el que estoy trabajando es [py-logic](https://github.com/dpalmasan/py-logic) que es una biblioteca para implementar programas lógicos (tipo Prolog) en `Python`. Esto no lo hago puramente por capricho, la lógica es uno de mis temas favoritos, y siento que es el eslabón perdido en inteligencia artificial (hubo muchos desarrollos en los inicios). Con razonamiento sobre incertidumbre, han salido enfoques interesantes como __Markov Logic Networks__ e ILP (__Inductive Logic Programming__). Este último es básicamente aplicar ML a una base lógica.

En un [post anterior]({% post_url 2022-12-29-logic %}), hice una introducción a `py-logic`, en el que mencionaba la lógica proposicional. Hace poco añadí soporte a un subconjunto de la lógica de primer orden: Cláusulas de Horn. Que es una cláusula de Horn, es básicamente una implicación con antecedentes (no negados) en conjunción. Por ejemplo, si tenemos las cláusulas:

* Todo hombre es mortal
* Sócrates es hombre

En lógica de primer orden sería:

$$\begin{array} {   rr} \forall x\quad Hombre(x) \Rightarrow Mortal(x) \\ Hombre(Sócrates) \\ \hline Mortal(Sócrates) \end{array}$$

Por Modus Ponens concluimos que Sócrates es mortal. Las cláusulas mostradas, son todas de Horn, ya que son conjunciones de predicados no negados, con un sólo consecuente.

Volvamos a `py-logic`, y probemos un problema un poco más complicado, este problema es el coloreado de mapas, y va como sigue:


<div align="center">

![CSP](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d52db22306ac19fb7ec42239b906f4043a9e0c82/map.png)

_Fig 1: CSP mapa coloreable._

</div>

La idea es colorear cada nodo, pero los nodos adyacentes (que tienen una arista entre ellos) deben ser de diferente color. Este es en esencia un problema de satisfacción de restricciones (CSP, en particular el caso especial de 3-SAT). No voy a hablar de teoría de computación, pero este problema es NP-Hard (el lector lo puede demostrar utilizando colorarios de teoría de computación). Si bien este problema se puede resolver con algoritmos de búsqueda como `depth first search` o `breadth first search`, también hay otra solución, utilizar lógica y reducir el problema a un problema de inferencia lógica. Consideremos la siguiente base de conocimiento:

$$\begin{array} {   rr} \quad Diff(wa, nt) \land Diff(wa, sa) \land \\ Diff(nt, q) \land Diff(nt, sa) \land \\ Diff(q, nsw) \land Diff(q, sa) \land \\ Diff(nsw, v) \land Diff(nsw, a) \land \\ Diff(v, sa) \Rightarrow Coloreable() \\ Diff(Red, Blue) \quad Diff(Red, Green) \\ Diff(Blue, Red) \quad Diff(Blue, Green) \\ Diff(Green, Red) \quad Diff(Green, Blue) \end{array}$$

Las variables en este caso están en minúscula, y los las constantes (conocimiento del mundo) en mayúscula (por ejemplo que los colores adyacentes deben ser diferentes). Utilizando `py-logic`, tendríamos:

```python
from pylogic.fol import (
    HornClauseFOL,
    Predicate,
    Term,
    TermType,
    fol_bc_ask,
    Substitution,
)

wa = Term("wa", TermType.VARIABLE)
sa = Term("sa", TermType.VARIABLE)
nt = Term("nt", TermType.VARIABLE)
q = Term("q", TermType.VARIABLE)
nsw = Term("nsw", TermType.VARIABLE)
v = Term("v", TermType.VARIABLE)
t = Term("t", TermType.VARIABLE)

map = HornClauseFOL(
    [
        Predicate("Diff", [wa, nt]),
        Predicate("Diff", [wa, sa]),
        Predicate("Diff", [nt, q]),
        Predicate("Diff", [nt, sa]),
        Predicate("Diff", [q, nsw]),
        Predicate("Diff", [q, sa]),
        Predicate("Diff", [nsw, v]),
        Predicate("Diff", [nsw, sa]),
        Predicate("Diff", [v, sa]),
    ],
    Predicate("Colorable", []),
)

red = Term("Red", TermType.CONSTANT)
blue = Term("Blue", TermType.CONSTANT)
green = Term("Green", TermType.CONSTANT)

p1 = HornClauseFOL(
    [],
    Predicate("Diff", [red, blue]),
)
p2 = HornClauseFOL(
    [],
    Predicate("Diff", [red, green]),
)
p3 = HornClauseFOL(
    [],
    Predicate("Diff", [green, red]),
)
p4 = HornClauseFOL(
    [],
    Predicate("Diff", [green, blue]),
)
p5 = HornClauseFOL(
    [],
    Predicate("Diff", [blue, red]),
)
p6 = HornClauseFOL([], Predicate("Diff", [blue, green]))

kb = [map, p1, p2, p3, p4, p5, p6]

goal = Predicate("Colorable", [])
answers = fol_bc_ask(kb, [goal], Substitution({}))

for answer in answers:
    for k, v in answer.substitution_values.items():
        print(k, v)
    print("=" * 7)

```

Cuya salida es:

```
v5 Blue
nsw4 Green
q3 Blue
sa2 Red
wa0 Blue
nt1 Green
=======
v5 Blue
nsw4 Red
q3 Blue
sa2 Green
wa0 Blue
nt1 Red
=======
v5 Green
nsw4 Blue
q3 Green
sa2 Red
wa0 Green
nt1 Blue
=======
v5 Green
nsw4 Red
q3 Green
sa2 Blue
wa0 Green
nt1 Red
=======
v5 Red
nsw4 Green
q3 Red
sa2 Blue
wa0 Red
nt1 Green
=======
v5 Red
nsw4 Blue
q3 Red
sa2 Green
wa0 Red
nt1 Blue
=======
```

Algunas de las soluciones que me entrega:   


<div align="center">

![MAPS](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/cdedf6e478fcd36dff8783b334ae5e6b7c56542e/maps.drawio.png)

_Fig 2: Algunas soluciones del   mapa coloreable._

</div>

Estas soluciones se encontraron utilizand __Backward Chaining__. Lo que hace este algoritmo es razonamiento lógico desde la conclusión hasta los caminos que llevan a esa conclusión. Por ello, retorna todas las posibles asignaciones de variables que permiten satisfacer la base de conocimiento.

Otro ejemplo es, considerar el siguiente texto:

__"The law says that it is a crime for an American to sell weapons to hostile nations. The country Nono, an enemy of America, has some missiles, and
all of its missiles were sold to it by Colonel West, who is American.

Prove that Col. West is a criminal"__

Podemos convertir cada fragmento de información en una cláusula lógica:

1. It is a crime for an american to sell weapons to hostile nations

$$American(x) \land Weapon(y) \land Sells(x, y, z) \land Hostile(z) \Rightarrow Criminal(x)$$

2. Nono has some missiles:

$$Owns(Nono, M1)$$

$$Missile(M1)$$

3. All missiles were sold to it by Colonel West

$$Missile(x) \land Owns(Nono, x) \Rightarrow Sells(West, x, Nono)$$

4. Missiles are weapons

$$Missile(x) \Rightarrow Weapon(x)$$

5. An enemy of America counts as Hostile:

$$Enemy(x, America) \Rightarrow Hostile(x)$$

6. West who is american

$$American(West)$$

7. The country of Nono an enemy of America

$$Enemy(Nono, America)$$

8. Prove west is a Criminal

$$Criminal(West)$$

También este problema se puede resolver con backward chaining, pero también se puede resolver con __Forward Chaining__. Forward chaining es un razonamiento más cercano a lo que hacemos los humanos. Es decir, a partir de los hechos (cláusulas sin antecedentes), aplicamos el conocimiento y llegamos a la conclusión. La biblioteca `pylogic` también soporta forward chaining, pueden mirar los ejemplos en el repo y los tests.

#### Y, ¿Cuál es la gracia de todo esto? ####

Cualquiera pensaría que hice esta __majamama__ de código (paja mental) por las puras. Pero imagina:

* ¿Qué estructuras de datos debo pensar para aplicar los algoritmos?
* ¿Cómo pruebo que el código está correcto?
* ¿Cómo administro el proyecto?
* ¿Cómo debugeo un programa lógico?

Te adelanto que implementar los algoritmos fue un desafío gigante. El libro del que los estudié (__Artificial Intelligence a Modern Approach__) los pone muy abstractos, agnósticos al lenguaje. Por eso, después de tener este proyecto opensource:

* ¿Crees que me darían miedo las entrevistas de código?
* Tendría algo interesante para mostrar en entrevista técnica

Por otro lado, estoy en mi camino de aprender ILP (__Inductive Logic Programming__), por ejemplo, el paper [Learning First-Order Horn Clauses from Web Text](https://aclanthology.org/D10-1106.pdf), literalmente extrae conocimiento en forma de cláusulas de Horn, para hacer inferencias sobre el conocimiento en la web. Con esto se puede implementar un chatBot (que razone), un sistema experto, etc.

### Consejos para entrevistas ###

Hay varios consejos, a mi cuando me ha tocado entrevistar, noto de inmediato un candidato bueno de uno no tan bueno. Por ejemplo, si apenas termino la pregunta veo a un candidato martillar código, me da una mala impresión. Por otro lado, si veo a una persona razonando sobre el problema y luego implementarlo, eso me convence al menos inicialmente. Algunas buenas señales:

* Razona el problema antes de siquiera escribir una línea de código
* Piensa en los pro-contras de la solución y complejidades (en tiempo y espacio)
* Escribe código limpio (sin martillazos) representando la solución
* Piensa sobre testing, casos de borde, etc.

Anecdóticamente, una entrevista que tuve hace un tiempo, la pregunta fue "Implementa un algoritmo que predice la siguiente palabra". El enunciado es ambiguo y no es sólo martillar código, es pensar:

* ¿Cuál es la entrada?
* ¿Qué es lo que debo procesar?
* ¿Cuénto con una base de información para extraer frecuencia de palabras?
* ¿Qué hago con las palabras fuera del vocabulario?
* ¿Cómo testeo el programa?
* Cómo puedo hacerlo más eficiente (caching, etc.)

No pondré la respuesta, pero spoiler: Hice Ace en esa entrevista (aprobé jeje). Dejo algunos ejemplos de problemas y razonamiento mi repo de [code-challenges](https://github.com/dpalmasan/code-challenges) donde pueden ver problemas, razonamiento y soluciones en `Python`.

## Notas Finales ##

* A mi no me venden la mula hablando de "lambdas" y cloud y bla bla bla.
* Lo que en verdad importa es experiencia + capacidad de resolver problemas
* Algunos dicen que los unicornios no existen, yo he visto ing 10x, y aunque yo no lo soy, he trabajado en distintos roles siempre con desempeño excelente.

Espero que hayan servido mis consejos y algún día escribiré otro blog.