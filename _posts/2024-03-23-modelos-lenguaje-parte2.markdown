---
layout: post
title:  "Entendiendo los Modelos del Lenguaje (Parte 2)"
date:   2024-03-03 15:30:00 -0400
categories: probability algorithms ai
---

# Introducción

Este es el segundo artículo de la serie que estoy escribiendo sobre modelos de lenguaje. En mi artículo previo:

* [_Entendiendo los Modelos del Lenguaje (Parte 1)_]({{ site.baseurl }}{% link _posts/2024-03-23-modelos-lenguaje-parte1.markdown %})

expliqué en qué consiste un modelo de lenguaje y cómo el problema a resolver es encontrar una distribución conjunta $p(w_1, w_2, \ldots w_N)$ donde $N$ es el largo del contexto a considerar y $w_i$ es la i-ésima palabra de un texto en esta distribución.

En este artículo, describiré en términos básicos, la solución del estado del arte en este problema y además contaré la historia de ChatGPT y los fundamentos de este sistema de IA.

# ¿Cómo se "_entrena_" ChatGPT?

Cuando hablamos de "_entrenamiento_", nos referimos al proceso de encontrar $p(w_1, w_2, \ldots w_N)$. En general esto se hace construyendo un **conjunto de datos**. Como se describió en el artículo previo, una forma de encontrar esta distribución es mediante la _predicción de la siguiente palabra_, estrategia que consiste en encadenar múltiples probabilidades condicionales. Sin embargo, ChatGPT es un sistema que tiene otros ingredientes que permiten que funcione bien para diferentes tareas, como por ejemplo, tener una conversación _coherente_ con un ser humano (en algunos casos, puede no ser coherente). En la figura 1. se muestra la evolución de ChatGPT.

<div align="center">

![chat-gpt](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/23af0a877eb19a022e0682f565d7c9a9d6475a82/chat-gpt-history.png)

_Fig 1: ChatGPT y su evolución en el tiempo._

</div>

El conjunto de datos para entrenar los modelos que utiliza el sistema, consisten en un repositorio de textos extraídos desde múltiples fuentes en internet, como artículos de Wikipedia, libros, y otras páginas web. Finalmente, chatGPT es un sistema de IA que debe ejecutar las siguientes tareas:

1. Predecir la palabra siguiente.
2. Aprender a seguir instrucciones.
3. Aprender a conversar.

## Predecir la siguiente palabra

Como mencioné en la parte 1 de esta serie, la tarea de aprender a predecir la siguiente palabra es una forma de estimar la distribución de probabilidad que representa el modelo de lenguaje. Tomemos el siguiente ejemplo:

* _El gato estaba sentado en el \_\_\_\_\__

En este caso existen varias formas de completar el texto. Por ejemplo las palabras _sofá_, _mat_, _techo_, _mesón_, llevarían a tener un texto coherente. Esto quiere decir, que el modelo podría **entregar respuestas diferentes cuando se le consulta el mismo texto en múltiples ocasiones**, pero cada una de estas respuestas tendrá sentido en el contexto de la oración. Si se provee mayor contexto, el modelo podrá ser más determinista/consistente en completar la oración:

* _Pedro buscó por todos lados a su gato que estaba perdido. Buscó en su casa, en el armario, incluso en el patio. Pedro decidió ir salir a la calle a buscar al felino, hasta que escuchó un maullido que venía desde arriba. El gato estaba sentado en el techo._

En este caso, el modelo eliminó las palabras mat, sofá y mesón, ya que el contexto dado sugería que dichas palabras no mantendrían la coherencia del texto. Cabe destacar, que para lograr considerar el contexto, se requiere un modelo sofisticado, que sea capaz tener cierta noción semántica de las palabras y poner **atención** a las palabras relevantes que entregan la información necesaria para hacer la mejor predicción. Métodos tradicionales utilizando cadenas de Markov y N-Gramas en general no tienen estas características y consideraban a todas las palabras con igual importancia (hubo intentos de mejora, pero nada que pudiese considerar un contexto _largo_).

En el 2018 investigadores de OpenAI entrenaron un modelo que era capaz de realizar la tarea de predecir la siguiente palabra utilizando una arquitectura de _transformers_: [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762). El modelo utilizó una gran cantidad de datos extraídos de distintas fuentes (ej. wikipedia, sitios web públicos, libros, etc). El modelo también tenía una gran cantidad de parámetros, lo que le permitió aprender patrones que era imposible aprender con modelos menos complejos. Este modelo lo llamaron **GPT**, y era capaz de completar oraciones y párrafos. En los siguientes dos años, mejoraron este modelo agregando incluso más parámetros y datos de entrenamiento. Finalmente, lograron el modelo **GPT-3** que tiene 175 billones de parámetros.

Cabe destacar que **GPT-3** no fue entrenado para alguna tarea específica, otra que no sea predecir la siguiente palabra. Sin embargo, el tamaño del conjunto de datos era masivamente grande que contenía texto con varios ejemplos de tareas específicas (sistemas pregunta-respuesta, resumir textos, traducción, etc.), lo que si se utilizaba una consulta/contexto apropiado, el modelo iba a ser capaz de resolver dicha tarea.

## Aprendiendo a seguir instrucciones

En la siguiente fase, el modelo GPT-3 fue entrenado para seguir instrucciones. Para lograr esto, se creó un conjunto de datos que contiene respuestas "deseables" generadas por humanos para una variada cantidad de instrucciones. Primeramente, el modelo fue entrenado de manera que aprendiera qué respuestas son deseables. Además, el modelo se fue ajustando con retroalimentación humana para mejorar su entendimiento  sobre contenido deseable. Cada vez que el modelo generaba una respuesta satisfactoria, se le recompensaba con un puntaje positivo, en caso contrario había una penalización. El modelo intentaba aprender cómo generar contenido, de manera de maximizar los puntajes entregados por la retroalimentación, aprendiendo lentamente a generar contenido de acuerdo a esta escala de deseabilidad. Este proceso de enseñarle al modelo vía retroalimentación humana se conoce como _Aprendizaje por Refuerzo con Retroalimentación Humana_ (_RHLF: Reinforcement Learning with Human Feedback_). A este sistema le pusieron **InstructGPT**, y fue lanzado el 2022: [_Training language models to follow instructions with human feedback_](https://arxiv.org/abs/2203.02155).

## Aprendiendo a Conversar

En la siguiente fase, que culminó en ChatGPT, OpenAI entrenó al modelo para que pudiese conversar de manera efectiva. El conjunto de datos inicial consistió en conversaciones donde los humanos actuaban en ambos roles: El usuario del _chatbot_ con IA, y el chatbot (un humano "actuaba" de ChatGPT). Este modelo también fue mejorado vía RLHF. El formato de diálogo permitió al modelo a responder preguntas de seguimiento, admitir errores, y desafiar premisas incorrectas.

## "_Prompt Engineering_ y Reflexiones"

Del uso masivo de ChatGPT, nació la ingeniería de consultas "_Prompt Engineering_", la que consiste en realizar una interacción _adecuada_ con el sistema de IA, de manera de obtener respuestas que satisfagan mejor o tengan la mejor "deseabilidad" posible. De aquí nacen super-usuarios del sistema, que logran optimizar al máximo esta herramienta de IA. El gran problema, es el sensacionalismo que algunos creadores de contenido generan, sobre-estimando y exagerando las capacidades del modelo. Mi opinión personal, no tengo nada en contra de que la gente utilice estas herramientas y comparta contenido que permita optimizar la experiencia de usuario. Es más, yo también utilizo algún modelo GPT para ayudarme a ser más eficiente para escribir código. Sin embargo, lo que me molesta es que existan creadores de contenido que divulgan información falsa/engañosa. Hay que ser responsables con el uso de la tecnología.

Finalmente, recordar que ChatGPT (y cualquier otro modelo similar), es simplemente un modelo que intenta estimar la distribución $p(w_1, w_2, \ldots w_N)$ y que gracias a tener un gran repositorio de textos para _"aprender"_, se logró re-ajustar para que pudiese resolver más tareas que predecir la siguiente palabra. Pero no debemos olvidar que es un simple muestreo de una distribución de probabilidad. Por el momento, estamos lejos de _ser reemplazados_ por IA. Es más, puede darse que algunos trabajos queden obsoletos, sin embargo pueden nacer nuevos empleos. Lo mismo ha ocurrido con varias revoluciones tecnológicas.

## Bonus: ¿Qué es el significado?

Al ver el lenguaje en términos de lógica se vislumbra el significado de ciertos tipos de expresiones justificándolas en la validez de los argumentos. Un argumento válido es un argumento donde si las premisas son verdaderas, entonces las conclusiones también son verdaderas. El significado de ciertas expresiones en el lengaje natural juegan un rol crucial en contribuir a la validez de los argumentos y así, uno puede representar su significado modelando dicha validez.

Por ejemplo, consideremos el siguiente argumento, que es un silogismo (Aristóteles) clásico:

$$
\begin{array} {c}
    \text{Todo hombre es mortal} \\\\
    \text{Sócrates es hombre} \\\\
    \hline
    \text{Sócrates es mortal}
\end{array}
$$

Cabe destacar que matemáticamente, estos axiomas se pueden fórmular con **Lógica de Orden**:

$$
\begin{array} {c}
    \forall x (Hombre(x) \rightarrow Mortal(x)) \\\\
    Hombre(s) \\\\
    \hline
    Mortal(s)
\end{array}
$$

En este caso $\forall$ es un cuantificador _universal_, $x$ es una entidad (u objeto), y $Hombre$ y $Mortal$ son predicados lógicos. Consideremos los siguientes axiomas:

1. Todos los niños aman al Viejo Pascuero
2. Todos los niños que aman al Viejo Pascuero, aman a cualquier reno
3. Rodolfo es un reno y Rodolfo tiene la nariz Roja
4. Cualquiera que tenga la nariz roja es extraño o es un payaso
5. Ningún reno es un payaso
6. Juan no ama cualquier cosa que sea extraña

¿Podemos concluir que Juan no es un niño?

Una posible respuesta de ChatGPT:

<div align="center">

![chat-gpt](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/874f82dc54fdb39bcca26fd04c9a4e6c99702059/logic-gpt.png)

_Fig 2: ChatGPT y su respuesta a los axiomas de navidad._

</div>

Sin embargo, si transformamos los axiomas a lógica de primer órden:

1. $\forall x \text{ niño}(x) \rightarrow ama(x, vp)$
2. $\forall x, y (\text{niño}(x) \land reno(y) \rightarrow ama(x, y))$
3. $reno(r) \land nariz\ roja(r)$
4. $\forall x (nariz\ roja(x) \rightarrow \text{extraño}(x) \lor payaso(x))$
5. $\neg \exists x(reno(x) \land payaso(x))$
6. $\forall x (\text{extraño}(x) \rightarrow \neg ama(j, x))$
7. Conclusión: $\neg \text{niño}(j)$

Supongamos que Juan fuese niño:

* Si Juan fuera un niño, entonces juan amaría al reno Rodolfo, por axioma `2.`
* Rodolfo tiene la nariz roja, por lo tanto o es extraño o es un payaso, por axioma `4.`
* Rodolfo no es un payaso ya que no existe una entidad que sea reno y payaso a la vez por axioma `5.`
* Rodolfo es extraño por axioma `4.`
* Juan no ama a nada que sea extraño, pero rodolfo es extraño, por lo que juan no amaría a rodolfo, axioma `6.`
* Por lo tanto hay una contradicción y Juan no puede ser un niño.

Recordemos que ChatGPT no es determinista, por lo que en algunos casos podría dar la respuesta correcta (ya que este es un ejemplo conocido). Sin embargo, en este caso podemos notar que ChatGPT llega a una conclusión errónea, por lo que no está realmente razonando.

# Conclusiones

* ChatGPT comenzó con un modelo GPT entrenado para predicción de la siguiente palabra.
* El modelo fue entrenado con una cantidad masiva de datos, por lo que indirectamente podía resolver algunas tareas dándole una consulta/contexto indicado.
* ChatGPT utiliza RLHF para aprender a interactuar con humanos.
* ChatGPT no está razonando lógicamente y no entiende el significado de los textos.
* Estamos lejos de ser reemplazados por IA, sin embargo, existen Super-usuarios o _power users_ que le pueden sacar máximo provecho a este tipo de sistemas con IA.
