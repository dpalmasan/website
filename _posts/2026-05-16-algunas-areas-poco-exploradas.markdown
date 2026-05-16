---
layout: post
title:  "Algunas áreas poco exploradas en IA moderna"
date:   2026-05-16 11:16:00 -0400
categories: swe dev machine learning ai
---

# Introducción

En este post toco variados temas, el título lo dice todo. Algunos comentarios pueden parecer ácidos, pero esto es simplemente compartir opiniones. No planeo generar debate, ni ser intencionalmente polémico; intento mantenerme lo más objetivo posible incluso tratando temas subjetivos, pero siempre enlazando con lo técnico y teoría sustentada.

# Algunos tópicos de IA

Hace un par de años, cuando estaba estudiando ciertos fundamentos, me dio por implementar [`pylogic`](https://github.com/dpalmasan/py-logic). En ese tiempo estaba estudiando _Knowledge Representation_, en particular con enfoques basados en teoría de números, lógica, y álgebra Booleana. Algunos problemas de satisfacción de restricciones se pueden resolver con enfoques lógicos, ya sea con _Forward Chaining_ o _Backward Chaining_: elegir el tipo de razonamiento, _top-down_, _bottom-up_ o híbrido, depende de las restricciones del problema a resolver. Un problema clásico y ejemplo de `Prolog`, es por ejemplo el problema del coloreo de mapas: Colorear un mapa en que nodos adjacentes no tengan el mismo color. Por ejemplo tomemos el siguiente mapa:

<div align="center">

![map-example](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/61278e88a47467c65385d88b3eda4826c2e3065a/map.png)

</div>

Puede representarse con los siguientes axiomas lógicos:

$$
\begin{aligned}
&Diff(wa, nt) \land Diff(wa, sa) \land \\\\
&Diff(nt, q) \land Diff(nt, sa) \land \\\\
&Diff(q, nsw) \land Diff(q, sa) \land \\\\
&Diff(nsw, v) \land Diff(nsw, a) \land \\\\
&Diff(v, sa) \Rightarrow Coloreable()
\end{aligned}
$$

$$
\begin{aligned}
&Diff(Red, Blue) \quad Diff(Red, Green) \\\\
&Diff(Blue, Red) \quad Diff(Blue, Green) \\\\
&Diff(Green, Red) \quad Diff(Green, Blue)
\end{aligned}
$$

Razonando usando backward chaining:

<div align="center">

![map-sol](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/61278e88a47467c65385d88b3eda4826c2e3065a/map_coloring_reasoning.gif)

</div>

Si bien, el tema de razonamiento lógico y axiomas es bien utilizado en matemáticas, transformar conocimiento "crudo" en el mundo real (por ejemplo que venga de textos) es un problema complicado, porque habría que reconocer los axiomas, hacer parsing (en lenguaje natural esto se pensaba como un problema intratable, los LLMs han mostrado algo de éxito), construir la _Knowledge Base_, y razonar. Ahora con LLMs, permite una buena aproximación para esta parte del flujo de razonamiento. Por otro lado, definir los axiomas y las relaciones es otro problema, pero, tenemos _Inductive Logic Programming_ (ILP), recomiendo explorar el tema. Como dato curioso, recientemente resolví un problema en el que el "prompt engineering" no se podía refinar debido a alucinaciones y la complejidad del problema (pérdida de contexto); no había _Knowledge Graph_ (KG). Sin embargo, pude codificar conocimiento de expertos y aplicar ILP, lo cual resolvió un problema que se pensaba que no tenía solución. Planeo publicar un paper en esto, junto a mis colegas.

Otro tema interesante: trabajo frecuentemente con investigadores de investigación aplicada (PhD en general), y algo que valoro mucho es la honestidad intelectual al momento de colaborar. Durante esta experiencia reciente que mencioné, uno de ellos me comentó honestamente: "Pasé esta parte por un LLM, pero creo que el modelo tampoco entiende realmente qué está ocurriendo aquí. ¿Podrías explicarlo?". Ese tipo de interacciones suelen ser mucho más productivas que asumir que los modelos siempre tienen la respuesta correcta.


## Ahondando en más tópicos

Otros ejemplos clásicos que veo en internet: el tema de las bases de datos vectoriales es viejísimo, de hecho de los tiempos de LSI (Latent Semantic Indexing) que las grandes empresas están haciendo esto, e incluso antes. Lo mismo con grafos de conocimiento, y verificación formal. Lo mismo con agentes IA, probablemente hace más de 40 años que existen, pero recuerdo que en LinkedIn estaban todos hablando de Big Data, Data Science, Crypto, AutoML, etc... Nadie especialista en estos temas, y de la nada _RAGS_, _LLMs_, _Agentic AI_, etc. y adivina: todos expertos y especialistas desde hace años.

Hay varios temas que creo que van a surgir (y que ya existen), En un proyecto reciente, pude abordar un problema que había resultado difícil de resolver utilizando enfoques más tradicionales basados únicamente en prompting o recuperación de contexto... De lo aplicado, varios de estos enfoques eran poco conocidos o poco explorados dentro del contexto en el que trabajaba (para los curiosos, lean sobre _Inductive Logic Programming (ILP)_, _Probabilistic Soft Logic_, entre otros); de hecho hice transferencia de conocimiento y llevé el método a otras áreas, obviamente mejorando mis redes y reputación dentro de la empresa.

Te aseguro, que cuando nuevos frameworks especializados en este tipo de problemas, muchos van a asegurar de que lo llevan haciendo por años, son _especialistas_, _expertos_, etc... pero nuevamente, es muy probable que pocos estén conscientes de estos enfoques... Gran parte de lo que hoy se comercializa como "AI Engineering" corresponde principalmente a integración de APIs, orquestación de herramientas y construcción de pipelines. Eso no necesariamente es negativo ya que muchos productos útiles se construyen de esa manera; pero es distinto de trabajar en problemas fundamentales de razonamiento, representación del conocimiento o inferencia. 

Las herramientas actuales permiten implementar ideas y prototipos mucho más rápido que antes. Sin embargo, comprender fundamentos, limitaciones y supuestos sigue siendo importante para abordar problemas nuevos o complejos más allá de simples integraciones ¿Cómo vas a aprovechar la IA para resolver problemas nuevos? Y te lo digo, la mayoría de las nuevas startups enfocadas en IA, necesitan especialistas y contratan investigadores e ingenieros con real expertise. Para el resto de los problemas, un buen ingeniero de software (o incluso de integraciones), puede hacerte escalar tu startup. Si llevamos el éxito a dinero (no es mi tema), es más importante formar redes, tener habilidades de venta y clientes, pero eso es otra historia y no particularmente donde yo tenga conocimiento o experiencia, por lo que no puedo hablar más de ello. Si emprendo, probablemente no sea en tecnología jeje.

# Titulares Inflados y Vicios

<div class="info-box info-box--green">
  <p><span class="info-box__label"><i class="fas fa-exclamation-triangle"></i> Disclaimer</span></p>
  <p>Antes de que me ataquen, no tiene nada de malo utilizar IA, y agregar conocimiento de herramientas/frameworks en el currículum. Pero figurar de experto/especialista es otro tema.</p>
</div>

Esto ya es obvio, y muchos usuarios de la plataforma `LinkedIn` lo han notado. Desde hace unos años, de la nada aparece una masa de expertos en IA, que figuran de _LLM Specialist_, _GenAI Engineer_, _AI Expert_, entre otros. El promedio de estos perfiles es simplemente anunciar una expertise en frameworks y APIs, y en general no proponen soluciones innovadoras a problemas abiertos, o problemas complejos que sí requieren conocimiento de IA. Esto no ocurre sólo a nivel de industria, también la academia tiene algunos vicios, e instituciones educacionales también intentan lucrar. Por ejemplo, el otro día revisando ciertas especializaciones en _IA y Data Science_ (Magíster y Doctorados), me topé con algo interesante: Había una escasez en el cuerpo académico de especialistas en IA (por ejemplo _Usar ML para X_ no te hace un especialista) ni tampoco con experiencia industrial a escala. Lo mismo con algunos otros post-grados que consisten en sólo cursos, que enseñan temas trillados y algunos ni siquiera se usan en la industria; además que sólo son una plétora de cursos, sin intención de culminar en algún trabajo que realmente resuelva un problema innovador en el área.

A los curiosos, estoy explorando relaciones e intersección teoría de números, lógica, matemáticas discretas y optimización. No perseguiré cartones que "validen" que tengo conocimientos de IA. Simplemente, seguiré mi propio camino y tengo pensado en el futuro invertir más tiempo en investigación, apuntando a IA desde otros ángulos y no de lo que te hacen creer que es útil.

Un concepto que es un poco despectivo (quizás bastante) es _API Plumbers_, que describe en esencia _Ingenieros de Integración_. No tiene nada de malo conectar APIs, y muchas veces resuelven ciertos problemas en ciertas industrias. ¿Es la forma óptima? Probablemente no, pero es conocimiento que se puede monetizar, y eso es algo bueno. Pero ello, es diferente de ser especialista, experto, etc.

# Mentorías en Tech, Cursos y Cartones

Otra opinión fuerte que tengo es de las mentorías en tech, y la venta de cursos. Estos "mentores" siempre comentan historias idealistas de que con su curso se pueden lograr sueldos altísimos, y conocimiento high-tech. Como dato curioso, a mí me han pedido mentorías, pero es algo que va contra mis principios. No lo hago tampoco por tiempo, y en general mentoreo ingenieros dentro de mi trabajo actual, ya que es un requisito y parte de mis tareas.

<div class="info-box info-box--red">
  <p><span class="info-box__label"><i class="fas fa-exclamation-circle"></i> Peligros de las Mentorías</span></p>
  <p>Tengan cuidado y vean las credenciales de los mentores, para que no gasten dinero innecesariamente y luego no tener resultados. El lucro con la venta de cursos, se ha vuelto otro vicio.</p>
</div>

## Cartones y Títulos

<div class="info-box info-box--amber">
  <p><span class="info-box__label"><i class="fas fa-lightbulb"></i> Observación</span></p>
  <p>Se romantiza mucho el "esfuerzo" en especial con cartones y cursos. Yo creo que arrogancia es pensar que con esfuerzo puedes llegar a cualquier parte.</p>
</div>

No creo en la meritocracia absoluta, sí creo que el trabajo duro es necesario. También creo que la vida es injusta. Aunque te dijera toda la ruta que seguí, no todo el mundo llega a Staff Engineer en Meta. La distribución de (E6 = Staff Engineer) es del 10%. No todo el mundo llega a Big Tech, también es una baja población. El lector puede ver hacia donde voy: no creo que el progreso profesional dependa únicamente del esfuerzo. Factores como contexto, oportunidades, intereses, capacidades individuales y timing también influyen significativamente. El trabajo duro sigue siendo importante, pero no garantiza los mismos resultados para todas las personas.

Explorar enfoques distintos suele generar escepticismo al principio, especialmente cuando se alejan de las tendencias dominantes. Aun así, muchas veces vale la pena profundizar en ideas menos populares si permiten resolver problemas reales o abrir nuevas líneas de investigación.

