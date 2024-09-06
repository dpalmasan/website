---
layout: post
title:  "Diferentes Experiencias"
date:   2024-09-06 11:23:00 -0400
categories: swe dev
---

# Introducción

Como muchos sabrán llevo un poco más de dos años trabajando en Meta. En este post comparto algunas experiencias en estos 2.4 años que llevo acá.

## Contexto

Dadas mis experiencias previas (nunca llegué al titulo "senior", excepto en un lugar cuyos títulos estaban infladísimos). Luego de pasar por el proceso de selección que consistió en 4 entrevistas de código y dos de diseño de sistemas y una de "comportamiento" (técnica STAR), quedé como ingeniero de ML E4 (o Semi-Senior). Debo admitir que cuando di las entrevistas para Meta estaba relajado y me daba lo mismo el resultado, ya que tenía dos ofertas en empresas "prestigiosas", una como _Tech Lead_ (MeLI) y en otra como SWE 2 (Microsoft); procesos de selcción similares. Sin embargo, se dio la casualidad de que recibí una oferta de Meta, donde me reubicaría en USA, y como vivir en el extranjero era uno de los logros que quería en mi vida, acepté (además los beneficios eran buenos).

## Primer Año

Todo complicado, al no saber nada. Al principio un poco estresado, tenía que sacar el SSN (similar al RUT en Chile), abrir cuenta en el banco y buscar departamento. En esto me demoré aproximadamente 1 mes, y con el tema del arriendo, estuve apretando el c*lo como dicen en Chilito ya que tuve un problema de último minuto jaja.

Comienza mi aventura en Meta, en ese tiempo tenían el proceso de _bootcamp_, donde uno pasar por una suerte de inducción en cómo funciona Meta, uno tiene un mentor de bootcamp, y se van haciendo tareas simples para distintos equipos (arreglar errores de lint, agregar tests, etc.). Entre eso hay que hacer cosas administrativas (sacar badge, registrarse en distintos temas y grupos etc.), cosas que no tenía idea jaja, ni siquiera tenía escritorio asignado (porque no era miembro de ningún equipo aún). Luego de estar un par de semanas en el bootcamp, tocó buscar equipo. Tuve varias conversaciones, me "senté" con distintos equipos, esto involucra ir a reuniones, tomar alguna tarea específica para dicho equipo, etc. Al final, me convenció el equipo de integridad del negocio, el manager me convenció jaja tenía un buen discurso.

Primer año en el equipo, involucra conocer a la gente: XFN, el equipo en sí, los proyectos, etc. Al principio, me asignaron tareas como "onboarding" y completar algunos proyectos. La verdad hice todo y pensaba que iba bien encaminado, hasta que en una conversación de almuerzo se tocó el tema de la evaluación de desempeño (o en Meta el PSC: Performance Summary Cycle). En dicha conversación se tiraron anécdotas, entre otros temas los cuales me hicieron entender dos cosas:

1. PSC es estresante
2. PSC es algo serio

De curioso me puse a investigar los distintos ratings de la evaluación de desempeño:

* Below Expectations
* Meets Most
* Meets All
* Exceeds Expectations
* Greatly Exceeded Expectations
* Redefined Expectations

En ese momento se activó mi sentido de supervivencia, y empecé a buscar problemas a resolver, para hacer más de lo "que me pedían", ya que entendí que sólo hacer lo justo **no** era suficiente. Lo primero que hice fue intentar interactuar más con los XFN y encontré un problema interesante a resolver. Todo esto mientras resolvía lo que ya se me había asignado. Luego llegó el "touchpoint" que es una previa al PSC final, este ocurre a mitad de año, y recibí feedback de pares. En general excelente feedback, excepto del tech lead. Este feedback no era negativo, pero en resumen: No estaba resolviendo problemas con gran impacto. Impacto es una palabra que se escucha casi todos los días en Meta. Debo admitir que esto me dejó con dolor de estomago y diarrea jaja.

En la segunda mitad, me dediqué a entender qué era el impacto y empecé a resolver problemas que lo tuvieran. Generalmente eran problemas que se esperaba tomaran meses y probablemente no se completarían en dicha mitad; sin embargo me propuse a completarlos. Cabe destacar que existen cuatro ejes que se evalúan en el PSC:

* Impacto: Qué impacto tuviste durante el año (e.g. generaste revenue, mejoraste alguna métrica que posiblemente mejore este, etc)
* Better Engineering: Código escalable, tolerante a fallas, buen coverage de testing, manejo de incidentes, etc.
* Dirección: Ser capaz de darte dirección y dar dirección a otros
* People: Mentorear, entrevistas, organizar eventos, etc.

En la segunda mitad logré completar proyectos con mayor impacto que el esperado (2x), además de terminar mis propios proyectos personales, que sí tuvieron impacto. Para los curiosos, implementé una lib para análisis de anuncios, implementé un modelo de detección de malos actores basado en NLP, inventé un modelo basado en grafos de negocios con LSA (Latent Semantic Análisis) para detección de cuentas comprometidas, anomalía de presupuestos en actividad de cuentas de anuncios para detección de malos actores, entre otros proyectos más pequeños.

En la segunda mitad tuve feedback perfecto de un E6, más excelente feedback de los XFN. Mi manager en varios 1:1 me dijo que a mi probablemente me habían hecho _low balling_ (que me contrataron en un nivel más bajo del que realmente era), y el último feedback que me dio era que yo estaba haciendo trabajo de E5 que tendría un rating _greatly exceeded_, al ser E4, saqué _redefined expectations_ y me promovieron a E5 en un año (que fue bastante rápido).

### Extras

Como nota aparte, más cosas me dejaron con diarrea, como por ejemplo presenciar recortes de presupuestos y dos despidos masivos. Digamos que en varios segmentos del año no lo pasé bien.

## Segundo Año

Este año me cambiaron de manager, ya que hubo una re-estructuración (posterior a los despidos), la cual involucraba "estrategia de achatamiento" cuyo objetivo era reducir las jerarquías y tener más SWEs que managers. Mi nuevo manager me dijo que ahora como E5 las expectativas eran más altas, pero que estaba entusiasmado con trabajar conmigo dado mi trabajo y mi rating del año previo. Nuevamente, la gente esperando mucho de mi, otra vez dolor de estómago, ansiedad, etc.

Este año ya pasé de ser un NN a ser alguien más o menos "famoso", por las contribuciones y los proyectos originales que había presentado (en especial por el impacto). Entre varias conversaciones, conocí a un "partner" con el cual tomamos un proyecto gigante que tenía múltiples sub-proyectos. No me quiero alargar mucho, pero resolvimos todo tipo de problemas y logramos mejorar la precisión de nuestros sistemas de 10% a 40% reduciendo el costo humano y agilizando la reacción a incidencias (lo que se traduce en más millones para la empresa), creamos múltiples sistemas:

1. Detección de anomalías en la publicación de anuncios basada en heurísticas
2. Detección de cuentas comprometidas en base a imágenes y vídeos
3. Detección de cuentas comprometidas basadas en NLP + heurísticas
4. Reducción de TAT en revisiones manuales de cuentas comprometidas
5. Priorización de revisiones humanas basadas en un modelo de NLP + similitud de imágenes
6. Detección de clústers de redes y dominios para "banear" sitios de actividad ilegal
7. Contra-estrategia de malos actores basada en "velocidad de campañas"
8. Fricción en publicación de anuncios sospechosos para habilitar revenue

Hasta ese punto y en conversaciones con este nuevo manager (que durante el año fue bastante exigente y me dejó con dolor de guata), me dio la "señal" de que mi rating sería EE (exceeded expectations), sin embargo yo quería más, estaba obsesionado y con mucha ambición y le mencioné que quería al menos GE. GE es equivalente a ser un E6 que saca EE, me respondió. Había un proyecto que nadie quería hacer, migración de un sistema gigante, básicamente un motor que estaba hecho en Haskell al lenguaje Hack (PHP con esteroides). Ese proyecto se estimaba que tomaría un año. Me dijo, si logras hacerlo antes de que termine el año, voy a pelear en el stack ranking para que tengas GE. Así que manos a la obra, cuento corto lo logré en aproximadamente 4 meses (tuve que ser una máquina).

Al momento de peer review, mucho feedback positivo. Le pedí nuevamente al tech lead que me diera feedback y esta vez me dijo que no se esperaba que yo fuera un "monstruo" y el feedback fue extremadamente positivo.

Finalmente me dieron el rating prometido (GE) y un jugoso bono jaja.

## Tercer Año

A fines del segundo año, le hice saber a mi manager que me cambiaría de equipo. Ahora estoy trabajando en `Pytorch` (biblioteca de deep learning), en un equipo de infra (distinto a mi equipo anterior que era de producto). Comencé con un manager, que se cambió de equipo y ahora tengo otro manager. Este nuevo manager es como un _coach_, pero muy exigente. Hasta ahora me trata de mover para que actúe como un E6, lo que, nuevamente me deja con dolor de estómago. No voy a dar detalle de lo que estoy trabajando ahora, pero quizás en un post el próximo año cuento como me fue. Por ahora, veo todo en una nebulosa y no tengo claro qué rating tendré. Mientras no sea MM, todo bien.

Como spoiler, ser un tech lead no es sólo liderar, sino que es hacer que otros lideren. Y liderar en Meta, tiene otro significado; no es sólo saber el stack, proponer diseños y participar en reuniones. Va mucho más allá de eso, un ejemplo es proponer una visión y roadmap para los siguientes $T$ periodos de tiempo, lo cual para mi no es claro aún. Sin embargo, algo de eso voy haciendo; pero intento reducir el estrés.

## Cierre

En definitiva, muchos pueden pensar que trabajar en Meta es flores y arcoiris, pero voy a dar el spoiler: No es así. Estar en otro país dependiendo de una visa, ya suma puntos base de ansiedad. Por otro lado, se valora y se espera mucha proactividad, y tener claro el impacto. El tener claro el impacto, como decía mi tech lead previo, es un músculo que hay que desarrollar, y al menos para mi, no fue fácil. De hecho ahora estoy intentando visualizar el impacto de los proyectos que me estoy inventando y que voy trabajando. Por otro lado, el PSC es un juego difícil de jugar. Lamentablemente "hacer lo que te piden" te llevará a lo mucho a un rating MM o MA, dependiendo de las ambiciones de la persona, puede estar ok. Algunos otros queremos más.

Finalmente, a pesar de lo estresante, la ansiedad que produce, los proyectos son muy divertidos y desafiantes. Además, es la empresa que más he disfrutado para trabajar (y la que más he durado hasta ahora)...
