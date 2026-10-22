---
layout: post
title:  "Reflexiones de Octubre"
date:   2026-10-22 05:20:00 -0400
categories: swe dev machine learning
---

# Introducción

Post corto de fines de Octubre, hablo sobre los siguientes temas:

* Discusiones en general
* Algunas opiniones y _revelaciones_

Espero que sea de su agrado, y como siempre, opiniones constructivas.

# Responder sin comprender

## Opinión fuera de las creencias

Cuando publico una opinión que se aparta de lo que suele considerarse aceptable, a veces encuentro respuestas que parecen reaccionar más a lo que la persona interpreta que he dicho que al argumento en sí.

Creo que hay varios factores que pueden contribuir a ello: no comprender del todo el planteamiento original, interpretar las ideas desde nuestras propias creencias o tener dificultades para construir una respuesta coherente con el tema de discusión.

No creo que esto sea necesariamente intencional. Todos estamos expuestos a estos sesgos. Sin embargo, creo que una discusión mejora cuando somos capaces de distinguir entre comprender una postura y estar de acuerdo con ella.

## Argumentos sin un hilo conductor

Aquí hay un tema interesante de metacognición. La comprensión lectora y las habilidades de escritura están relacionadas, aunque no sean lo mismo. Comprender un argumento requiere identificar sus ideas principales y las relaciones entre ellas. Expresar una opinión coherente, por otro lado, requiere organizar esas ideas y construir una argumentación que tenga sentido.

# Aclaración ¿Son estos consejos?

La verdad es que no, yo no soy nadie para andar aconsejando. De hecho también me he equivocado. Y basado en algunas discusiones, a veces siento que la gente cree que soy una suerte de "genio", y nada puede estar más lejos de la realidad, mostremos un poco de "vulnerabilidad":

* Nunca fui un _crack_ de la programación.
* Yo también me equivoco y a veces tengo opiniones erradas, que tengo que cambiar: un ejemplo reciente, no me esperaba que los agentes de código evolucionaran tan rápido.
* A veces he opinado sin saber bien del tema, pero lo bueno es que me he llevado conocimiento de gente que sí sabe
* A veces también pienso que no valgo, o que _no soy suficientemente bueno_. A veces es un golpe al ego cuando la realidad te demuestra que hay personas con un nivel muy superior al tuyo, y que quizás nunca llegues a alcanzar. Pero eso no significa que hayas llegado a tu límite.

Algo que sí reconozco en mí es una gran facilidad para imaginar soluciones a problemas difíciles, incluso cuando otras aproximaciones no han funcionado. También tengo una disciplina que, en ciertos temas, llega a convertirse en obsesión. Ambas cosas me han permitido avanzar en objetivos personales y profesionales.

# Un ejemplo de mi nueva experiencia

Por alguna razón, situaciones de la vida cotidiana me inspiran a buscar soluciones para problemas en los que estoy trabajando.

No daré demasiados detalles, pero uno de los problemas que abordo en observabilidad es la reducción de falsos positivos en las alarmas que hacen "paging" a ingenieros. El objetivo es evitar que un ingeniero que esté en turno (on-call) reciba notificaciones innecesarias y, al mismo tiempo, identificar qué señales están realmente asociadas con incidentes que tienen un impacto significativo.

Curiosamente, esto me hizo pensar en las discusiones que mencionaba antes y en algo que considero fundamental: nuestra capacidad para actualizar nuestras creencias cuando la evidencia no respalda lo que inicialmente pensábamos.

En ingeniería, tampoco podemos asumir que toda señal representa un problema real. Necesitamos evaluar la evidencia, considerar el contexto y distinguir entre una anomalía transitoria y una situación que requiere intervención.

Por eso, en esta simulación estoy explorando el uso del Bayes factor para evaluar cuánta evidencia aporta una observación a favor de la hipótesis de que existe un incidente significativo.

<video controls style="width: 100%; height: auto; display: block; margin: 20px 0;">
  <source src="https://gist.github.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/b1a20e8fafa1b9eceeb0ee50611becb5cecd2e01/bayes_factor_monitor.mp4" type="video/mp4">
  Tu navegador no soporta el elemento de video.
</video>

La idea es monitorear una métrica de _healthiness_ de un servicio y distinguir los spikes que aportan evidencia de un incidente real de aquellos que podrían ser transitorios o resolverse automáticamente, por ejemplo, mediante auto-scaling.

El objetivo no es reaccionar a cada variación, sino determinar cuándo existe evidencia suficiente para justificar una alerta. El Bayes factor es una herramienta para comparar hipótesis a partir de los datos; su utilidad depende de cómo definamos esas hipótesis y de cómo interpretemos la evidencia obtenida.

Me parece interesante cómo una misma idea, ejem... no sacar conclusiones precipitadas y estar dispuestos a cambiar de opinión ante nueva evidencia, puede aparecer tanto en una discusión como en un problema de ingeniería.

# Reflexiones finales

Mi reflexión es simple: sigue tu camino, pero no te aferres tanto a él que seas incapaz de cambiar de dirección cuando la evidencia te demuestre que es necesario.

Perseverar es importante, pero también lo es reconocer cuándo estamos equivocados, aprender de los demás y adaptar nuestra forma de pensar.

Y, por supuesto, no todo depende de nosotros. El contexto y la suerte también influyen en los resultados, aunque prefiero reservar esa reflexión para otra ocasión.