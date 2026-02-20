---
layout: post
title:  "De mago a hechizo: Cómo llegué a E6"
date:   2026-02-19 19:31:00 -0400
categories: swe dev machine learning
---

# Introducción

Ha pasado un año desde mi último post. No porque no tuviera ideas, sino porque el tiempo y las prioridades se fueron a otros lados. Este sitio ha quedado casi muerto; a veces pienso en moverme a `Substack`, pero luego recuerdo que no estoy optimizando por visibilidad. Este blog sigue siendo lo que siempre fue para mí: un espacio para ordenar ideas y compartir aprendizajes. Si al menos una persona lo lee y le sirve, ya estoy pagado.

Este año fue un punto de inflexión. Pasé de **E5 a E6**, y asumí responsabilidades que me obligaron a replantear cómo genero impacto y qué significa realmente liderar desde lo técnico. Tuve que dejar de ser quien resuelve todo para convertirme en quien habilita que otros lo hagan.

En este post hablo de ese proceso. De otro año más en Big Tech, de mi experiencia como _Tech Lead_, de errores y aciertos, de aprendizajes técnicos y humanos. También comparto algunas reflexiones sobre el estado actual del mundo tech, avances personales fuera del trabajo, y cierro con pensamientos sobre la IA y su impacto.

Si llegaste hasta aquí, asegúrate de llegar leer el final 😊.


# Un año de _Tech Lead_

Luego de un año trabajando en `pytorch edge`, y algo de éxito, como por ejemplo mi trabajo siendo mencionado en la [pytorch conference 2024](https://youtu.be/45yNTi7c1Q0?t=983), decidí cambiarme de equipo nuevamente y volver a la org en la que estaba inicialmente. Las razones me las reservo, pero puedo decir que me divertí mucho en `pytorch`, especialmente aprendí mucho de código a más bajo nivel (`C++`), y nunca entendí por completo el código de server en nuestro stack de _federated learning_ 😅, pero sí aprendí del código en el lado del dispositivo: compilar bibliotecas `jni` para luego integrarlas en `kotlin` (programación a bajo nivel en su esplendor).

El contexto estaba desafiante, ya que prácticamente se re-estructuró el equipo: Quedé como único ingeniero "Senior", con el resto del equipo recién llegado a la empresa. Esto fue un desafío, pero también una oportunidad, donde decidí darle dirección al equipo y definir la "Northstar", en medio de varios desafíos técnicos y no técnicos. Ser líder técnico "no oficial" fue bastante complicado, ya que las expectativas son más de dirección y orquestación, pero aún tenía que cumplir mis labores como ML Eng: manejo y solución de incidentes, proyectos e impacto. 

Estaba acostumbrado a ser el que subía la mayor cantidad de PRs del equipo, y lo seguí haciendo, pero además tuve que dar apoyo y hacer _ramp up_ básicamente a todo el equipo. No fue fácil la verdad y cometí muchos errores, como tratar de "mantener el brillo", hasta que un día hice click: **No tengo que ser el mago del evento; tengo que ser el hechizo que habilita que otros reciban la ovación**. Por lo que cambié mi enfoque e intenté que el equipo tuviera más ownership, dando apoyo únicamente si había mucha incertidumbre y las cosas se pusieran feas. Cuento corto, dejé de subir tanto código, únicamente lo hice en casos que hubiese que salvar algún proyecto.

<div class="info-box info-box--green">
  <p><span class="info-box__label"><i class="fas fa-exclamation-triangle"></i> Disclaimer</span></p>
  <p>No siempre he hecho todo perfecto. He cometido errores técnicos y humanos, pero aceptarlos y usarlos como oportunidad para crecer me ha ayudado mucho a tener una mejor ejecución en mi carrera profesional.</p>
</div>

¿Qué es lo que llevo practicando constantemente?

1. **Consistencia**: Mantenerme activo en las discusiones técnicas, pendiente de las revisiones de código, y hacer cambios que promuevan la excelencia en ingeniería (mejores prácticas, cerrar gaps, etc.)
2. **Consciencia sobre estar equivocado**: Todos tenemos errores, pocos los aceptamos. Aceptar equivocarse es crecer. En redes sociales es común ver discusiones donde, incluso frente a observaciones técnicas válidas, algunas personas sienten la necesidad de tener siempre la razón.
3. **Curiosidad**: Estudiar constantemente, repasar fundamentos teóricos, pensar y conectar ideas.
4. **Admitir ignorancia**: _La enfermedad del ignorante es ignorar su propia ignorancia_. Posar de erudito y tener siempre una respuesta para todo no es necesariamente una buena señal; estar consciente de lo que uno no sabe deja espacio real para crecer.
5. **Tomar feedback y escuchar**: Escuchar en reuniones, no sólo hablar. Tomar nota, procesar el feedback y, si es accionable, trabajar en ello.

¿Qué construí? (créditos a mi gran equipo)

* Primer sistema en el área que utiliza _reinforcement learning_ y que tuvo un impacto **positivo en revenue (en millones de USD)**
* Migración de un sistema legacy a una plataforma unificada con otras orgs, para mejora de observabilidad, logging y confiabilidad (reliability)
* Manejo de incidencias críticas, **previniendo millones en revenue leakage en USD**
* Sistema de LLM para revisión automática de incidentes y rollback en el contexto de advertising (**impacto de millones de USD en revenue**)
* Arquitectura y visión de sistemas multi-agente para el área
* Visión a largo plazo (horizonte de 2-3 años)

<div class="info-box info-box--amber">
  <p><span class="info-box__label"><i class="fas fa-lightbulb"></i> Observación</span></p>
  <p>En este entorno, desde E5, no hay un <em>lo que me dicen que tengo que hacer</em>, ni tampoco la productividad y desempeño se miden <em>cerrando tickets</em>. Lo que importa para tener una buena evaluación es el impacto, y puedes tener impacto incluso sin escribir una línea de código. Como opinión adicional con los avances de los <b>agentes de IA en código</b>, esto probablemente será una realidad transversal en la industria, no exclusiva de un pequeño grupo de empresas.</p>
</div>

Para mi sorpresa, este año en el "mid checkpoint" fue la primera vez que obtuve "SAE" (Significantly Above Expectations), que es una excelente señal para la evaluación de desempeño al final del año. Finalmente, tengo actualmente los resultados de dicha evaluación y obtuve _Greatly Exceeded Expectations_ y una promoción de `E5` a `E6`.

Como tema adicional, tuve la experiencia de ser mentor, y creo que lo hice bien. A todos mis mentoreados les fue bien, y me agradecieron el apoyo y el haberlos hecho crecer (dos van a ser promovid@s pronto según se ve la cosa). Esto es algo que no me imaginaba que haría, pero llegó el día y es una experiencia enriquecedora. Ser mentor y ser _mentee_ ayuda bastante a crecer como profesional, cosa que en buen chileno, _antes miraba a huevo_, pero el tiempo me demostró estar errado, y bueno, al final todo fue un aprendizaje.

#  Mi Proyecto Personal (Game Dev), y off-topic

Curiosamente, inició como una exploración de _vibe coding_, y terminó siendo el despertar de mi molestia frente al _AI slop_. Empecé a desarrollar un juego desde cero en `C++` y `SDL 2.0`. Mi experiencia en `C++` no es extensa: algo de universidad y mi año previo en `pytorch edge`, donde me tocó hacer cambios en una base de código real. 

Volviendo al tema del juego, inicialmente los assets gráficos los generé utilizando IA. Si bien pude progresar, en algún momento subí un vídeo, utilizando de fondo la imagen de la pantalla de título (generada con IA), y recibí el comentario de "AI slop". Empecé a buscar en internet, ya que no tenía claro el significado, y luego de participar activamente en una comunidad de "solo dev" de juegos en reddit, mi percepción del uso de IA cambió.

Me di cuenta que internet ahora está plagado de AI slop: posts claramente generados con LLMs, discusiones donde se usan respuestas automáticas sin verificación, y “arte” que presenta deficiencias evidentes (paleta de colores, composición, etc.). Esto no es exclusivo de un lugar en particular, pero sí es un fenómeno cada vez más visible.

El juego que estoy desarrollando se llama _The Ghost of the North_ y su demo se puede jugar desde el navegador. No contiene IA, y todo el contenido (gráficos y música) fueron hechos por mi. Publicaré avances y `devlogs` del juego constantemente.

<p align="center">
<iframe frameborder="0" src="https://itch.io/embed/4260615" width="552" height="167"><a href="https://dpalmasan.itch.io/the-ghost-of-the-north">The Ghost of the North by dpalmasan</a></iframe>
</p>

## Un poco de arte para el juego
Como me gusta el arte de los juegos retro, me dije ¿por qué no intento aprender a dibujar en lugar de usar assets artificialmente generados? En esta sección comparto un poco de arte de mi juego, pixel art y composiciones musicales.

### Pixel Art

Algunos pantallazos del juego:

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 20px 0;">
  <div><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/7f2f538c847ecc655c9ac52c8427bddf9d94528e/title-screen.png" alt="Title Screen" style="width: 100%; height: auto; display: block;"></div>
  <div><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/7f2f538c847ecc655c9ac52c8427bddf9d94528e/ss1.png" alt="Screenshot 1" style="width: 100%; height: auto; display: block;"></div>
  <div><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/17275428551156e0c69d7291675cffdcb2994d72/worldmap.png" alt="Worldmap" style="width: 100%; height: auto; display: block;"></div>
  <div><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/2000fd90a8c91e308f04533033d86ee8fdd74b80/wind-armor.png" alt="Boss" style="width: 100%; height: auto; display: block;"></div>
  <div><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/7f2f538c847ecc655c9ac52c8427bddf9d94528e/menu.png" alt="Menu" style="width: 100%; height: auto; display: block;"></div>
  <div><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/2000fd90a8c91e308f04533033d86ee8fdd74b80/load-screen.png" alt="Load Screen" style="width: 100%; height: auto; display: block;"></div>
</div>

Algunos de mis sprites:

<table>
  <thead>
    <tr>
      <th>Nombre</th>
      <th>Animación</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Arachnoid</td>
      <td><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/f6ab354210f817cb5f84c17974125b664f7c5940/arachnoid-walk.gif" alt="Arachnoid" style="display: block;"></span></td>
    </tr>
    <tr>
      <td>Robot</td>
      <td><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/f6ab354210f817cb5f84c17974125b664f7c5940/robot-attack.gif" alt="Robot" style="display: block;"></span></td>
    </tr>
    <tr>
      <td>Frenzy Wolf</td>
      <td><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d0bf67434728fb8108d2e777631601f7107f3bb4/frenzy-wolf-idle.gif" alt="Frenzy Wolf" style="display: block;"></span></td>
    </tr>
    <tr>
      <td>Polar Bear Walking (varias armaduras)</td>
      <td><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/3b02e73f83d0a21df161ae0be19f923849984a9c/animation-all-bears.gif" alt="Polar Bear Walking" style="display: block;"></span></td>
    </tr>
    <tr>
      <td>Polar Bear Death</td>
      <td><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d94c1edb2065d1729e41813ee7385ab61f4233f8/polar-bear-death.gif" alt="Polar Bear Death" style="display: block;"></span></td>
    </tr>
    <tr>
      <td>Snow Robot</td>
      <td>
        <table style="border: none; border-collapse: collapse;">
          <tr>
            <td style="border: none; padding: 2px;"><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/a9edb6bc28f654f6a987efbd8414172d3b9c1536/snow-robot-presentation.gif" alt="Snow Robot Presentation" style="display: block;"></span></td>
            <td style="border: none; padding: 2px;"><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/a9edb6bc28f654f6a987efbd8414172d3b9c1536/snow-robot-cannon.gif" alt="Snow Robot Cannon" style="display: block;"></span></td>
          </tr>
          <tr>
            <td style="border: none; padding: 2px;"><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/a9edb6bc28f654f6a987efbd8414172d3b9c1536/snow-robot-dash.gif" alt="Snow Robot Dash" style="display: block;"></span></td>
            <td style="border: none; padding: 2px;"><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/a9edb6bc28f654f6a987efbd8414172d3b9c1536/snow-robot-vulnerable.gif" alt="Snow Robot Vulnerable" style="display: block;"></span></td>
          </tr>
        </table>
      </td>
    </tr>
  </tbody>
</table>

### Composiciones musicales

Soy un aficionado por la música, y en especial la música de videojuegos retro. Estoy componiendo toda la música de mi juego. Aquí algunas de ellas:

![](https://www.youtube.com/watch?v=rB4GGMpPEWc)

![](https://www.youtube.com/watch?v=eEq3HDc19lg)

## Off Topic

Este año también tuve un estilo de vida saludable, con dos grandes hitos:

1. Bajé 9% de grasa y gané músculo
2. Logré mi primer muscle up con 10lbs de peso extra (falta trabajar la técnica pero la fuerza está 😅)

<div align="center">

![title-screen](https://gist.github.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/dad2a325003ef229a5f2d26e4eca33f4b6fa215d/muscle-up.gif)

</div>

Aún trabajando en mi press de banca, pero me falta... no llevo mucho de levantamiento de pesas.

# Cierre

No mucho que decir, por ahora siento que logré una meta profesional importante, estoy en la búsqueda de algo más profundo: ¿disciplina, talento, visión? No tengo la respuesta. Seguiré persiguiendo la excelencia, seguiré entrenando, idealmente seguir contribuyendo cuando tenga algo que realmente valga la pena compartir. Me gusta rodearme de la energía que cultiva conocimiento y progreso continuo; además de compartir este proceso para estar en presencia de diferentes perspectivas.

Por otro lado, seguiré revisando los fundamentos y cuestionando algún conocimiento que no tenga solución cerrada, y veré a qué me lleva el destino. Incluso tomando todas las precauciones, el viaje no está exento de dificultades y es incierto, pero en dichas situaciones es cuando alimentamos al fantasma del aprendizaje. Aprendí bastante al dejar el protagonismo del mago y transformarme en hechizo.

<div class="info-box info-box--amber">
  <p><span class="info-box__label"><i class="fas fa-lightbulb"></i> Observación</span></p>
  <p>Ahora la parte un poco más incierta y quizás dura: el logro que comparto aquí es algo que considero genial. Sin embargo, eso no cambia la realidad de las cosas. ¿Será algo pasajero? ¿La IA nos va a reemplazar? Quizás nos reemplace en el mundo del software, y lo haga en otros rubros. Es decir, si ahora ves personas que hacen sistemas a vibe-coding, no quita que alguien en tech pueda reinventarse y explorar otro rubro, usando la IA como acelerador de años de experiencia. En fin, algo para reflexionar. Yo por ahora disfrutaré de los triunfos y bueno... si toca, habrá que reinventarse, tampoco es el fin del mundo...</p>
</div>

<div class="info-box info-box--red">
  <p><span class="info-box__label"><i class="fas fa-exclamation-circle"></i> Peligros del AI Slop</span></p>
  <p>Internet se está plagando de AI-slop. Como experiencia personal, mi feed de LinkedIn está cada vez más lleno de contenido generado con IA, muchas veces superficial o poco verificado. Cabe destacar que muchos de ellos acompañados de una imagen generada con IA (y baja calidad usualmente). Está ocurriendo algo similar con el código. Creo que hay una presión que <em>viene de más arriba</em> en adoptar las tecnologías. Mi opinión, creo que hay procesos que se pueden acelerar, pero en otros casos tener un impacto catastrófico. A modo personal, dedicaré un tiempo de entrenamiento a <em>hacer todo a la antigua</em>, así no impacto mis conexiones neuronales. Por presión, seguiré usando IA, en especial cuando me acelera ciertas tareas rutinarias de software; mi consejo, al igual que uno entrena su cuerpo, también debe hacerlo con su mente. Sigo haciendo integrales a mano aunque podría pasarlas a un computador, por ejemplo.</p>
</div>

Finalmente al lector, entiendo que estos puntos son debatibles y mi intención es precisamente invitar a cuestionar nuestra línea de base para seguir creciendo. Negarse a cuestionar, y la falta de reflexión llevan a varias rutas, pero este viaje puede terminar en la famosilla ciudad "Status Quo". Al final del día, lo que queda es tu impacto en las personas y cómo crecieron, el resto se desvanece con el tiempo y los cambios de estado.
