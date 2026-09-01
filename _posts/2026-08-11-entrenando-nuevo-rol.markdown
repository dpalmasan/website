---
layout: post
title:  "Entrenando para mi nuevo rol"
date:   2026-09-01 08:25:00 -0400
categories: swe dev machine learning
---

<div style="margin: 20px 0;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6ec83e10d208fa073efa9d9cb122e069315b8d93/title-screen.png" alt="Title Screen" style="width: 100%; height: auto; display: block;"></div>

# Introducción

Como mencioné en un post previo y en LinkedIn, estoy empezando una nueva aventura. Como siempre, ya tengo pánico escénico, por lo que estoy siguiendo mi misma filosofía de siempre:

* Cuando quería superar mi récord de 5K a menos de 20 minutos: practiqué trotes en zona 2 en distancias largas, para mejorar mis habilidades aeróbicas. Luego practiqué trotes con aumento de velocidad. Finalmente, después de semanas de entrenamiento, logré batir mi récord.
* Para lograr sacar un muscle up, entrené meses haciendo pull ups y carga progresiva: 5kg, 10kg, 15kg, hasta llegar a 50kg. Una vez ganada la fuerza, la habilidad se transfiere y pude hacer muscle ups.
* En mi experiencia pasada en Evernote, tomé un rol que nunca había ejercido: Data Engineering. Antes de empezar, invertí en libros y estudié procesamiento distribuido, diferentes motores de bases de datos y arquitecturas. También un poco de sistemas operativos.

El lector ya podrá inferir hacia dónde va la cosa. Ahora estoy entrenando mis habilidades de Machine Learning, ya que voy a trabajar en un dominio en el cual no tengo mucha experiencia. En este post muestro cómo estoy entrenando para aprender mejor Reinforcement Learning.

Al final del post hay un demo de lo que logré hacer jeje.

# Hackeando mi propio juego

En una de mis caminatas, se me ocurrió una buena forma de entrenar Reinforcement Learning. Hace un tiempo, empecé el desarrollo de un juego como desarrollador Indie. Para los curiosos, mi juego se puede jugar desde el navegador en la plataforma itch.io (es solo un demo): [The Ghost of the North](https://dpalmasan.itch.io/the-ghost-of-the-north).

Volviendo a lo anterior, lo que me pregunté fue: ¿Puedo hacer un agente con AI que aprenda a jugar mi juego?

## Primer Intento: Aprender a jugar mirando la pantalla

Empecé el desafío. Lo primero que se me ocurrió fue "Reinforcement Learning" (RL), vía métodos clásicos de "Q-Values". El primer desafío fue cómo manejar los inputs. Intenté hacer un cliente separado en Python que jugara el juego y estimara los estados del juego para aplicar RL a partir de los píxeles, usando Computer Vision. No haré una descripción exhaustiva de por qué fallé miserablemente, ya que el espacio de estados explota combinatoriamente. Y hay otras complejidades, como estimar el layout del mapa, enemigos, etc.

## Segundo Intento: Features representando estados y telemetría

Telemetría. En lugar de estimar el espacio de estados utilizando capturas del juego, agregué telemetría al juego y definí los estados a partir de un tensor de features:

* Vida
* Vecindad en el mapa (estimar tiles "pisables", pits)

Intenté aplicar Q-Values y PPO. Entrené al agente a jugar el primer nivel del juego y fallé miserablemente. Al parecer, el nivel es muy complejo para lograr capturar todos los detalles, así que finalmente terminé con un agente que se quedaba trabado oscilando, haciendo acciones innecesarias, como por ejemplo saltar, atacar, avanzar, luego retroceder, pero todo se veía bastante aleatorio. También tuve varios problemas de implementación que no entraré en detalle porque fueron muchos.

## Tercer Intento: Curriculum de Aprendizaje

Se me ocurrió aplicar mi filosofía: entrenar los fundamentos y luego usar transferencia de conocimiento. El curriculum de aprendizaje que definí fue:

1. Nivel simple, línea recta. El agente tiene que aprender que si no hay razón para saltar o atacar, debiese sólo caminar hacia el objetivo (final del mapa).
2. Agregué un precipicio. Ya que el agente aprendió a caminar, lo siguiente es que aprenda a saltar cuando sea necesario.
3. Un nivel simple de línea recta, pero con un enemigo que dispara bolas de fuego, para que el agente aprenda a esquivar y luego atacar.
4. El mismo nivel previo, pero con otro enemigo que corre para embestir al jugador.
5. Un nivel con obstáculos (cubo de hielo) en el que el personaje tiene que saltar para atravesar.
6. Un pequeño nivel con todos los obstáculos previos, precipicios y enemigos.

### Curriculum

<table>
  <thead>
    <tr>
      <th>Etapa</th>
      <th>Ejemplo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Línea recta</td>
      <td><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6ec83e10d208fa073efa9d9cb122e069315b8d93/training-curriculum-walk.png" alt="Linea recta" style="display: block;"></span></td>
    </tr>
    <tr>
      <td>Precipicios</td>
      <td><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6ec83e10d208fa073efa9d9cb122e069315b8d93/training-curriculum-jump.png" alt="Precipicios" style="display: block;"></span></td>
    </tr>
    <tr>
      <td>Nivel simple</td>
      <td><span style="background-color: black; display: inline-block; padding: 5px;"><img src="https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6ec83e10d208fa073efa9d9cb122e069315b8d93/training-curriculum-mini-level.png" alt="Nivel simple" style="display: block;"></span></td>
    </tr>
  </tbody>
</table>

Finalmente, un nivel complejo (parte del primer nivel de mi juego).

# Demo

<video controls style="width: 100%; height: auto; display: block; margin: 20px 0;">
  <source src="https://gist.github.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6ec83e10d208fa073efa9d9cb122e069315b8d93/rl-demo-2026-09-01_07.43.27.mp4" type="video/mp4">
  Tu navegador no soporta el elemento de video.
</video>

# La matemática de fondo (Setup)

El juego corre un paso de decisión a la vez. En cada paso, el agente recibe una fotografía del mundo —23 números: dónde está el objetivo, qué tan cerca está el borde del pozo más cercano, si hay un enemigo o una bola de fuego cerca, cuántos corazones le quedan, etc. Luego elige una de 7 acciones (`left`, `right`, `jump_right`, `attack`, ...). El juego ejecuta esa acción durante 4 frames de física y devuelve la siguiente fotografía. Y repetimos hasta que el oso llega al objetivo o muere.

El agente en sí es una pequeña red neuronal: entran 23 números -> dos capas de 64 neuronas -> salen 7 puntajes, uno por acción. Se aplica *softmax* para convertir esos puntajes en probabilidades y el agente muestrea una acción a partir de ellas. Aprender consiste en ir ajustando los pesos de la red para que las buenas acciones tengan cada vez mayor probabilidad.

## Qué significa "bueno": Recompensa (*reward*)

En cada paso, el agente recibe un número: el *reward*. Su objetivo completo es maximizar el reward total durante un episodio, descontando ligeramente los rewards futuros (un reward ahora vale un poquito más que el mismo reward recibido más adelante):

$$
G = r_0 + \gamma r_1 + \gamma^2 r_2 + \dots \qquad (\gamma = 0.99)
$$

El truco está en diseñar los rewards de manera que maximizar $G$ sea equivalente a jugar bien. En nuestro caso:

* **Progreso hacia el objetivo.** Definimos $\phi(s) = -k \cdot(\text{distancia al objetivo})$. En cada paso, el reward incluye $\phi(s') - \phi(s)$: es positivo cuando te acercaste al objetivo y negativo cuando te alejaste.

  A lo largo de una partida completa, esto suma $k \cdot (\text{distancia recorrida})$, así que en la práctica es simplemente "te damos recompensa cada vez que avanzas hacia el objetivo", nada más.

  (Una versión anterior también consideraba accidentalmente una pequeña cantidad por cada paso que el agente permaneciera cerca del objetivo. El agente descubrió que podía explotar esto quedándose dando vueltas junto al objetivo sin terminar el nivel... esto fue eliminado.)
* **−0.04 por paso.** Demorarse es algo negativo, y evita que el agente oscile sin hacer nada.
* **+12 por llegar al objetivo, −25 por morir, −2 por agotar el tiempo.**
* **−15 por recibir un golpe.** Más que lo que vale llegar al objetivo, de manera que cualquier episodio en el que el agente recibe daño obtiene peor puntuación que una partida perfecta. Morir sigue siendo el peor resultado posible, para que el agente nunca decida lanzarse a un pozo simplemente para esquivar un golpe.

Una consideración importante: **no hay ninguna recompensa por saltar**.

El agente aprende a saltar el pozo simplemente porque no saltarlo significa recibir −25 y perder el +12 de llegar al objetivo. El "cuándo saltar" lo descubre por sí mismo.

## Cómo aprende realmente: PPO

Después de recolectar varios miles de pasos, le preguntamos, para cada acción que tomó: ¿el resultado fue mejor o peor de lo que esperábamos? Esa "sorpresa" es lo que se conoce como "ventaja" o *advantage* $A$: aproximadamente, lo que realmente ocurrió menos lo que había predicho una estimación de valor separada. Una segunda salida de la red aprende a predecir ese valor, exclusivamente para darnos una referencia contra la cual comparar los resultados. Después ajustamos los pesos. La parte ingeniosa de PPO es la siguiente proporción:

$$
\rho = \frac{\pi_\text{new}(a\mid s)}{\pi_\text{old}(a\mid s)}
$$

Es decir: cuánto más (o menos) probable es que la red actualizada tome esa acción comparada con la red que recolectó los datos. PPO empuja $\rho$ hacia arriba para las acciones con una buena *advantage* y hacia abajo para las malas, pero limita $\rho$ al rango $[0.8,\ 1.2]$. De esta forma, un solo episodio "de suerte" no puede hacer que la política de acciones cambie radicalmente de un momento a otro.

Este *clipping* es lo que hace que PPO sea relativamente estable: en vez de que la política tenga variaciones altas y termine colapsando, obtenemos una mejora mucho más gradual. Lo único que incentiva la exploración es un pequeño *entropy bonus*: un empujoncito para evitar que las probabilidades de acción se vuelvan demasiado extremas demasiado pronto. Ajustar esto terminó siendo bastante delicado: muy poca entropía y se llegaba a un óptimo local ("simplemente camina a la derecha" directo hacia el hielo). Luego el agente se dedicaba a hacer cualquier cosa al azar.

## El curriculum

Como mencioné anteriormente, aprender un nivel completo desde cero es demasiado difícil. El agente moriría de mil maneras distintas antes de tener siquiera la suerte de completar una partida. Así que construí una serie de mapas simples, cada uno con una sola idea:

**caminar, saltar un pozo, esquivar una bola de fuego, enemigo que embiste, saltar un bloque de hielo, las cuatro cosas juntas en un nivel pequeño, un nivel real**

Cada etapa comienza usando los pesos entrenados en la etapa anterior. De esta manera, "caminar hacia el objetivo" y "timing del salto" se transfieren a la siguiente etapa y el agente sólo tiene que aprender la nueva habilidad.

Sin embargo, hay una sutileza: esto convierte cada *checkpoint* en un especialista. El agente del mini-nivel puede jugar su propio nivel perfectamente, pero luego fracasa en la etapa del salto simple. Después de 1.2 millones de pasos dedicados exclusivamente a saltar ese único pozo, había sobrescrito el reflejo general que tenía antes.

Por lo tanto, *curriculum transfer* es un punto de partida para el aprendizaje posterior... **no es una forma de mantener las habilidades anteriores**.

## Extra: Banda Sonora

Olvidé agregarle la música del nivel a mi juego, pero tengo la banda sonora en una lista en mi canal de YouTube: [YouTube dpalmasan](https://www.youtube.com/watch?v=ecOqkAsiDpc&list=PLGOMPs6tY6Y4)

# Conclusiones

No tengo mucho que concluir, sólo que cuando digo que me gusta ir paso a paso, a esto me refiero. Mi rol involucrará más RL, y la verdad disto de ser un experto en la materia, pero conozco los fundamentos y los primeros algoritmos. Por ahora sigo entrenando y en preparación, ya que siempre entro en pánico escénico antes de comenzar una aventura. Si bien tener confianza en uno mismo es una buena característica, de todos modos es siempre bueno tener una cuota de consciencia en las limitaciones. **Al final del día, como en RL, no todo es explotar, también hay que explorar**.
