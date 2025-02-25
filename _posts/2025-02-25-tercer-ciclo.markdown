---
layout: post
title:  "Mi Tercer Ciclo"
date:   2025-02-25 14:30:00 -0400
categories: swe dev
---

<div align="center">

![fl-genai](http://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/6ce5ad83ead7e76414a35ffffe14aa79f1082fa9/fl-genai.jpeg)

</div>

# Introducción

Como muchos sabrán llevo un poco más de dos años trabajando en Meta. En este post comparto algunas experiencias último fragmento en estos 2.8 años que llevo acá. Adelanto que no será un post tan largo ni con mucho detalle, y tendrá dos secciones principales:

1. Mi experiencia en este 3er ciclo (2024)
2. Algunas reflexiones/opiniones

Vale la pena leer el post completo para llegar a lo jugoso/sabroso.

## Contexto

En [otro post hablé sobre mis experiencias en Meta]({{ site.baseurl }}{% link _posts/2024-09-06-diferentes-experiencias.markdown %}), básicamente sobre los primeros dos años que tuve. En este post hablaré de mi tercer ciclo, de cómo me cambié de equipo, las diferencias.

# Mi tercer año en USA

Primero lo más importante, sobreviví 3 rondas de despidos: 2022 y 2023 hubo recortes de presupuesto y el más reciente (Febrero 10, 2025) fue por desempeño. No me referiré mucho a estos temas, ya que en mi post [_Diferentes Experiencias_]({{ site.baseurl }}{% link _posts/2024-09-06-diferentes-experiencias.markdown %}) mencioné cómo es el tema de la evaluación de desempeño en Meta (que vale la pena recordar, es muy intensa, más intensa que en cualquier experiencia previa que haya tenido).

Mi _performance rating_ de este tercer ciclo (2024) fue **Exceeded Expectations**. He ido a la baja jaja, mi primer año saqué redefined, el segundo greatly exceeded y ahora exceeded... En fin, significa que hay espacio para mejorar 😊

## Algunas Memorias Personales

Para destacar este año:

* Trabajé en problemas interesantes, en particular en el espacio de _Federated Learning_.
* Me tocó tocar e implementar bibliotecas nativas (`jni`) (android y otros dispositivos), tuve que hacer implementaciones en `java` y `kotlin`. Este último nunca lo había tocado, fue una experiencia interesante.
* He sido mentor 4 veces en Meta. Un@ de ell@s logró el rating Exceeds Expectations habiendo entrado en la segunda mitad del año. Originalmente se iba a ir por saltarse la evaluación al ser nuev@ pero le di soporte y logró algo mucho mejor. Pecho inflado cuando me agradeció mi apoyo 😊.
* Voy a empezar a ser entrevistador para posiciones de SWE/ML.
* Tuve excelente feedback de ingenieros que admiro (y tengo en un pedestal). Palabras como _top engineer in the problem space_, _standout engineer_, _great engineer_, _best in class XFN_, etc.
* Bajé más de 11 kg, estoy en mi aventura de ser influencer _fitness_ 😂.
* Rompí mi record de pull ups y estoy haciendo pull ups con +50kg (extras).
* Logré sacar el muscle up 😊. También de a poco mejorando mi press de banca.

<div align="center">

![muscle-up](https://gist.github.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/66a60e26abc012301ce7a15e98481eb0f0d53d77/muscle-up.gif)

_Fig 1: Yo haciendo un muscle up, anduve por Santiago en Enero._

</div>

## Tiempo para una Historia

A fines del 2023 me cambié de equipo, rechazando una posible promoción a E6, y tirándome hacia lo desconocido. Me transferí de un equipo de producto, a un equipo de infraestructura. Qué significa ello, en un equipo de producto los proyectos están orientados a resolver problemas con objetivos claros y específicos, por ejemplo _implementar X para aumentar el revenue en un Y%_. En este tipo de ambiente el **impacto** está alineado a objetivos globales directos. Lo que he notado, es que las iteraciones son rápidas, ya que hay que mover los gradientes en dirección a un mejoramiento de métricas que impacten el _revenue_ y los proyectos son en general cortoplacistas. Por otro lado, en un equipo de infraestructura, las iteraciones son lentas. Los proyectos son más a largo plazo y el objetivo principal es implementar infraestructura que sea escalable, confiable, fácil de usar, y con buen desempeño en medidas de eficiencia (uso de memoria, latencia, etc.). En este tipo de proyecto el **impacto** por lo general es indirecto y más complicado de medir en comparación a un equipo de producto. Cabe destacar que la dirección es a largo plazo.

## Aprendizaje Federado

En particular, mi equipo actual provee una infraestructura para el aprendizaje federado (_Federated Learning_, _FL_), el cual expliqué en mi post [_Introducción al Aprendizaje Federado_]({{ site.baseurl }}{% link _posts/2024-04-11-federated-learning.markdown %}). Debo admitir que inicialmente sufrí en mi fase de _ramp up_, ya que el stack era completamente diferente a lo que estaba acostumbrado. En mi equipo previo el stack era principalmente [Haskell](https://www.haskell.org/), [Hack](https://hacklang.org/) y [python](https://www.python.org/). Tuve que aprender y meterme en una base de código gigante donde el 90% del código está escrito en [C++](https://cplusplus.com/). Las únicas experiencias que había tenido en C++ fueron más bien académicas, y nada comparables a programar en este lenguaje en un sistema a nivel de industria. Mucho menos lidiar con una base de código con cientos de archivos, líneas de código, múltiples servicios, y un problema nuevo (FL) en el que no tenía experiencia.

Primeros meses, tengo la tarea de entender los diferentes ejecutores de código nativo en pytorch. No voy a discutir el stack en detalle, pero para ejecutar pytorch en multi-plataformas se puede usar [JIT](https://pytorch.org/docs/main/jit.html), básicamete scriptear los modelos y hacer tracing de todos los componentes necesarios para su ejecución. Debiese ser obvio, pero para aclarar, hay ciertas restricciones a la hora de scriptear, como por ejemplo no se pueden tener todas las directivas de `python` y sólo se puede scriptear un subconjunto de tipos/objetos. Por otro lado, el tamaño del binario importa, por lo que no podemos llegar y scriptear el modelo utilizando todos los operadores de pytorch. La ingeniería de esto no la mencionaré, pero es un tema complicado, dado que en federated learning se debe poder ejecutar el modelo de ML en múltiples dispositivos, que pueden tener diferentes APIs a nivel de OS, y también otras restricciones como uso de memoria y CPU. No se puede llegar y matar la batería del dispositivo ejecutando un proceso alto en el uso de CPU (como lo es entrenar un modelo), por ejemplo.

Por otro lado, existen otras restricciones ¿cómo se deben estandarizar las entradas al modelo (batches) para poder tener soporte en múltiples tipos de modelos que pueden tener diferentes tipos de entrada? (ej. imágenes, textos, features). Un sub-proyecto que propuse, consistió en traer claridad en estas restricciones y ambiente. Luego de múltiples desarrollos, logré implementar diferentes tipos de modelos (modelos basados en features simples, visión computacional, NLP, clasificadores, auto-codificadores variacionales _VAE_, etc) que se podían ejecutar en una arquitectura homogénea. En la implementación, para lograr esta estandarización, se requiere incrustar metadata al modelo (qué tipo de feature es, entre otras cosas). Mis XFN estaban implementando un modelo de visión computacional bastante complejo (no quiero ni recordar lo complejo que era ese código y esa red neuronal 😅). En la implementación existente del sistema que estaba en nuestro lado (código viejo), para poder normalizar los datos y ejecutarlos en este motor, se debía leer la imagen y cada pixel era una feature.

Para modelos con entradas pequeñas como [FEMNIST](https://paperswithcode.com/dataset/femnist), esto funcionaba perfecto. Sin embargo, para el modelo que se estaba desarrollando, las imagenes a procesar eran mucho más grandes, por lo que había problemas de escalabilidad. No entraré en detalle, pero se me ocurrió implementar un algoritmo que redujo la cabtidad de escrituras de $O(C\cdot H \cdot W)$ a $O(C)$ (donde $C$ es el número de canales, $H$ la altura y $W$ la anchura de la imagen). Esto solucionó el problema de escalabilidad y permitió a los XFN a continuar explorando FL. Como el lector se dará cuenta esto era un _launch blocker_.

En fin, hubo una plétora de _launch blockers_ durante el año (encriptación, problemas con _backpropagation_, observabilidad, exportación de operadores, _model authoring_, _you name it_), que afortunadamente como logré familarizarme rápido con el código, solucioné de forma satisfactoria. Lo curioso es que me tocó meter mano en base de código ajena, lo que fue una experiencia enriquecedora, que añadió más cicatrices a mis años de circo lidiando con báses de código gigantes, desconocidas y de forma rápida. El impacto, logramos poner en producción el primer modelo de visión computacional utilizando FL, un logro nada menor.

## Un poco más de on-device training

Otro proyecto interesante en el que trabajé fue hacer fine-tuning _on device_ de LLMs usando [ExecuTorch](https://pytorch.org/executorch-overview). Los resultados fueron presentados en la [Pytoch Conference 2024](https://pytorch2024.sched.com/event/1fHln/executorch-beta-and-on-device-generative-ai-support-mergen-nachin-mengtao-martin-yuan-meta) (minuto 16:24). Esto fue un gran desafío y tiene un impacto a nivel de industria, lo que hace que me sienta orgulloso del trabajo y los proyectos que he escogido. Por supuesto, hay consideraciones técnicas importantes, y hay que ser muy orientado al detalle: Tener benchmarks, asegurarse de una correcta implementación a bajo nivel, tener claro cómo delegar instrucciones y que el backend tenga soporte de las distintas operaciones a nivel de OS. Por ejemplo, es una mala idea usar `malloc` en _embedded systems_. Dejo al lector investigar por qué...

## Simulaciones FL

Otro proyecto interesante fue escalar un simulador de FL, proyecto OSS [flsim](https://github.com/facebookresearch/FLSim). Sin embargo, para uso interno, había problemas de escalabilidad entre otros detalles. Escalar el sistema no fue simple. Además de la lógica de serialización y deserialización de los modelos, para simular dispositivos el servidor, hay otros detalles, como el hacer la transferencia por red de forma eficiente. En este caso utilizamos APIs de bajo nivel `isend` e `irecv` (provistas por `pytoch` distributed). Después de una fase experimental de benchmarks, probamos distintos tipos de tareas (SFT, clasificación, procesamiento de imágenes). Finalmente, en temas de _throughput_ nos decantamos por una solución usando el protocolo `RPC` (habían otros temas de uso de memoria, por ejemplo entrenando LLMs que fueron un gran desafío técnico). Para hacer escalar la carga de datos, y la ejecución, hubo que diseñar un sistema como el mostrado en la figura 2. Hubo otros desafíos como por ejemplo no agregar todos los modelos de los clientes en el servidor (cuello de botella), si no que distribuir la carga de forma inteligente para aprovechar de mejor manera la concurrencia.

<div align="center">

![fl-dist-design](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/fcb3d8cdcdf5fbcfdcf4a4699e9d48dd42a4d4f9/sys-design-dist-fl.png)

_Fig 2: Diseño de sistema distribuido para simulaciones FL._

</div>

El diseño está muy simplificado, hay muchos detalles técnicos que no mencionaré. Sin embargo, para un caso de uso importante, logramos superar la meta y throughput de `100k users/min` a `250k users/min`, que es bastante bueno para simulaciones a gran escala. También logramos mejorar la carga de datos para soportar billones de entradas. Yo diseñé el sistema y los sub-sistemas, hice algunas implementaciones más complejas y el _boilerplate_. Luego otros ingenieros y researchers tomaron mi diseño y lo completaron. Básicamente lideré múltiples proyectos.

# Reflexiones

No tengo mucho más que contar. Sólo para agregar, yo también me estreso y no tengo la respuesta para todo. Sin embargo, en situaciones extremas se activa mi _modo de supervivencia_. Muchas veces estuve a contra-reloj con problemas que parecían no tener solución y de alguna forma logré **hackear el universo y hacer aparecer una solución**. Lo que he podido notar, y que me han comentado algunas personas, es que no toda la gente tiene ese _modo_ (aunque para mi sea algo natural).

Por otro lado, vi muchos casos de ingenieros que fueron etiquetados como NRA (_Non-Regrettable Attrition_). Algunos se memorizaron mucho el proceso de entrevistas, quedaron en niveles más altos del que deberían haber estado y lamentablemente _los dejaron ir_. Otros, nunca fueron promovidos en los límites de tiempo establecidos y después de un par de semestres, también _los dejaron ir_. En fin, no todos son un buen ajuste para todos lados. Incluso yo no he sido buen ajuste en algunas empresas (por suerte no he quedado en dichos procesos). Sin embargo, también presencié casos en que el ingeniero tuvo muy buen desempeño, pero una mala mitad. Siempre hay externalidades como: problemas familiares, enfermedades, etc.

Nadie _está a salvo_, definitivamente **nadie es imprescindible**. Me tiene con dolor de estómago pensar en el futuro, en el sentido de que yo igual podría tener un _mal periodo en la chamba_. Sin embargo, el hecho de que **existe una alta probabilidad de ser despedido** en la vida laboral, me deja menos amargura y tampoco es el fin del mundo. Creo que lo más importante es cuidar la salud y las relaciones personales. Lamentablemente, por ahora yo soy un trabajólico 😅, pero estoy intentando definir un límite entre lo laboral y lo personal...

# Cierre

Casos anecdóticos, he recibido comentarios como: "se presenta como experto en ML" o "cómo te contrataron en Meta", etc. Yo **nunca me he autoproclamado experto**, pueden revisar mi LinkedIn de pies a cabeza y no van a encontrar ese adjetivo (lo uso muy rara vez).

Creo de todas maneras que hago cosas que no son tan comunes:

* He contribuido al open-source en proyectos grandes
* he contribuido al estado del arte en la literatura (en ML y lingüística computacional), en la figura 3 muestro un pantallazo a mi google scholar. Puedo ser un pelagato, pero el hecho de haber publicado en revistas donde la revisión es rigurosa, me imagino que me da algunos puntos.

<div align="center">

![dpalma-google-scholar](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/0431d1bb37ebb4690d5bf8571646104b0d4b48fa/citations.png)

_Fig 3: Citas en google scholar._

</div>

* Nunca he tomado el _atajo_ de ir a copiar y pegar tutoriales, o revisar artículos que no hayan sido revisados y contrastados.

Para el mito _los ingenieros de FAANG no saben tal tecnología, o memorizan leetcode, blah blah_. Existe una **bi-direccionalidad**. Me ha tocado ver ingenieros considerados _senior_ en algunas empresas (con todo el stack + cloud, etc.) que han echado a los 6 meses porque no cumplieron con las expectativas.

En mi opinión se cumplen 3 axiomas:

$$\text{Saber } N \text{ "tecnologías"}\nRightarrow \text{ser buen ingeniero} \quad (1)$$

Hace unos años se salpicoteaban bastante los términos "data pipeline" _"big data" "spark" "cloud"_. Ahora es _"DB vectoriales" "agentes" "LLM"_ (generalmente autoproclamados expertos).

$$ \text{Llegar a una FAANG} \nRightarrow \text{ser buen ingeniero} \quad (2)$$

Lamentablemente he visto ingenieros que no tuvieron buen desempeño en sus primeros 6 meses - 2 años.

$$\text{No trabajar en FAANGs} \nRightarrow \text{no ser bueno/top} \quad (3)$$

He conocido ingenieros _No FAANGs_ excelentes a lo largo de mi carrera.
