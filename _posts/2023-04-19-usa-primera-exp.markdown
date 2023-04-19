---
layout: post
title:  "Primer año en USA y algunas reflexiones"
date:   2023-04-19 11:10:03 -0400
categories: python algorithms classification machine-learning
---

Este post no será tan técnico, si no que escribiré algunas reflexiones sobre mi primer año en una `FAANG` (o `MAANG`) y viviendo en USA; todo esto en el contexto de los recientes despidos en mi empresa actual.

## ¿En qué estoy ahora?

Como he mencionado y podrán ver en mi perfil de Linkedin, he trabajado en múltiples roles en el mundo tecnológico:

* Quality Assurance Engineer (o SWE en testing)
* Data Scientist
* Data Engineer
* Software Engineer Backend
* Machine Learning Engineer

En mi tiempo libre también me dedico a publicar papers, en particular sobre técnicas de procesamiento de lenguaje natural (NLP) e inteligencia artificial aplicadas en el contexto educativo. A veces me dedico al open-source y también he sido autor de algunas bibliotecas como:

* [TRUNAJOD](https://github.com/dpalmasan/TRUNAJOD2.0)
* [Pylogic](https://github.com/dpalmasan/py-logic)

Actualmente (en la linea temporal en la que escribo este post) trabajo como _Machine Learning Engineer_ en [Meta](https://www.meta.com/), en particular en el área de _intrgridad de negocios_.

## Sobreviviendo 2 Despidos Masivos (a.k.a _layoffs_) y Vulnerabilidad

Este año que llevo acá ha sido de emociones intensas, no sólo por el estrés de ser _inmigrante_ (en realidad no lo soy, sólo soy trabajador temporal con visa H1B), si no que también el momento actual es complejo, con la crisis económica en particular en el mundo T.I; en parte por la sobre-contratación, probablemente relacionada con el COVID-19.

Por otro lado, he sido testigo de dos despidos masivos, uno en Noviembre de 2022 y otro hoy (Abril 19 2023). Debo decir que enfocarse ha costado bastante, en especial con esta segunda ronda de despidos que fue anunciada con meses de anticipación. Debo decir que en ambas, fue complicado conciliar el sueño y a la vez rendir al nivel esperado. Por otro lado, el tener personas de tu equipo afectadas, afecta la moral completamente (diría que incluso a nivel organizacional); y uno piensa que podría ser uno mismo el impactado, lo cual genera mucho estrés.

En mi humilde opinión, creo que este es el primer trabajo que me siento _vulnerable_, más por el hecho de que mi permiso de trabajo está relacionado al empleador, contrario a si estuviera en Chile (mi país de origen), en el cual, si me llegaran a despedir, no tendría tanta incertidumbre ni un límite tan estricto para buscar trabajo.

## Evaluación de Desempeño y la presión implícita

Meta es el primer lugar en el que he trabajado donde me he estresado con la evaluación de desempeño. La cultura en general gira entorno al _PSC_ (_performance summary cycle_), y el tema es serio, habiendo múltiples "ratings", haré la traducción al chilensis (escala de 1 a 7) para que quede claro al lector:

* No cumple expectativas (es básicamente nota entre 1-3, reprobado)
* Cumple con algunas de las expectativas (nota entre 3-4 reprobado)
* Cumple con la mayoría de las expectativas (nota 4-5, aprobado apenas)
* Cumple con todas las expectativas (nota 5-5.5, aprobado)
* Excede expectativas (nota 5.5-6.0, aprobado con distinción)
* Excede expectativas extremadamente (6.0-7.0 aprobado con distinción máxima)
* Redefine expectativas (Supera toda escala >7.0)

El promedio de personas en general obtiene _cumple con todas las expectativas_, y aproximadamente $1\sigma$ de la media obtiene _excede expeactativas_ (no es poco frecuente). Sin embargo ratings más altos son mucho más complejos, por ejemplo redefine expectativas se lo dan al top 1% de los ingenieros; obviamente la evaluación está acompañada de un multiplicador para el bono anual; pero para una idea de la diferencia, excede expectativas tiene un multiplicador de 1.25, mientras que redefine es de 2.5. Por otro lado, en la evaluación se hace ajuste de curva y [_stack ranking_](https://lattice.com/library/what-is-stack-ranking-and-why-is-it-a-problem); básicamente tienes que superar a la media de los ingenieros lo que hace que, si todos tienen buen desempeño, hay que innovar mucho más para obtener ratings más altos.

Por otro lado, las expectativas dependen del nivel, en generial como contribuidor individual (IC), están los siguientes niveles:

* E3: Junior
* E4: Semi-Senior
* E5: Senior
* E6: Staff
* E7: Senior Staff

Hay niveles más altos, pero no entraré en detalle. Todo esto es para el contexto, para que no me juzgue el lector con mis reflexiones. Siendo honesto yo entré como E4 en Machine Learning y no me querían meter a ese ciclo de entrevistas (que es más complejo que el de Software Engineer SWE) por el tema de la experiencia en ML (tenía más experiencia como SWE), pero como quería probar un nuevo rol hice la entrevista. El nivel en que uno entra está correlacionado con la experiencia previa (poca) y por el resultado de la entrevista. Yo entré como E4, y si tienen curiosidad, mi rating en el PSC fue _redefine expectativas_ y me subieron a E5 en 6 meses (ya que los primeros 3 meses uno está en _bootcamp_ y tiene que aprender lo básico de la infrastructura y hay la oportunidad de explorar distintos equipos).

Todo lo anterior no es por presumir, es para dar el contexto de que uno sí se estresa, hay presión implícita. Lo que tiene Meta respecto a otras empresas que he trabajado es que es _bottom up_ y no _top down_, es decir que uno puede resolver los problemas que uno quiera, siempre que tengan _impacto_. Y aquí está el gran problema, ¿Qué es lo que tiene impacto? El impacto tiene que ser medible e idealmente generar dinero a la empresa; uno puede tomar riesgos e intentar resolver problemas complicados, pero si no hay impacto, no sirve de mucho. También hay otros "ejes" que son criterios para la evaluación como "dirección", "personas" y "excelencia en ingeniería". Los que me conocen saben que he pasado por episodios de ansiedad bastante serios, y un estrés constante; por lo que esta presión implícita puede tener efectos severos si no se está preparado.

Cabe destacar, que los despidos en este caso no fueron por desempeño, fueron por necesidades de la empresa. Es decir, el "rating" no afecta en nada el si uno es despedido y no, y diría que es más cosa de suerte (quizás fue uno de los criterios pero probablemente con poco peso.)

## Ser un Ninja

Un tema interesante y parte enriquecedora de la experiencia, es la tecnología que hay detrás. Y voy a ser honesto, ninguna de mis habilidades por ejemplo en la nube (GCP, AWS, etc) o cualquier framework especializado, es útil aquí, ya que toda la tecnología y _frameworks_ son internos. Por ello, cuando me preguntan si hay una ruta, mi respuesta es en general un conjunto de referencias y literatura. Personalmente para ML recomiendo:

* [Designing Data-Intensive Applications](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/)
* [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/)
* [Distributed Systems](https://www.amazon.com/-/es/Maarten-van-Steen/dp/1543057381/ref=sr_1_7?__mk_es_US=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=1TDH831PPVQW6&keywords=distributed+systems&qid=1681927199&sprefix=distributed+system%2Caps%2C128&sr=8-7)
* [Operating Systems](https://www.amazon.com/-/es/Remzi-H-Arpaci-Dusseau/dp/198508659X/ref=sr_1_3?__mk_es_US=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=39S2RGOCH3YX0&keywords=operating+systems&qid=1681927218&sprefix=operating+system%2Caps%2C134&sr=8-3)
* [Mathematics for Machine Learning](https://www.amazon.com/-/es/Marc-Peter-Deisenroth/dp/110845514X/ref=sr_1_1?keywords=mathematics+for+machine+learning&qid=1681927235&sprefix=mathemati%2Caps%2C136&sr=8-1)
* [Speech and Language Processing](https://www.amazon.com/-/es/Daniel-Jurafsky/dp/0131873210/ref=sr_1_1?__mk_es_US=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=1TNNJW9CENLAD&keywords=natural+language+processing+jurafsky&qid=1681927271&sprefix=natural+language+processing+jurafsky%2Caps%2C127&sr=8-1)

Hay más, pero esos son los que más he utilizado. Mi opinión es que uno no debe saltarse la etapa de aprendizaje y comprensión, por ello soy detractor del _aprender a hacer_, y soy más del método _aprender y hacer_.

Creo que para sobrevivir aquí, y tener un buen rating hay que ser un _ninja_. Es decir, ser capaz de armar tu red, crear colaboración, proyectos; ser capaz de liderar proyectos de `0 -> 1` y que sea compatible con el stack tecnológico; tener la capacidad de digerir una base de código gigante, con poca o sin nada de guía.

Por otro lado, el buscar proyectos es tarea complicada, porque idealmente tienen que tener impacto alineados con la organización. Como dato freak, todos los ML de mi equipo son PhD, y la broma que echamos, para los que hemos estado un poco en el mundo académico, es que se siente como buscar tema de paper; es como volver al post-grado en la universidad, y buscar un tema que tenga impacto y sea lo suficientemente "innovador" para lograr esa cuota de dirección necesaria para la evaluación de desempeño.

## Conclusiones

* Este es un mal momento para estar afuera, por la crisis y la incertidumbre
* Estar en equipos de alto desempeño, activa frecuentemente el _síndrome del impostor_ y el subestimar las propias capacidades
* Algunas veces los despidos no tienen que ver con desempeño y pueden tener una componente de azar
* Más que mecanizar y _aprender a hacer_, una estrategia efectiva es _aprender y hacer_ y evitar saltarse pasos necesarios en la adquisición de conocimiento
* Cuando realmente se aprende, se puede transferir el conocimiento a cualquier tipo de framework y problemática

## Problema de Código

Este problema de código fue en una _screening_ (entrevista semi-técnica, para descarte inicial):

_Dado un string `s` que representa una operación matemática `"1 + 2*3"` crear una función que calcule el resultado de la operación. No se pueden utilizar bibliotecas, ni llamar al compilador/intérprete o parser externo (ej. usar `eval("1 + 2*3")` en `python` no estaría permitido._