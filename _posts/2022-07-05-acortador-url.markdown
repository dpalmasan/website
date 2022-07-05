---
layout: post
title:  "Tips para Entrevistas TI y un poco de \"System Design\""
date:   2022-07-04 16:03:03 -0400
categories: entrevistas ti
---
*Aclaración*: Me disculpo de antemano si abuso de los anglicismos (no me gustan), pero lamentablemente es un vocabulario común en redes profesionales 😔.

*Obs*: Si quieres ir al problema de código del día, [click aquí](#code-challenge).

No es sopresa que en estos tiempos muchas empresas han optado por estrategias conservadoras, debido a diferentes eventos que han afectado el mercado a nivel mundial. Ello también ha impactado en el mundo tecnológico y entre las consecuencias de estas estrategias se encuentra el _hiring freeze_ (paulatinamente pausar las contrataciones). Esta incertidumbre hace que un proceso que de por sí es estresante (buscar trabajo, proceso de entrevistas), cause más ansiedad de lo normal. Es sabido en el mundo TI, en especial en roles ligados al desarrollo (ej. _software engineer_, _machine learning engineer_) el proceso de entrevistas es estresante, causa ansiedad, y a veces llega a ser complicado. Existe un debate constante en redes profesionales (como _LinkedIn_) sobre cómo debiese ser el proceso de entrevistas para desarrolladores, algunos se quejan de los _code challenges_ en vivo, otros se quejan de las tareas (_homework assignment_), otros se quejan de las entrevistas no técnicas, en fin, quejas hay para todo. Las únicas verdades, en mi opinión:

* A nadie lo van a contratar por ser `X` o por tener un currículum bien pulido (a veces exagerado) (excepción: ha hecho algo muy relevante en la industria)
* Nadie tiene garantizado un puesto, o el "éxito" (por eso hay que ser agradecido e intentar ayudar siempre; dejar el ego a un lado)

Volviendo al tema de las entrevistas, en general creo que los procesos de selección que más ansiedad producen, son los que involucran demostrar conocimiento técnico "en vivo". En general, los _homework assignments_ son simples por los plazos y porque uno puede googlear. Un ejemplo anecdótico, una vez hice una prueba de `ReactJs` sin saber nada del framework, pero entre la documentación y googleando saqué el puntaje máximo (era de estas pruebas tipo hackerrank). Otras veces estas tareas, las usan malintencionadamente para que les hagas trabajo gratis, me ha pasado y se nota demasiado; esto en general ocurre en roles de diseño, y _data science_, me imagino que por la naturaleza de los entregables. Los procesos de entrevistas que involucran pruebas "en vivo" generalmente tienen las siguientes componentes:

* Rondas de _code challenges_, para probar la capacidad de la persona a la hora de resolver problemas.
* Rondas de _system design_, para medir el _seniority_ del postulante.

Si bien este proceso tiene ciertos vicios, en la práctica se da que minimiza los falsos positivos (pudiendo obtener falsos negativos). Desde el punto de vista de la empresa, y por lo que he leído en blogs, imagino que la razón principal de ello es que contratar es caro. En mi experiencia este proceso es divertido, excepto raras veces en que salen con un problema de programación dinámica (tipo _hard_ de leetcode) y esperan que lo resuelvas en 20 minutos. Pero fuera de esos casos raros, el proceso es divertido, pero causa ansiedad. Si soy honesto, resolver problemas de código no es complicado, ya que en general en estas entrevistas preguntan problemas de dificultad media (a veces fácil, y rara vez difíciles). En general se busca evaluar la capacidad analítica del postulante (generalmente es libre de escoger el lenguaje de programación que guste), y también conocimiento básico de algoritmos y estructuras de datos (ordenamiento, grafos, _heaps_, árboles binarios, _arrays_, testing, capacidad de plantear una solución y luego implementarla). Como al final de cada entrada pondré algún ejercicio de este tipo, en esta oportunidad me enfocare en las entrevistas que sí me asustan: _System Design_. En este tipo de entrevistas, el problema en general es ambigüo, no existe una solución clara, y el postulante es el que debe hacer supuesto, aclarar requerimientos, hacer cálculos (latencia, consultas por segundo, memoria, escalabilidad, uso de red, etc.).

En esta entrada de blog, diseñaremos el cliché de los sistemas, un acortador de `URLs` (tipo _tinyurl_).

## Acortador de URLs ##

Cuando te encuentras en una entrevista de diseño de sistemas, el enunciado probablemente sea algo como: _Diseñe un acortador de urls_. Como se puede notar, el enunciado es bastante ambigüo e incompleto, y empezar a diseñar "a la mala" es un mal indicio para el entrevistador. Algunas preguntas que se pueden hacer:

* ¿De qué largo tienen que ser las URLs sin considerar el dominio? ¿Es 7 caractéres adecuado?
* ¿Es necesario que un usuario se registre para usar el sistema?
* ¿Cuál es la carga esperada de usuarios?
* ¿Cuál es el largo máximo esperado para una URL?

Dependiendo de las aclaraciones que haga el entrevistador, se debe tomar nota de los requerimientos y hacer estimaciones, como cantidad de consultas por segundo, espacio en disco requerido, tolerancia a fallas, entre otras cosas. Supongamos que iterando con el entrevistador obtuvimos las siguientes respuestas:

* El largo de 7 caractéres es adecuado
* Se esperan urls de hasta 500 caractéres
* Promedio de `10000` consultas al día
* Prototipo inicial vivirá 1 año
* Por el momento no preocuparse de la tolerancia a fallas

Con ello se puede hacer una estimación:

* Largo de URL `500 B`
* Largo del hash `7 B`
* Cantidad de consultas diarias `10000`
* No se requiere registro de usuarios

Por lo tanto el espacio requerido diario `507 B * 10000` sería aproximadamente `5 MB` y considerando 1 año `~2GB`. Para soportar este caso de uso, se puede utilizar cualquier motor de base de datos (relacional o no relacional), como de momento la tolerancia a fallas no es un impedimento, se puede dejar, pero valdría la pena mencionar que se pueden utilizar réplicas de la base de datos para tolerancia a falla. Pero para este caso se considerará un servidor, y se puede comentar que luego del diseño inicial se propondrán ideas para escalar el sistema.

El flujo del sistema se muestra en la figura 1. Básicamente, se hace una petición a la URL acortada, y el sistema redirecciona a la URL original. Un posible diseño de este sistema se muestra en la figura 2, en esencia se requieren los siguientes componentes:

* Un server que permita crear URLs acortadas y también redireccionar a las URLs originales
* Una base de datos para almacenar el mapeo de las URLs acortadas a las originales
* Un caché para evitar hacer múltiples llamadas innecesarias a la BBDD.

<div align="center">

![Funcionamiento Acortador de URLs](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/99e6ae0ed96337ad83354e04bdb87b2e309837b8/url-shortener.png)

_Fig 1: Funcionamiento del acortador de URLs._

![Diseño en Alto Nivel del Acortador de URLs](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/99e6ae0ed96337ad83354e04bdb87b2e309837b8/url-shortener-hl.png)

_Fig 2: Diseño en alto nivel del acortador de URLs._

</div>

El modelo de datos puede ser el siguiente:

| Col         | Type     |
|-------------|----------|
| url_id      | `int64`  |
| url         | `string` |
| shorten_url | `string` |



{% highlight golang %}
var codes = [...]string{
	"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
	"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
	"k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
	"u", "v", "w", "x", "y", "z", "A", "B", "C", "D",
	"E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
	"O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
	"Y", "Z"}

func Base62(number int64) string {
	result := ""
	for number > 0 {
		result = codes[number%62] + result
		number /= 62
	}
	return result
}
{% endhighlight %}

## Code Challenge ##
