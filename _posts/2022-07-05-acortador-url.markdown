---
layout: post
title:  "Tips para Entrevistas TI y un poco de \"System Design\" 💻"
date:   2022-07-04 16:03:03 -0400
categories: entrevistas ti
---
*Aclaración*: Me disculpo de antemano si abuso de los anglicismos (no me gustan), pero lamentablemente es un vocabulario común en redes profesionales 😔.

*Obs*: Si quieres ir al problema de código del día, [click aquí](#code-challenge).

No es sopresa que en estos tiempos muchas empresas han optado por estrategias conservadoras, debido a diferentes eventos que han afectado el mercado a nivel mundial. Ello también ha impactado en el mundo tecnológico y entre las consecuencias de estas estrategias se encuentra el _hiring freeze_ (paulatinamente pausar las contrataciones). Esta incertidumbre hace que un proceso que de por sí es estresante (buscar trabajo, proceso de entrevistas), cause más ansiedad de lo normal. Es sabido en el mundo TI, en especial en roles ligados al desarrollo (ej. _software engineer_, _machine learning engineer_) el proceso de entrevistas es estresante, causa ansiedad, y a veces llega a ser complicado. Existe un debate constante en redes profesionales (como _LinkedIn_) sobre cómo debiese ser el proceso de entrevistas para desarrolladores, algunos se quejan de los _code challenges_ en vivo, otros se quejan de las tareas (_homework assignments_), otros se quejan de las entrevistas no técnicas, en fin, quejas hay para todo. Las únicas verdades, en mi opinión:

* A nadie lo van a contratar por ser `X` o por tener un currículum bien pulido (a veces exagerado) (excepción: ha hecho algo muy relevante en la industria)
* Nadie tiene garantizado un puesto, o el "éxito" (por eso hay que ser agradecido e intentar ayudar siempre; dejar el ego a un lado)

Volviendo al tema de las entrevistas, en general creo que los procesos de selección que más ansiedad producen, son los que involucran demostrar conocimiento técnico "en vivo". En general, los _homework assignments_ son simples por los plazos y porque uno puede googlear. Un ejemplo anecdótico, una vez hice una prueba de `ReactJs` sin saber nada del framework, pero entre la documentación y googleando saqué el puntaje máximo (era de estas pruebas tipo hackerrank). Otras veces estas tareas, las usan malintencionadamente para que les hagas trabajo gratis, me ha pasado y se nota demasiado; esto en general ocurre en roles de diseño, y _data science_, me imagino que por la naturaleza de los entregables. Los procesos de entrevistas que involucran pruebas "en vivo" generalmente tienen las siguientes componentes:

* Rondas de _code challenges_, para probar la capacidad de la persona a la hora de resolver problemas.
* Rondas de _system design_, para medir el _seniority_ del postulante.

Si bien este proceso tiene ciertos vicios, en la práctica se da que minimiza los falsos positivos (pudiendo obtener falsos negativos). Desde el punto de vista de la empresa, y por lo que he leído en blogs, imagino que la razón principal de ello es que contratar es caro. En mi experiencia este proceso es divertido, excepto raras veces en que salen con un problema de programación dinámica (tipo _hard_ de leetcode) y esperan que lo resuelvas en 20 minutos. Pero fuera de esos casos raros, el proceso es divertido, pero causa ansiedad. Si soy honesto, resolver problemas de código no es complicado, ya que en general en estas entrevistas preguntan problemas de dificultad media (a veces fácil, y rara vez difíciles). En general se busca evaluar la capacidad analítica del postulante (generalmente es libre de escoger el lenguaje de programación que guste), y también conocimiento básico de algoritmos y estructuras de datos (ordenamiento, grafos, _heaps_, árboles binarios, _arrays_, testing, capacidad de plantear una solución y luego implementarla). Como al final de cada entrada pondré algún ejercicio de este tipo, en esta oportunidad me enfocare en las entrevistas que sí me asustan: _System Design_. En este tipo de entrevistas, el problema en general es ambigüo, no existe una solución clara, y el postulante es el que debe hacer supuesto, aclarar requerimientos, hacer cálculos (latencia, consultas por segundo, memoria, escalabilidad, uso de red, etc.).

En esta entrada de blog, diseñaremos el cliché de los sistemas, un acortador de `URLs` (tipo _tinyurl_).

## Acortador de URLs ##

Cuando te encuentras en una entrevista de diseño de sistemas, el enunciado probablemente sea algo como: _Diseñe un acortador de urls_. Como se puede notar, el enunciado es bastante ambigüo e incompleto, y empezar a diseñar "a la mala" es un mal indicio para el entrevistador. Algunas preguntas que se pueden hacer:

* ¿De qué largo tienen que ser las URLs sin considerar el dominio?
* ¿Es necesario que un usuario se registre para usar el sistema?
* ¿Cuántas URLs se espera que se generen diariamente?
* ¿Cuál es el largo máximo esperado para una URL?

Dependiendo de las aclaraciones que haga el entrevistador, se debe tomar nota de los requerimientos y hacer estimaciones, como cantidad de consultas por segundo, espacio en disco requerido, tolerancia a fallas, entre otras cosas. Supongamos que iterando con el entrevistador obtuvimos las siguientes respuestas:

* Se esperan urls de en promedio `100` caractéres
* Promedio de `10000` URLs al día
* Prototipo inicial vivirá 1 año
* Por el momento no preocuparse de la tolerancia a fallas

Con ello se puede hacer una estimación:

* Largo de URL `100 B`
* Cantidad de consultas diarias `10000`
* No se requiere registro de usuarios

Por lo tanto el espacio requerido diario `100 Bytes * 10000` sería aproximadamente `1 MB` y considerando 1 año `~365MB`. Para soportar este caso de uso, se puede utilizar cualquier motor de base de datos (relacional o no relacional), como de momento la tolerancia a fallas no es un impedimento, se puede dejar, pero valdría la pena mencionar que se pueden utilizar réplicas de la base de datos para tolerancia a falla. Pero para este caso se considerará un servidor, y se puede comentar que luego del diseño inicial se propondrán ideas para escalar el sistema.

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

Por lo tanto, se podría diseñar una API con dos _endpoints_:

* `POST /shortener` que recibe una URL y la transforma en una URL acortada
* `GET /<URL_ID>` que a partir de la URL acortada, retorna la URL original

Para transformar la url, se puede utilizar una función _hash_, como se muestra en la figura 3. Esta función retornará un valor que se utilizará para mapear hacia la URL original.

<div align="center">

![Hash](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/ab212faf039dac2e65559066f2019b08259fbf79/hash.png)

_Fig 3: Función hash._

</div>

### Modelo de datos ###

El modelo de datos puede ser el siguiente:

| Col           | Type     |
|---------------|----------|
| `url_id`      | `int64`  |
| `url`         | `string` |
| `shorten_url` | `string` |

Si queremos que los valores de la función hash sean valores alfanuméricos (`[0-9A-Za-z]`), entonces tenemos 62 símbolos para escoger. El largo del `string` resultante del hash, dará la cantidad de URL diferentes que se pueden tener. Por ejemplo, si consideramos _hashes_ de largo 3, entonces: 

`# Posibles URLs = 62^3 = 238328`.

El largo debe considerarse, si a futuro se quiere escalar el sistema a millones o billones de URLs.

Se pueden explorar dos posibles alternativas:

1. Utilizar una función hash existente (`CRC32`, `MD5`, `SHA-1`)
2. Generar un `Id` único para cada `URL`

En la primera alternativa, para reducir el largo del hash (idea de acortar URLs), habría que truncar el valor resultante al aplicarlo a la `URL` original. Por otro lado, existe el riesgo de tener colisiones (`f(x1) = f(x2)` para algún `x1 != x2`), por lo que habría que implementar un método que permita lidiar con colisiones. Hacer que esto sea eficiente es un desafío complejo, pero es una alternativa válida. Por otro lado, podría utilizarse un `Id` único y transformarlo a base 62. Esto tiene la desventaja que los hashes no serán de largo fijo, y por otro lado dependen de cómo se genere el `Id` (es un poco complejo para el caso distribuido). Sin embargo, para el prototipo, podemos usar de `Id` el `timestamp` en el que la `url` es creada y realizar la transformación. El siguiente código en `golang` transforma un entero de base 10 a base 62:

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

Por ejemplo si la `url` fue creada con un timestamp `1657000477`, entonces el _hash_ sería `1O8Bw9`.

### Implementación ###

Implementé este sistema con el siguiente stack:

* `MySQL`
* `Golang`
* `Redis`

<div align="center">

![Create Hash](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/cafe2106f31be18c521789d7d7adc764fa2e73d1/postman.png)

_Fig 4: Creando hash para url._

![Demo](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/cafe2106f31be18c521789d7d7adc764fa2e73d1/demo-url.gif)

_Fig 5: Demo acortador de url redireccionando._

</div>

Dejo un [enlace al projecto en github](https://github.com/dpalmasan/url-shortener).

## Code Challenge ##

Escriba una función con la siguiente firma `bool oneEditApart(string s1, string s2)`. La función debe retornar `true` si con una sola *edición* el string `s1/s2` puede ser transformado a `s2/s1`. Una edición consiste en:

* Insertar un caracter en cualquier posición del `string`
* Eliminar un caracter
* Reemplazar un caracter

Ejemplos:

```
OneEditApart("cat", "dog") = false
OneEditApart("cat", "cats") = true
OneEditApart("cat", "cut") = true
OneEditApart("cat", "cast") = true
OneEditApart("cat", "at") = true
OneEditApart("cat", "act") = false
```

<details><summary>Ver Solución</summary>
<p>

La primera observación que hay que tener en este problema, es que si el largo de los strings difiere en 2 o más, significa que se necesita más de una edición para convertir `s1` en `s2`, por lo tanto para este caso siempre deberíamos retornar `false`. Por ejemplo si consideramos los strings `"c"` y `"cat"`, necesitamos o insertar dos caracteres en `"c"` o eliminar dos caracteres en `"cat"`. Esta regla nos da una idea de cómo podemos abordar el problema, debido a la simetría de las operaciones.

Primero debemos chequear si `|s1 - s2| <= 1`, en caso contrario siempre retornamos `false`. Luego, podemos definir `s1'` y `s2'`, tal que `s1'.length >= s2'.length`. En esta definición tenemos un caso de borde extra, por ejemplo si `s1'` o `s2'` son de largo 1, significa que siempre van a estar a una distancia de edición o menos. Por lo tanto retornamos `true`.

Finalmente, debemos considerar los siguientes 3 casos para `s1'` y `s2'`:

* `"cat"` y `"at"`
* `"cat"` y `"ca"`
* `"cat"` y `"cut"`

Para el primer caso, chequeamos el primer caracter y lo "ignoramos" si es diferente. Luego, iteramos sobre `s2'` y contamos las diferencias. Si hay más de dos diferencias, significa que los strings no están a una distancia de edición igual 1 y por lo tanto retornamos `false`, en caso contrario, retornamos `true`. Se llega al siguiente algoritmo:

```
algoritmo one_edit_apart:
  entrada: s1, s2
  salida: true si s1 y s2 están a una distancia de edición igual a 1, false en caso contrario

  len_diff = |s1.length - s2.length|

    if len_diff > 1:
        return false

    if s1.length > s2.length:
        s1' = s1
        s2' = s2
    else:
        s1' = s2
        s2' = s1

    if s1'.length == 1:
        return True

    diffs = 0

    # Caso de borde
    s1_idx = 1 if s1p[0] != s2p[0] else 0
    for s2_idx = 0 to s2'.length - 1
        if s1_idx < s1'.length and s2'[s2_idx] != s1'[s1_idx]:
            if diffs < 1 and len_diff > 0:
                s1_idx += 1
            diffs += 1
        s1_idx += 1
    return diffs <= 1
```

### Implementación tentativa en python

{% highlight python %}

def one_edit_apart(s1: str, s2: str) -> bool:
    """Check if two strings are one edit apart.
    The possible operations to consider are:
    - Insert
    - Replace
    - Remove
    If using one of these operations we can convert string ``s1`` in string
    ``s2`` then the function returns True, otherwise it will return False.

    :param s1: Input string 1
    :type s1: str
    :param s2: Input string 2
    :type s2: str
    :return: True if strings are one edit apart, False otherwise
    :rtype: bool
    """
    len_diff = abs(len(s1) - len(s2))

    # To be one edit apart this is required
    if len_diff > 1:
        return False

    # Making s1' as the largest string
    if len(s1) > len(s2):
        s1p = s1
        s2p = s2
    else:
        s1p = s2
        s2p = s1

    # Base case, if s1' is one char then we now distance is 1
    if len(s1p) == 1:
        return True

    diffs = 0

    # For corner cases like "cat" "at", we check the first character.
    s1_idx = 1 if s1p[0] != s2p[0] else 0
    for s2_idx, c in enumerate(s2p):
        if s1_idx < len(s1p) and c != s1p[s1_idx]:
            # This is for corner cases like "cat" "ca"
            if diffs < 1 and len_diff > 0:
                s1_idx += 1
            diffs += 1
        s1_idx += 1

    # If we found less or 1 difference, then strings are one edit apart
    return diffs <= 1

{% endhighlight %}

</p>
</details>