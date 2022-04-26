---
layout: post
title:  "PrimerPost ¡Salud 🍺!"
date:   2022-04-25 19:22:03 -0400
categories: python algorithms
---
En este primer post me gustaría introducir un poco de qué tratará el Blog. Como dice la descripción se tratará de un blog en el cual compartiré conocimiento técnico en diferentes áreas de software en las que me ha tocado trabajar. He tenido la fortuna de trabajar en varios roles en la industria, todos ellos relacionados a la tecnología:

* `QA Engineer`
* `Data Scientist`
* `Data Engineer`
* `Backend Engineer`
* `Full Stack Engineer`

También he trabajado en proyectos de investigación y desarrollo con la Universidad de Concepción, entre los cuales destaco [TRUNAJOD](https://github.com/dpalmasan/TRUNAJOD2.0) que es una biblioteca para análisis de complejidad textual y ha sido utilizada en diversas investigaciones relacionadas con comprensión lectora y relación lectura-escritura en el aprendizaje.

Para no perder el propósito del foro, en este primer post dejo un pequeño problema a resolver, digamos en `python` (aunque el lenguaje debiese ser irrelevante). El problema es el que sigue: _Invierta la representación de bits de un número `n` dado._. Ejemplos:

Si la entrada es `n = 43261596 (00000010100101000001111010011100)`, la salida debiese ser `964176192 (00111001011110000010100101000000)`. Un caso un poco más simple de seguir, si la entrada es (en binario) `11111111111111111111111111111101`, la salida debiese ser `10111111111111111111111111111111`. Para que quede claro, la llamada `invertir_bits(43261596)` debiese retornar `964176192`.

Una posible solución es el siguiente algoritmo:

1. Inicializar `delta = 31` (32 bits)
2. Tomar el dígito menos significativo (LSB)
3. Desplazar dicho dígito en delta
4. Actualizar delta con `delta <- delta - 1`
5. Dividir `n` en 2 (desplazar un bit a la derecha)
6. Repetir 2 hasta que `n` sea `0`

Una implementación tentativa en `python`:

{% highlight python %}
def invertir_bits(n: int) -> int:
    result = 0
    delta = 31
    while n:
        result |= ((n & 1) << delta)
        delta -= 1
        n >>= 1
    return result
{% endhighlight %}

Con esto termina mi primer post, espero que haya sido de su agrado.