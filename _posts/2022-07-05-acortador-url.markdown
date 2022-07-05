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