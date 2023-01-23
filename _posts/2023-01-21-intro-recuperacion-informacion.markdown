---
layout: post
title:  "Motores de Búsqueda y Procesamiento de Texto"
date:   2023-01-21 17:15:03 -0400
categories: python algorithms ir
---

Una de las entrevistas de diseño de sistemas (hice un [post en que explico en qué consisten]({{ site.baseurl }}{% link _posts/2022-07-05-acortador-url.markdown %})) que tuve para cierta empresa, me preguntaron sobre implementar un motor de búsqueda que dada una consulta, recuperara documentos de un repositorio, siempre que los documentos contuvieran las palabras de la consulta. Yo que venía utilizando técnicas como Indexado Semántico Latente (_LSI: Latent Semantic Indexing_), o modelos de espacio vectorial me enredé un poco por los nervios, y dije lo más simple: *Índice Invertido*, y de ahí vino la pregunta, ¿cómo implementas un índice invertido? Ahí me quedé helado, porque la verdad no me acordaba, intenté razonarlo, pero al final expliqué sólo la parte de búsqueda Booleana, pero no pude explicar cómo implementarlo eficientemente. Para los morbosos, sí fallé un poco en esa pregunta pero eso era parte de la entrevista, en el resto me fue bien porque pasé :smile:.

Para redimirme un poco, en este post explico lo básico de búsqueda de texto, y en particular el índice invertido, para dar pie a probablemente un siguiente post, donde hablaré de clasificación de textos y otros temas y anécdotas.

## Recuperación de Documentos Relevantes

### Enfoque Ingenuo

Supongamos que tenemos un listado de 5 documentos `D1`, `D2`, `D3`, `D4`, `D5` y un _vocabulario_ como el mostrado en la primera columna de la tabla 1. La mostrada se conoce como matriz de término-documentos, y en este caso, cada celda tiene un valor Booleano de ocurrencia o no ocurrencia del término $t_i$ en el documento $D_j$.

|             | D1 | D2 | D3 | D4 | D5 |
| ----------- |----|----|----|----|----|
| partido     | 1  | 0  | 0  | 0  | 1  |
| perro       | 0  | 1  | 0  | 1  | 1  |
| comida      | 0  | 0  | 1  | 1  | 1  |
| casa        | 1  | 1  | 1  | 0  | 1  |

Supongamos que queremos encontrar todos los documentos que contengan los términos perro y comida, utilizando la representación dada, podríamos considerar los bits que representan estas palabras $perro = 01011$, $comida = 00111$, luego el conjunto de documentos que contienen los dos términos:

$$perro \land comida = 01011 \land 00111 = 00011$$

Es decir, los documentos `D4` y `D5`. Si bien esta representación es intuitiva, y simple, no es práctica, ya que la cantidad de memoria necesaria dependerá de la cantidad de términos únicos a considerar y también de la cantidad de documentos. En especial si consideramos que la matriz resultante para estos casos será dispersa (muchas entradas nulas). Por lo que se necesitan mejores representaciones para lograr escalar a una mayor cantidad de documentos y términos.

### Índice Invertido

El índice invertido, toma las ideas descritas en la sección anterior, pero utiliza una representación en que no se almacenan los valores nulos, es decir se elimina la dispersión de los datos (matriz término-documento). La idea es simple, en lugar de utilizar todos los "bits", representamos la matriz como un lexicon de términos-publicaciones, como se muestra en la figura 1:

<div align="center">

![postings](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/54e9984d25b566dbfe8f54e8bcdce1c2dde17ca9/inverted-index-postings.png)

_Fig 1: Representación lexicon-publicaciones_

</div>

Las documentos, deben estar ordenados por ID en cada lista, esto, para poder implementar una intersección de forma eficiente. Para construir el índice, generalmente se sigue el siguiente proceso:

1. Crear pares `término - documento`
2. Ordenar pares por término y luego por documento
3. Combinar pares para crear la lista de publicaciones para cada término (ver figura 1)

Notar que la secuencia de pasos 2, 3 es algo similar a ejecutar `sort | uniq` en una terminal. Una vez teniendo las listas, se pueden tener dos punteros, y encontrar calces de documentos por término, y finalmente retornar la lista de calces. Esto tendría una complejidad asintótica de $O(x + y)$ donde $x$ e $y$ son los largos de las listas a intersectar. De forma más general, la complejidad asintótica sería $O(N)$ donde $N$ es la cantidad de documentos.

#### Tokenización y Canonicalización

A menos que requeramos hacer búsqueda exacta de términos, por lo general, para reducir la variabilidad del lenguaje se intenta canonicalizar cada término. Un ejemplo de canonicalización es remover las inflexiones de cada lexema, por ejemplo llevando las palabras a una forma base, supongamos que tenemos la siguiente oración:

* Juanito escapó de un perrito que lo quería morder

Si llevamos cada palabra a su forma base y eliminamos las palabras funcionales, la oración canonicalizada seria:

* Juanito escape perro querer morder

Existen distintos tipos de canonicalización, los más comunes en la práctica: _lematización_, _stemming_ o incluso corrección ortográfica. Sin embargo, la canonicalización puede ser cualquier tipo de transoformación que transforme las oraciones/términos a una forma base. Los ejemplos mencionados:

* Lematización: Llevar palabras a su forma raíz: $corriendo \rightarrow correr$. Generalmente se necesita un [_lexicon_](https://github.com/michmech/lemmatization-lists) que tenga los pares `(término, lema)`.
* Stemming: Conjunto de reglas para reducir las palabras a una forma base, no necesariamente una palabra válida, ejemplo $corriendo \rightarrow corr$. Un ejemplo clásico es el [stemmer de Porter](https://tartarus.org/martin/PorterStemmer/).
* Corrección ortográfica: Este paso puede ser pre-procesamiento a los pasos anteriores. Existen varios tipos de corrección ortográfica (libre de contexto o dependiente del contexto). Una forma simple de implementar un corrector ortográfico, es extraer la frecuencia de términos en un lenguaje, y generar un modelo del lenguaje a partir de dicha frecuencia. Finalmente, generar posibles transformaciones de la palabra a un `string` cuya distancia de edición es `k` de la palabra original y retornar la palabra más probable del modelo del lenguaje.

#### Implementación

La ventaja del índice invertido es que es simple de implementar. A continuación se muestra una implementación trivial en memoria utilizando `python` :

```python
class InMemoryInvertedIndex:
    def __init__(self, docs: Iterable[str], tokenizer: Tokenizer):
        self._terms = defaultdict(list)
        self._build_terms(docs, tokenizer)
        self._tokenizer = tokenizer

    def _build_terms(self, docs, tokenizer) -> None:
        for i, doc in enumerate(docs):
            curr_term = None
            for term in sorted(tokenizer.tokenize(doc)):
                if term != curr_term:
                    self._terms[term].append(i)
                    curr_term = term

    def search(self, query: str) -> Iterable[int]:
        terms = self._tokenizer.tokenize(query)
        return self._intersect(terms)

    def _intersect(self, terms) -> Iterable[int]:
        # Sort by frequency
        terms = sorted(terms, key=lambda x: len(self._terms[x]))
        curr_term = 0
        result = self._terms[terms[curr_term]]
        curr_term += 1
        while curr_term < len(terms) and result:
            result = self._intersect_two(result, self._terms[terms[curr_term]])
            curr_term += 1

        return result

    def _intersect_two(self, p1, p2):
        i1 = 0
        i2 = 0
        answer = []
        while i1 < len(p1) and i2 < len(p2):
            if p1[i1] == p2[i2]:
                answer.append(p1[i1])
                i1 += 1
                i2 += 2
            else:
                if p1[i1] < p2[i2]:
                    i1 += 1
                else:
                    i2 += 1
        return answer
```

Donde `Tokenizer` es una interfaz que implementa tokenización, por ejemplo, una implementación simple utilizando lematización:

```python
class Tokenizer:
    @abstractmethod
    def tokenize(self, text: str) -> Iterable[str]:
        pass


class LemmaTokenizer(Tokenizer):
    def __init__(self, lemmas: Dict[str, str], threshold: int = 3):
        self._lemmas = lemmas
        self._re = re.compile("[^0-9a-zA-Z]+")
        self._threshold = threshold

    def tokenize(self, text: str) -> Iterable[str]:
        return [
            self._lemmas.get(word.lower(), word.lower())
            for word in self._re.sub(" ", text).split()
            if len(word) >= self._threshold
        ]

    @classmethod
    def load_from_file(cls, filepath: Path):
        lexicon = {}
        with open(filepath, "r") as fp:
            for line in fp:
                lemma, word = line.split()
                lexicon[word.strip()] = lemma.strip()

        return cls(lexicon)
```

#### Aplicación: Motor de Búsqueda de Noticias

Primero que todo, el código se encuentra en un [repositorio en github](https://github.com/dpalmasan/python-nlp-examples). Manualmente, extraje textos de noticias de [Bío-Bío Chile](https://www.biobiochile.cl/), luego cree un índice invertido utilizando el conjunto de documentos, y como canonicalización, utilicé lematización utilizando el lexicon disponible en [https://github.com/michmech/lemmatization-lists](https://github.com/michmech/lemmatization-lists). El código se ve como sigue:

```python
from pathlib import Path
from ir import InMemoryInvertedIndex, LemmaTokenizer

RESOURCE_PATH = Path("./resources")


if __name__ == "__main__":
    tokenizer = LemmaTokenizer.load_from_file(RESOURCE_PATH / "lemmatization-es.txt")
    docs = []
    for path in (RESOURCE_PATH / "documents").glob("*.txt"):
        with open(path, "r") as doc:
            docs.append(doc.read())

    inverted_index = InMemoryInvertedIndex(docs, tokenizer)
    while True:
        query = input("Ingresar búsqueda: ")
        results = inverted_index.search(query)
        print(results)
        query = input("Mostrar texto? (S/s)")
        if query.lower().strip() == "s":
            for i, result in enumerate(results, start=1):
                print("=" * 79)
                print(f"Resultado {i}")
                print("=" * 79)
                print(docs[result])
```

Algunas consultas envíadas al motor de búsqueda:

```
Ingresar búsqueda: universidad chile
[10, 13]
Mostrar texto? (S/s)s
===============================================================================
Resultado 1
===============================================================================
Johnny Herrera, exportero de Universidad de Chile y hoy comentarista de TNT Sports, fue captado golpeando a un hombre en un club nocturno de Vitacura.

El violento hecho ocurrió el pasado jueves 19 de enero en el Bar Candelaria de la comuna del sector oriente, lugar donde el retirado guardametas agredió a una persona por razones que se desconocen.

Los detalles los reveló el periodista Sergio Rojas a través de una transmisión en vivo en Instagram, donde mostró el video de las cámaras de seguridad donde se aprecia al ‘Samurái’ golpeando a un hombre hasta botarlo al piso.

De acuerdo a las imágenes, pese a que la víctima no respondió a los primer golpes, las agresiones del exarquero no se detuvieron.

El citado comunicador detalló que habló con la persona agredida. Según él, no hubo razones para ser atacado por el ídolo de Universidad de Chile, por lo que analiza tomar acciones legales en contra de Johnny Herrera.

Rojas, además, aseguró que desde TNT Sport evalúan sancionar al otrora seleccionado de La Roja por este violento hecho.

Herrera, pese a ser contactado por el reconocido periodista de farándula, optó por no dar declaraciones sobre lo acontecido.
===============================================================================
Resultado 2
===============================================================================
Universidad de Chile debuta en el Campeonato Nacional 2023 ante Huachipato, en lo que será el debut oficial de Mauricio Pellegrino en la banca de los azules.

El duelo está programado para las 19:00 horas de este lunes 23 de enero, en el estadio Santa Laura.

Comienza otro año para La U, uno donde esperan -luego de tres temporadas consecutivas peleando por el descenso- luchar por la parte alta de la tabla.

Para eso, destacan las incorporaciones de jugadores como Matías Zaldivia, Juan Pablo Gómez, Federico Mateos Leandro Fernández, Nicolás Guerra y el portero Cristopher Toselli.

En el papel, es un plantel competitivo con el que Pellegrino espera dejar en el olvido los fracasos azules de los últimos años y con el que quiere empezar ganando.

Al frente estará un Huachipato que también tuvo un mal 2022, al punto de acabar entre los últimos elencos de la tabla.

Gustavo Álvarez es el nuevo entrenador de los ‘acereros’, que sumaron interesantes refuerzos como Mateo Acosta, Carlo Villanueva y Julián Brea.

La U y Huachipato se enfrentan este lunes 23 de enero, a partir de las 19:00 horas, en el estadio Santa Laura.

Ingresar búsqueda: diabetes
[8]
Mostrar texto? (S/s)s
===============================================================================
Resultado 1
===============================================================================
Un estudio finlandés sobre la influencia del estilo de vida del gestante en el neurodesarrollo de los infantes determinó que los hijos de personas con diabetes gestacional tienen peores habilidades lingüísticas que los bebés de madres sin la enfermedad.

Una investigación finlandesa reveló que los hijos de embarazadas y pacientes de diabetes gestacional pueden tener peores habilidades lingüísticas que aquellos que nacen de madres sin la enfermedad.

La Universidad Turku y el Hospital Universitario de Turku de Finlandia, investigaron cómo la salud y el estilo de vida de la madre durante el embarazo condicionan el neurodesarrollo de los niños de 2 años, de donde se extrajeron los resultados.

El proyecto de investigación publicado en la revista Pediatric Research examinó el desarrollo de las habilidades cognitivas, lingüísticas y motoras de los niños.

La adiposidad materna se determinó mediante pletismografía por desplazamiento de aire y la diabetes gestacional con prueba de tolerancia oral a la glucosa. Además, la ingesta dietética durante el embarazo se evaluó con índices de calidad de la dieta y cuestionarios de consumo de pescado.

“En promedio, el neurodesarrollo infantil en nuestros datos estaba en el rango normal. Los resultados de nuestra investigación mostraron que los niños de 2 años cuyas madres habían sido diagnosticadas con diabetes gestacional tenían peores habilidades lingüísticas que los niños cuyas madres no habían sido diagnosticadas con diabetes gestacional”, dijeron los expertos.

Además, el estudio descubrió que un mayor porcentaje de grasa corporal materna estaba asociado con habilidades cognitivas, de lenguaje y motoras más débiles en los niños.

“Nuestra observación es única, ya que estudios previos no han examinado la asociación entre la composición corporal materna y el neurodesarrollo de los niños”, recalcaron los investigadores.

Por otro lado, el estudio también ha mostrado que una mejor calidad dietética de la dieta de la madre se asoció con un mejor desarrollo del lenguaje del niño, al igual que un consumo elevado de pescado por parte de la madre.

“Una dieta saludable e integral durante el embarazo puede ser particularmente beneficiosa para el neurodesarrollo de los niños cuyas madres pertenecen al grupo de riesgo de diabetes gestacional por sobrepeso u obesidad”, han zanjado los expertos.
```

#### Mejoras utilizando Skip Lists

Una implementación un poco más eficiente es utilizar [`SkipLists`](https://en.wikipedia.org/wiki/Skip_list). Esto funciona únicamente si decidimos soportar únicamente consultas en conjunción (es decir, considerando únicamente el operador Booleano $\land$). La idea es reducir el número de comparaciones entre IDs de documentos, aprovechando que los documentos están ordenados por ID, por ejemplo, si tenemos las lista de publicaciones `[1, 3, 5, 8, 9, 11, 13, 20]` y `20`, podríamos saltar desde `1` a `20` sin tener recorrer todos los elementos y así reduciendo la cantidad de comparaciones. Un ejemplo de `SkipList` en `python`:

```python
class SkipList:
    class Node:
        def __init__(self, key: int, level: int):
            self._key = key
            self.forward = [None] * (level + 1)

        @property
        def key(self) -> int:
            return self._key

    def __init__(self, max_level, p):
        self._max_level = max_level
        self._p = p
        self._sentinel = SkipList.Node(-1, self._max_level)

        self.level = 0

    def _random_level(self) -> int:
        level = 0
        while random.random() < self._p and level < self._max_level:
            level += 1
        return level

    def insert(self, key) -> None:
        update = [None] * (self._max_level + 1)
        current = self._sentinel

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if current is None or current.key != key:
            random_level = self._random_level()
            if random_level > self.level:
                for i in range(self.level + 1, random_level + 1):
                    update[i] = self._sentinel
                self.level = random_level

            node = SkipList.Node(key, random_level)
            for i in range(random_level + 1):
                node.forward[i] = update[i].forward[i]
                update[i].forward[i] = node

    def search(self, key, current=None) -> "SkipList.Node":
        """Go to the previous key of the possible next key
        :param key: Value to find
        :type key: int
        :param current: Starting node, defaults to None
        :type current: SkipList.Node, optional
        :return: Node to be left of the target node
        :rtype: Optional[SkipList.Node]
        """
        current = current
        if current is None:
            current = self._sentinel
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]

        return current

    def __contains__(self, key: int) -> bool:
        """Check membership for key
        :param key: Item to be checked
        :type key: int
        :return: True if list contains key False otherwise
        :rtype: bool
        """
        current = self._sentinel
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]

        current = current.forward[0]
        return current and current.key == key

    def __str__(self) -> str:
        head = self._sentinel
        res = ""
        for lvl in range(self.level + 1):
            res += f"Level {lvl}: "
            node = head.forward[lvl]
            while node != None:
                res += f"{node.key} "
                node = node.forward[lvl]
            res += "\n"
        return res
```

Queda como ejercicio para el lector, implementar la intersección utilizando `SkipList`.

### Escalando a Millones de Documentos

Como podrá observar el lector, para una cantidad de documentos que no se puedan cargar en RAM, hay que utilizar otras estrategias. Todo, en general depende del hardware disponible. Una estrategia simple es utilizar [_blocked sort-based indexing_ (BSBI)](https://nlp.stanford.edu/IR-book/html/htmledition/blocked-sort-based-indexing-1.html). En simples términos, lo que se hace es cargar los datos en buffers. Para minimizar las llamadas a disco, se utiliza un buffer del máximo tamaño disponible, se aplica el índice invertido a los pares encontrados, se escriben a disco, y se prosigue hasta haber leído todos los documentos. Finalmente existe un paso de combinación de todos estos archivos intermedios, para obtener el índice final.

Para soportar billones de documentos, ya se requiere un enfoque distribuído, el cual queda como ejercicio para el lector investigar.

## Conclusiones

* Una representación simple de textos, puede ser basada en ocurrencia o no ocurrencia de términos
* En general, las matrices término-documentos son dispersas, en la práctica la mayor cantidad de entradas es nula, debido a la variabilidad del lenguaje.
* Existen distintas estrategias para canonicalizar términos, y dependen del problema a tratar (como siempre, no hay enfoque que sirva para todos los casos)
* Para lograr búsquedas eficientes, se debe lidiar con reducción en el espacio de búsqueda, esto se puede lograr vía índices (por ejemplo utilizando una tabla hash)

## Problema del día

Para ir con la temática, el problema de hoy estará relacionado con [búsqueda de palabras](https://github.com/dpalmasan/code-challenges/issues/16).