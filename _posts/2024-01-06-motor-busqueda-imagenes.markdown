---
layout: post
title:  "Motor de búsqueda de imágenes"
date:   2024-01-06 18:00:00 -0400
categories: python algorithms ir
---

En un post previo describí cómo funciona un motor de búsqueda textual clásico y [mostré la implementación un índice invertido]({{ site.baseurl }}{% link _posts/2023-01-21-intro-recuperacion-informacion.markdown %}) utilizando noticias de Chile. En mi vida laboral, en una ocasión, utilicé dicha técnica + esteroides (por ejemplo `Tf-Idf` y distancia coseno) para un problema en el que tenía que hacer calce eficiente entre un repositorio de textos dada una consulta (esto es una reducción del problema, pues era bastante más complejo 😅).

Lo curioso, es que me tocó resolver un problema similar pero con imágenes. En este caso hice una integración de APIs y utilicé un modelo pre-entrenado para similitud de imágenes. Sin embargo, el tener una caja misteriosa que me resuelve el problema no me deja satisfecho. Estudiando un poco algunos libros que tenía en el estante (y nunca había mirado 😅), me inspiré e implemente mi propio motor de búsqueda de imágenes, y este es el tema de este post.

## El Problema de Recuperación de Imágenes

El problema de recuperación de imágenes, se puede describir como sigue: _Dado un repositorio de imágenes, y una imagen como consulta, recuperar todas las imágenes relevantes a la consulta_. En este post, consideraremos "relevante" como imágenes que son _similares_ a la imagen de la consulta.

<div align="center">

![cifar-ex](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d8d3eaf7886b931048df96ac88af4fd0626a33be/img-ret-11.png)

_Fig 1: Recuperación de imágenes relevantes de un repositorio dada una imagen como consulta_

</div>

Algunas aplicaciones:

* Búsqueda de imágenes
* Clasificación de imágenes
* Calce de contenido (por ejemplo visión artificial)
* Detección de contenidos abusivos/ilegales en plataformas
* Etc.

### Representación de Imágenes

Una imagen puede pensarse como un conjunto de capas. Por lo general, se piensa en el conjunto de colores rojo, verde y azul. La idea es, que cualquier color puede representarse como una combinación de estos tres colores primarios. Por otro lado, una imagen es una grilla de pixeles (matriz), donde cada pixel tiene un color asignado.

### Similitud de Imágenes

Existen varios enfoques para definir una métrica de similitud de imágenes. Sin embargo, cada enfoque depende del caso de uso específico:

1. Histograma de Colores: Por ejemplo, podemos calcular un histograma de cada conjunto de colores (con sus distintas tonalidades), y escoger una cantidad fija de intervalos (_bins_), por ejemplo 32. Si consideramos estos 3 colores, obtendríamos un vector de 96 dimensiones, como se muestra:

$$x_i = (x_{i,rojo}^{32}, x_{i,verde}^{32}, x_{i,azul}^{32})$$

Y podríamos recuperar las imágenes más relevantes mediante una medida de similitud, por ejemplo similitud coseno:

$$sim(x_i, x_j) = cos(x_i, x_j) = \displaystyle \frac{x_i \cdot x_j}{||x_i||||x_j||}$$

2. Vector de pixeles: Asumiendo que las imágenes serán de tamaño fijo, considerar un vector de dimension $N$ donde $N$ es la cantidad de píxeles de cada imagen. Luego cada componente del vector sería un pixel.

3. Vector de componentes. Aplicar PCA (análisis de componentes principales) a la representación mencionada en el punto previo.

4. Representación latente (ej. capas internas en una red neuronal)

En los primeros 3 enfoques, no se tiene la información contextual, ya que se consideran los pixeles como independientes entre sí. La ventaja del último enfoque, es que considera que ciertos pixeles pueden tener injerencia (si ya me quiere criticar, le aviso que [es con JOTA](https://dle.rae.es/injerencia) 😊) sobre pixeles vecinos, lo que lograría hacer comparaciones a nivel de estructura y otras propiedades latentes.

### Implementación de un Motor de Búsqueda de Imágenes

Para este ejercicio, utilizaré el conjunto de datos [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html), que consiste en imágenes de colores y `32x32` pixeles. Todos los experimentos los ejecuté en un notebook en [Colab](https://colab.research.google.com/), y los datos los almacené en mi Google drive.

#### Análisis Exploratorio

Primero cargamos las dependencias a utilizar. Para instalar FAISS, ejecutar: `!pip install faiss-cpu --no-cache`

```python
from google.colab import drive
import pickle
import faiss
from matplotlib import pyplot as plt
import numpy as np
import torch
from pathlib import Path
from collections import Counter
```

Para montar mi gdrive en la sesión de Colab:

```python
drive.mount('/content/drive')
```

Para cargar el conjunto de datos, sigo las instrucciones en la página de CIFAR:

```python
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

IMAGE_FILE_PATH = Path("/content/drive/$DIRECTORIO/cifar-10-batches-py/data_batch_1")
LABELS_FILE_PATH = Path("/content/drive/$DIRECTORIO/cifar-10-batches-py/batches.meta")
label_data = unpickle(LABELS_FILE_PATH)
label_maping = label_data[b"label_names"]
data = unpickle(IMAGE_FILE_PATH)
data.keys()
```

Observamos que los datos están almacenados en un diccionario de `python` que tiene las siguientes propiedades:

```python
dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
```

Miremos la distribución de las etiquetas:

```python
dataset = data[b"data"]
labels = [label_maping[i] for i in data[b"labels"]]
Counter(labels)
```

En este podemos ver que el conjunto de datos está casi uniformemente distribuido.

```python
Counter({b'frog': 1030,
         b'truck': 981,
         b'deer': 999,
         b'automobile': 974,
         b'bird': 1032,
         b'horse': 1001,
         b'ship': 1025,
         b'cat': 1016,
         b'dog': 937,
         b'airplane': 1005})
```

Procesamos los datos para tener las imagenes en 3 canales y además las etiquetas de cada imagen:

```python
new_data = []
for img, label in zip(dataset, labels):
    new_data.append((img.reshape(3, 32, 32), label))
```

Echemos una mirada a alguna imagen arbitrariamente seleccionada:

```python
image_example = new_data[3][0].transpose([1, 2, 0])
plt.figure(figsize=(32/50, 32/50))
plt.imshow(image_example)
plt.axis('off')
```

<div align="center">

![cifar-ex](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/84520fc82c9db233d6b4d132f01028bbeee065ff/img-ret-1.png)

_Fig 2: Imagen del conjunto de datos CIFAR_

</div>

#### Entrenando un modelo Codificador-Decodificador

Un modelo codificador-decodificador en el caso de las imágenes sigue la siguiente intuición: _Una imagen está compuesta de características latentes (ej. formas, coloración, etc.). Con estas características, puedo reconstruir la imagen original_. En este caso, el codificador tiene la tarea de representar la imagen utilizando estas características latentes, y el decodificador tiene la tarea de, dada dicha representación, reconstruir la imagen. Esta arquitectura se muestra en la figura 3.

<div align="center">

![cifar-ex](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d8d3eaf7886b931048df96ac88af4fd0626a33be/img-ret-10.png)

_Fig 3: Arquitectura modelo codificador-decodificador_

</div>

En esencia, el modelo debe ser optimizado, de tal forma, que la salida sea idealmente la misma imagen de entrada.

Primero, dividimos el conjunto de datos en un conjunto de entrenamiento y uno de prueba:

```python
train_size = int(0.8 * len(new_data))
test_size = len(new_data) - train_size
train, test = torch.utils.data.random_split(new_data, [train_size, test_size])
```

Verificamos que tenemos los resultados esperados:

```python
print(f"Training set size: {len(train)}")
print(f"Test set size: {len(test)}")
```

Definimos el autoencoder y los parámetros como función de costo, cantidad de epochs.

**Spoiler**: Utilizo capas convolucionales para considerar el contexto en la imagen de entrada (es decir, pixeles colindantes). Intenté hacerlo sólo con capas lineales, pero no logré una buena representación; me queda de tarea pendiente explorar múltiples arquitecturas para ver qué tan buenos resultados se pueden obtener.

```python
class SimpleAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, (3, 3), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, (2, 2), stride=(2, 2)),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = SimpleAutoencoder()
criterion = torch.nn.MSELoss()
num_epochs = 30
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Para cargar los datos en los modelos, `Torch` necesita que se defina un `DataLoader`:

```python
train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers=2)
```

Ahora comienza el entrenamiento del modelo:

```python
# Para almacenar los valores de función de pérdida
train_loss = []
batch_size = len(train_loader)

for epoch in range(num_epochs):
    avg_loss = 0
    for img, label in train_loader:
        # Convertir tipo a float
        img = img.float()
        out = model(img)

        # Comparar la salida del decodificador con la imagen
        # original
        loss = criterion(out, img)

        # Actualizar pesos
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Actualizar promedio de pérdida
        avg_loss += loss.item()

    avg_loss /= batch_size
    train_loss.append(avg_loss)
    print(f"Epoch {epoch + 1}|{num_epochs}; Running loss {avg_loss}")
```

Idealmente, la función de pérdida tiene que ir reduciéndose a medida que avanza el entrenamiento. Si no es así, el modelo no está aprendiendo, por lo que habría que revisar la arquitectura del modelo:

```
Epoch 1|30; Running loss 925.2084133262634
Epoch 2|30; Running loss 225.9821763420105
Epoch 3|30; Running loss 194.27336073112488
Epoch 4|30; Running loss 178.33449586868286
Epoch 5|30; Running loss 167.82453124427795
Epoch 6|30; Running loss 160.89200536727904
Epoch 7|30; Running loss 155.2542067222595
Epoch 8|30; Running loss 150.87735285377502
Epoch 9|30; Running loss 147.33312226867676
Epoch 10|30; Running loss 144.14536425971986
Epoch 11|30; Running loss 142.13272936439515
Epoch 12|30; Running loss 139.8893744430542
Epoch 13|30; Running loss 137.89374314308168
Epoch 14|30; Running loss 136.1791773853302
Epoch 15|30; Running loss 134.33099251937867
Epoch 16|30; Running loss 133.34205507850646
Epoch 17|30; Running loss 131.32471635627746
Epoch 18|30; Running loss 130.05829406356813
Epoch 19|30; Running loss 128.98449465942383
Epoch 20|30; Running loss 127.10736661911011
Epoch 21|30; Running loss 126.33275876045226
Epoch 22|30; Running loss 125.31079289817811
Epoch 23|30; Running loss 124.149567653656
Epoch 24|30; Running loss 123.63063527297973
Epoch 25|30; Running loss 122.39257022094726
Epoch 26|30; Running loss 121.8142949180603
Epoch 27|30; Running loss 120.99319108963013
Epoch 28|30; Running loss 120.2384320640564
Epoch 29|30; Running loss 119.61346651077271
Epoch 30|30; Running loss 118.89974891853332
```

Graficamos la curva de aprendizaje:

```python
plt.plot(range(1, len(train_loss) + 1),train_loss)
plt.xlabel("Epochs")
plt.ylabel("Pérdida en entrenamiento")
plt.title("Curva de aprendizaje del modelo")
plt.show()
```

<div align="center">

![model](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/84520fc82c9db233d6b4d132f01028bbeee065ff/img-ret-2.png)

_Fig 4: Curva de aprendizaje del modelo encoder-decoder_

</div>

Ahora a ver qué ocurre con datos de prueba:

```python
test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=2)
```

Tomamos un sólo ejemplo al azar:

```python
with torch.no_grad():
    for img, label in test_loader:
        img_proc = img.float()
        out = model(img_proc)
        break
```

Mostramos la imagen original:

```python
image_example = np.array(img[0]).transpose([1, 2, 0])
plt.figure(figsize=(32/50, 32/50))
plt.imshow(image_example)
plt.axis('off')
```

<div align="center">

![model](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/84520fc82c9db233d6b4d132f01028bbeee065ff/img-ret-3.png)

_Fig 5: Imagen original del conjunto de prueba_

</div>

Mostramos la reconstrucción (en un mundo ideal, debería ser igual a la imagen original):

```python
image_example = np.array(out_img).transpose([1, 2, 0])
plt.figure(figsize=(32/50, 32/50))
plt.imshow(image_example)
plt.axis('off')
```

<div align="center">

![model](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/84520fc82c9db233d6b4d132f01028bbeee065ff/img-ret-4.png)

_Fig 6: Imagen reconstruída del conjunto de entrenamiento_

</div>

#### Extrayendo Representación Latente (_Embeddings_)

Extraemos representación latente luego de aplicar el codificador (_embeddings_):

```python
embeddings = torch.Tensor()
images = []
with torch.no_grad():
    for img, label in train_loader:
        img_proc = img.float()
        for im, lbl in zip(img, label):
            images.append((im, lbl))
        enc_output = model.encoder(img_proc).cpu()
        embeddings = torch.cat((embeddings, enc_output), 0)
```

Aplanamos los embeddings para tener vectores:

```python
embeddings_flattened = embeddings.flatten(start_dim=1)
```

En este punto ya tenemos todas las imágenes del conjunto de entrenamiento en una representación vectorial, basada en la salida del codificador en la red neuronal.

#### Creando un Índice para Recuperar Imágenes Relevantes

Ya tenemos una representación de la imágen que el computador puede "procesar" y realizar operaciones matemáticas. Sin embargo, queda la pregunta, ¿Cómo recuperamos las imágenes más relevantes de la base de datos dado un vector de consulta $x_q$?

<div align="center">

![model](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d8d3eaf7886b931048df96ac88af4fd0626a33be/img-ret-7.png)

_Fig 7: Problema de búsqueda de imágenes relevantes dada una consulta._

</div>

Un algoritmo ingenuo sería:

1. Inicializar `resultado` como un conjunto vacío `[]`
2. Para cada imagen del repositorio:
    - Calcular similitud con la consulta
    - Guardar esta similitud y una referencia a la imagen en `resultado`
3. Ordenar `resultado` de acuerdo a la similitud
4. Retornar `resultado`

Sin embargo, este algoritmo es muy ineficiente. Asumiendo que la dimensionalidad $D$ de los vectores es fija, tendríamos que comparar la consulta con todos los vectores de la base de datos $O(n)$. Luego ordenar $O(n \log{n})$. Este enfoque no escala.

Un enfoque más "inteligente" podría ser, agrupar las imágenes más similares (por ejemplo vía _K-Means_), y luego, determinar el centroide más cercano a la consulta, y obtener los resultados más relevantes sólo del subconjunto de imágenes que pertenecen a dicho grupo:

<div align="center">

![model](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d8d3eaf7886b931048df96ac88af4fd0626a33be/img-ret-8.png)

_Fig 8: Agrupando imágenes similares._

</div>

Finalmente, podemos también, considerando los centroides, hacer algo un poco más "inteligente" y particionar el espacio utilizando diagramas de Voronoi:

<div align="center">

![model](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d8d3eaf7886b931048df96ac88af4fd0626a33be/img-ret-9.png)

_Fig 8: Diagrama de Voronoi del espacio vectorial._

</div>

Las bases de datos basadas en vectores, utilizan principios similares para hacer la búsqueda de datos más eficiente. Por ejemplo un motor de búsqueda podría utilizar una representación semántica (_word embeddings_, _sentence embeddings_) para búsqueda textual. En este caso, se búsca el centroide más cercano y la búsqueda se limita a los vectores que se encuentren dentro de dicha partición.

Para implementar la partición utilizando Voronoi, creamos un índice utilizando la biblioteca `FAISS` (_Facebook Artificial Intelligence Similarity Search_):

```python
nlist = 50
d = embeddings_flattened.shape[1]
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
```

Entrenamos el índice:

```python
index.train(embeddings_flattened)
```

Agregamos imágenes al índice:

```python
index.add(embeddings_flattened)
```

Obtenemos los embeddings de los datos de prueba:

```python
with torch.no_grad():
    for test_img, test_label in test_loader:
        img = test_img.float()
        enc_output = model.encoder(img).cpu()
        break
```

Escogemos un dato de prueba arbitrariamente y lo consideramos como la consulta a buscar:

```python
query = enc_output.flatten(start_dim=1)
```

Visualizamos la consulta:

```python
image_example = np.array(test_img[0]).transpose([1, 2, 0])
plt.figure(figsize=(32/50, 32/50))
plt.imshow(image_example)
plt.title(test_label[0].decode())
plt.axis('off')
```

<div align="center">

![model](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/84520fc82c9db233d6b4d132f01028bbeee065ff/img-ret-5.png)

_Fig 9: Ejemplo de consulta al motor de búsqueda de imágenes_

</div>

Hacemos la búsqueda en el índice definido previamente:

```python
dists, result_indexes = index.search(query, 5)
```

Mostramos los resultados:

```python
rows = 2
columns = 3
padding = 100
fig = plt.figure(figsize=(10, 7))
for fig_num, idx in enumerate(result_indexes[0], 1):
    img = np.array(images[idx][0], dtype=np.uint8).transpose([1, 2, 0])
    fig.add_subplot(rows, columns, fig_num)
    plt.imshow(img)
    plt.title(images[idx][1].decode())
    plt.axis('off')
```

<div align="center">

![model](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/84520fc82c9db233d6b4d132f01028bbeee065ff/img-ret-6.png)

_Fig 10: Ejemplo de resultados a la consulta_

</div>

En este caso, para la consulta, los resultados tienen sentido. Por ejemplo, la mayoría de los resultados son imágenes de aviones, lo cual es esperado dada la consulta. Sin embargo, también el motor recuperó imágenes de aves, porque en la representación latente, estas imágenes son similares a la consulta.

Sin embargo, no hay garantías de que todas las consultas entregarán los resultados esperados. El conjunto de datos tiene sus particularidades, además de haber utilizado imágenes de baja resolución.

## Conclusiones

* Las imágenes pueden representarse utilizando diferentes enfoques. El mejor enfoque dependerá del caso de uso.
* Las arquitectura codificador-decodificador puede hacer una representación de las características latentes de una imagen. La dimensionalidad dependerá de la arquitectura de la red (cuidado que puede chupar mucha RAM 😅).
* Utilizando representaciones latentes, se pueden crear aplicaciones computacionales interesantes: Clasificador de imágenes, motor de búsqueda de imágenes, detección de objetos en imágenes, etc.
* Para encontrar los vectores más "cercanos" a un vector consulta, existen varios enfoques y algoritmos. El mejor y más eficiente dependerá del caso, del volumen de datos y de la precisión deseada.

## Ejercicios Interesantes

* Hacer un motor de búsqueda utilizando
    - Representación basada en histogramas de colores
    - Representación utilizando pixeles
    - Aplicando PCA/ICA/inserte su algoritmo de componentes principales
    - Utilizando otro tipo de capas (no convolucionales)
* Utilizar la representación interna para implementar un clasificador de imágenes
