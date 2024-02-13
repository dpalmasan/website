---
layout: post
title:  "Reflexiones y Jugando con Pixeles"
date:   2024-02-12 18:00:00 -0400
categories: python algorithms ai
---

<div align="center">

![art](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/2e37ee86e8987e3392682f65c3a9670061737b13/art.jpg)

</div>

# Introducción

En mi post más reciente, me puse a jugar un poco con redes neuronales convolucionales y en particular con arquitecturas Codificador-Decodificador (Autocodificadores) para crear un simple motor de búsqueda de imágenes. Debo decir que, existen muchas razones por las cuales empecé estos proyectos personales, algunas por ejemplo:

* Familiarizarme más con `Pytorch` ya que ahora trabajaré en lado de infraestructura de este _framework_, por lo que necesito conocer sus usos y familiarizarme con la `API`.
* Curiosidad, ya que en la práctica utilicé un modelo de calce de imágenes, y quise entender los fundamentos de fondo para este tipo de modelos.
* Aprender más sobre el estado del arte y los fundamentos que mueven la inteligencia artificial actualmente.

Este post, tendrá una mezcla de dos sabores:

1. Consejos para lidiar con el _Síndrome del Impostor_
2. Reflexiones mías sobre el panorama Tech en general
3. Exploraré un poco sobre Inteligencia Artificial Generativa (a.k.a _GenAI_)

# Reflexiones

En esta sección escribiré algunas reflexiones sobre el escenario actual en el mundo tech. Voy a informar y aclarar, que doy mis opiniones sesgadas a mi experiencia. Generalmente mis opiniones desafían el estatus quo, y a las personas en general les cae como _una patada en la guata_. Me disculpo de antemano si es así, pero como digo, es mi opinión. Si es polémica o no, dependerá del lector; si al lector le molesta, es libre de tener su propia opinión 😊. Advierto, de todas formas, que por lo general no pierdo tiempo en debatir en internet (como lo hacen algunas personas en los comentarios 😆), por lo tanto, me disculpo de antemano si es que no sigo el juego de crear un hilo gigante de comentarios.

## La Cura para el _Síndrome del Impostor_

En esta sección describo una posible solución/cura para el conocido síndrome del impostor. El lector debe notar, que estoy haciendo esto gratis (sin costosas sesiones de coaching 😊).

<details><summary>Click para ver la cura al Síndrome del Impostor</summary>
<p>

<div align="center">

![imposter](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/2220830024f03619f8c96752e9a47673e6d9bce1/imposter-syndrome.jpeg)

</div>

_¿Eres lo suficientemente bueno para realmente tener el Síndrome del Impostor?_

No te preocupes del Síndrome del Impostor, y como dice el dicho: _No hay que preocuparse, hay que ocuparse_. Si sientes que tienes carencias/falencias, sé responsable contigo mismo/a y hazte cargo de ellas 💪. La disciplina es lo que lleva al éxito (si no, miren la gente del mundo fitness, o los atletas de alto rendimiento). También está la variable talento, y hay personas que tienen más talento que otras (dejando constante el esfuerzo, el talentoso genera mayor utilidad). Pero el talento es casi una variable aleatoria y no es posible de controlar, por lo que no vale la pena ni pensar en ello.

</p>
</details>

## Aspectos Humanos en el día a día Tech

El estatus quo en la industria Tech en general en términos de la metodología de trabajo, es seguir alguna metodología ágil, del sabor que sea; y luego implementar de alguna forma las ceremonias y los aspectos que se deben seguir en dicha metodología/marco de trabajo.

Un aspecto común son las famosas reuniones diarias (_dailies_), que en resumen consisten en lo siguiente:

* Cada miembro del equipo dice su status
    * En qué trabajé ayer
    * En que estoy trabajando ahora
    * Hay algo que esté bloqueando el desarrollo
* En algunos casos se revisan "tickets" y esto hace que se pierda más tiempo

Mi opinión y lo que llevo evangelizando hace un tiempo es:

> Si quieres matar la productividad de un equipo de Ingeniería, simplemente crea reuniones diarias (daily)
>
> -- <cite>Quien les escribe</cite>

Siempre me he cuestionado dichas reuniones, algunas preguntas que me hago:

* ¿Qué me importa lo que esté haciendo el resto?
* ¿Hay un gradiente significativo entre lo que hice ayer y lo que haré hoy?
* ¿Si estoy bloqueado, es realmente una _daily_ el momento de mencionar dicho bloqueo?
* ¿De verdad vamos a sacrificar `(15 mins * 5 días * X ing) = 1.25 hr * $costo ing/hr` provocando cambios de contexto, corte de inspiración y costos en ingeniería todas las semanas?

Yo creo que no tiene sentido. Seguir recetas, nunca va a ser el camino para llegar al objetivo (si fuera así de simple, todos seriamos exitosos). Creo que los cambios de contexto frecuentes son perjudiciales para los equipos de ingeniería.

Lamento mucho tocar sensibilidades de gente que practique y utilice marcos de trabajo (_frameworks_) relacionados. Sin embargo, lo mío es una opinión y yo opino lo que quiero 😂. Por otro lado, no quiero que se malentienda lo que digo, no digo que estos _frameworks_ sean inútiles, pero eso de intentar inyectarlo en todos lados es contraproducente. Quizás para una empresa `X` en una situación `Y`, la solución sea utilizar el _framework_ `Z`; pero un caso no hace la norma. Por ejemplo, trabajar para una _startup_ es totalmente diferente que trabajar para una empresa con más años de circo, que ya tiene una estructura burocrática (que en algunos casos hay que cambiar/desafiar).

Lo otro que he observado, los famosos memes de:

* _Una reunión que podría haber sido un correo_
* _No me llamen por teléfono, prefiero que me escriban_
* etc.

Siento que todos somos suficientemente adultos para tomar decisiones en el mundo laboral. Por ejemplo, esto es algo personal, si considero que una reunión no va a ser relevante para mí, o no soy estrictamente necesario, simplemente rechazo la reunión y le comunico a los organizadores que no asistiré, dando razones que justifiquen esta ausencia. Muchas veces me pasó este último año, y me propuse a no participar en ninguna reunión que no considerara que iba a ser aporte, precisamente porque necesito optimizar mi tiempo y pensar formas para generar impacto. Es cierto que existen otros roles que sí deben coordinarse y deben tener muchas reuniones, esto no lo niego. Sin embargo, en el caso de los ingenieros, estos debiesen asistir solo a las reuniones que puedan generar utilidades en el corto y largo plazo; por algo existe la famosa _comunicación asíncrona_.

# Jugando con Pixeles y Modelos Generativos

En esta sección simplente describiré algunos experimentos que hice para generar pixel-art (en particular _sprites_) de un estilo definido, a partir de un conjunto de datos de pixel art. Este mini-proyecto apareció por dos motivos:

1. Un amigo me hizo volver al vicio de los video-juegos y como decimos en Conce (mi ciudad natal) _me camellé_ 😅
2. Tenía ganas de aprender un poco sobre lo reciente en redes neuronales

## Inteligencia Artificial Generativa aplicada a imágenes

En un artículo anterior, hablé sobre cómo implementar un [motor de búsqueda de imágenes]({{ site.baseurl }}{% link _posts/2024-01-06-motor-busqueda-imagenes.markdown %}). En esta sección, hablaré sobre un modelo de inteligencia artificial generativa (GenAI) que es una versión probabilística del modelo que explique previamente, y también intentaré resolver el problema de generar pixel-art a partir de un estilo definido en un conjunto de datos dado.

### Autocodificador Variacional (VAE: _Variational Auto-Encoder_)

Asumimos que la variable $x$ que representa nuestros datos se genera a partir de una variable latente $z$ (representación codificada), la cual no es observable. Por lo tanto, el proceso generativo que cada dato sigue puede describirse como:

1. Se hace un muestreo de la representación latente $z$ desde la distribución apriori $p(z)$
2. Los datos originales, son muestrados de la distribución de probabilidad condicional $p(x|z)$

Con esta noción de modelo probabilistico, podemos definir una versión probabilística de los codificadores y decodificadores. El decodificador "probabilístico" está dado por $p(x|z)$ (obtenemos la reconstrucción de nuestros datos, dada su versión codificada), mientras que el "codificador probabilístico" está definido por $p(z|x)$, la cual describe la distribución de la variable codificada dada su versión decodificada.

Utilizando el _Teorema de Bayes_, podemos encontrar una relación entre estas distribuciones:

$$p(z|x) = \displaystyle \frac{p(x|z)p(z)}{p(x)} = \frac{p(x|z)p(z)}{\int p(x|u)p(u)du}$$

Ahora asumamos que $p(z)$ es una distribución Gaussiana y que $p(x|z)$ es también una distribución Gaussiana cuya media está definida por una función $f$ de la variable $z$ y cuya matriz de covarianza tiene la forma $cI$ donde $I$ es la matriz identidad y $c$ es una constante. Esta función $f$ pertenece a una familia de funciones $F$ que se deja sin especificar de momento y se escogerá más adelante. Hasta ahora, tenemos:

$$
\begin{array}{lll}
 p(z) \equiv \cal{N(0, I)} &  &   \\\\
 p(x|z) \equiv \cal{N(f(z), cI)} & \quad f \in F & \quad c > 0
\end{array}
$$

Consideremos que $f$ es fija y bien definida. Como mencionamos anteriormente, conocemos $p(z)$ y $p(x|z)$, por lo que podríamos utilizar el teorema de Bayes para calcular $p(z|x)$. Este es un problema de inferencia Bayesiana, que usualmente es intratable (integral en el denominador), y se requiere utilizar técnicas de aproximación.

En este caso, el problema puede ser visto como un problema de inferencia variacional, en el cual queremos aproximar $p(z|x)$ a una distribución Gaussiana $q_x(z)$, tal que su media y covarianza están definidas por dos funciones, $g$ y $h$, cuyo parámetro es $x$:

$$
\begin{array}{lll}
 q_x(z) \equiv \cal{N(g(x), h(x))} & \quad g \in G & \quad h \in H
\end{array}
$$

En simples términos, queremos minimizar la _distancia_ entre estas dos distribuciones. Para ello podemos utilizar la _divergencia de Kullback-Leibler_ entre la aproximación y la distribución $p(z|x)$ objetivo:

$$
\begin{align}
 (g^*, h^*) & = \underset{(g, h) \in G\times H}{\mathrm{argmin}} D_{KL}(q_x(z), p(z|x)) \\\\\\\\
 & = \underset{(g, h) \in G\times H}{\mathrm{argmin}} \left(\mathop{\mathbb{E}_{z\sim q_x}}(\log q_x(z)) - \mathop{\mathbb{E}_{z\sim q_x}}\left(\log \displaystyle \frac{p(x|z)p(z)}{p(x)}\right) \right) \\\\\\\\
 & = \underset{(g, h) \in G\times H}{\mathrm{argmin}} \left(\mathop{\mathbb{E}_{z\sim q_x}}(\log q_x(z)) - \mathop{\mathbb{E}_{z\sim q_x}}(\log p(z)) - \mathop{\mathbb{E}_{z\sim q_x}}(\log p(x|z)) + \mathop{\mathbb{E}_{z\sim q_x}}(\log p(x)) \right) \\\\\\\\
 & = \underset{(g, h) \in G\times H}{\mathrm{argmax}} \left(\mathop{\mathbb{E}_{z\sim q_x}} (\log p(x|z)) - D_{KL}(q_x(z), p(z)) \right) \\\\\\\\
 & = \underset{(g, h) \in G\times H}{\mathrm{argmax}} \left(\mathop{\mathbb{E}_{z\sim q_x}} \left(\displaystyle -\frac{||x - f(z)||}{2c}^2\right) - D_{KL}(q_x(z), p(z)) \right)
\end{align}
$$

Podemos identificar que existen dos términos, el error de reconstruccion entre $x$ y $f(z)$ y el término de regularización dado por la divergencia de Kullback-Leibler entre $q_x(z)$ y $p(z)$. Podemos también identificar la constante $c$ que balancea los dos términos. A mayor $c$ asumimos mayor varianza alrededor de $f(z)$.

Ahora si llevamos el modelo a redes neuronales, tendríamos una arquitectura como la mostrada en la figura 1.

<div align="center">

![vae](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/2f72cc5476a22b888e00ab78ca11a29d1ff0c738/vae-arch.png)

_Fig 1: Arquitectura codificador-decodificador variacional_

</div>

Notar que el espacio latente no son puntos fijos, si no que cada componente mapea a una distribución de probabilidad. Notar que en medio de la red hay un proceso de muestreo. Este muestreo, debe hacerse tal que permita que el error se propague en la red neuronal (para actualizar los parámetros de la red).

Si simplemente muestreamos en medio de la red, va a ocurrir que agregamos aleatoriedad al proceso y el gradiente no va a poder fluir ya que será aleatorio en cada paso del algoritmo de retro-propagación. Un truco para evitar esto, es **el truco de la re-parametrización**. En este caso, dado que $z$ es una variable aleatoria que sigue una distribución Gaussiana, con media $g(x)$ y covarianza $H(x) = h(x) \cdot h^T(x)$, $z$ se puede expresar como:

$$z = h(x) \zeta + g(x) \quad \quad \zeta \sim \cal{N(0, I)}$$

<div align="center">

![repar](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/2f72cc5476a22b888e00ab78ca11a29d1ff0c738/reparametrisation.png)

_Fig 2: Truco de la reparametrización_

</div>

Finalmente, dada una imagen $x$ y su reconstrucción $\hat{x}$, entonces la función de costo que debe minimizarse para entrenar el codificador-decodificador variacional, puede escribirse como:

$$Loss = C ||x - \hat{x}||^2 + D_{KL}(\cal{N(\mu_x, \sigma_x)}, \cal{N(0, I)})$$

### Generando Sprites (¡Pokemones!)

Por diversión, y para revivir años de vicio, quise intentar generar sprites de videojuegos. El conjunto de datos que utilicé es una lista de sprites de Pokémon.

<div align="center">

![dataset](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d8d410828cf4816ae6e00daf17724c0b1a0849ed/pkmn-icons.png)

_Fig 4: Conjunto de datos utilizado._

</div>

#### Modelo VAE

En `pytorch` el codificador se vería como el siguiente código:

```py
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            torch.nn.Conv2d(n_channels,
                            n_encoder_features,
                            kernel_size=(2, 2),
                            stride=(2, 2)),
            nn.BatchNorm2d(n_encoder_features),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_encoder_features,
                            n_encoder_features * 2,
                            kernel_size=(2, 2),
                            stride=(2, 2)),
            nn.BatchNorm2d(n_encoder_features * 2),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_encoder_features * 2,
                            n_encoder_features * 4,
                            kernel_size=3,
                            stride=1),
            nn.BatchNorm2d(n_encoder_features * 4),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_encoder_features * 4,
                            n_encoder_features * 8,
                            kernel_size=3,
                            stride=1),
            nn.BatchNorm2d(n_encoder_features * 8),
            nn.ReLU(inplace=True),
        )
        # After all the convolutions we end up with a tensor of size 3x6
        # Al finalizar las convoluciones con los parámetros definidos:
        # (batch, n_encoder_features * 8, 3, 6)
        self.flatten = torch.nn.Flatten()
        self.dense = nn.Sequential(
            torch.nn.Linear(8 * n_encoder_features * 3 * 6, 2 * z_dim),
        )

    def _reparametrize(self, mu, log_var):
        zeta = torch.randn(*mu.shape, device=device)
        return mu + torch.exp(log_var / 2) * zeta

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        z = self.dense(x)
        # De una capa obtenemos mu y log_var
        mu, log_var = z[:,:self.z_dim], z[:,self.z_dim:]
        z = self._reparametrize(mu, log_var)
        return z, mu, log_var
```

Notar que al reparametrizar retorno el vector `z` muestreado, y los vectores de reparametrización $\mu$ y $\log \sigma^2$. La razón para trabajar con el logaritmo, es simplemente para la estabilidad del modelo. Si queremos convertir a la ecuación anterior, entonces tenemos que $\log \sigma^2 = 2 \log \sigma$, entonces `torch.exp(log_var / 2)` simplemente es $\sigma$.

El decodificador, simplemente toma este vector latente, y a partir de transformaciones (ej. convoluciones transpuestas), reconstruye la imagen original. Mi decodificador se ve algo así:

```py
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            torch.nn.Linear(z_dim, 8 * n_encoder_features * 3 * 6),
            torch.nn.Unflatten(1, (8 * n_encoder_features, 3, 6)),
            torch.nn.ConvTranspose2d(8 * n_encoder_features,
                                     n_decoder_features * 8,
                                     kernel_size=1,
                                     stride=1),
            nn.BatchNorm2d(n_decoder_features * 8),
            nn.ReLU(True),
            torch.nn.ConvTranspose2d(8 * n_encoder_features,
                                     n_decoder_features * 4,
                                     kernel_size=2,
                                     stride=2),
            nn.BatchNorm2d(n_decoder_features * 4),
            nn.ReLU(True),
            torch.nn.ConvTranspose2d(n_decoder_features * 4,
                                     n_decoder_features * 2,
                                     kernel_size=(5, 9),
                                     stride=(2, 1)),
            nn.BatchNorm2d(n_decoder_features * 2),
            nn.ReLU(True),
            torch.nn.ConvTranspose2d(n_decoder_features * 2,
                                     n_decoder_features,
                                     kernel_size=(2, 2),
                                     stride=(2, 2)),
            nn.BatchNorm2d(n_decoder_features),
            nn.ReLU(True),
            torch.nn.ConvTranspose2d(n_decoder_features, n_channels, kernel_size=(1, 1), stride=1),
            torch.nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)
```

Utilicé la función de activación $Tanh$, para que todos los elementos se encuentren entre -1 y 1 (mejor estabilidad y convergencia); Aunque podría haber utilizado una función sigmoide. Lo siguiente es definir la función de costo:

```py
reconstruction_loss = torch.nn.MSELoss()
def vae_reconstruction_loss(y_true, y_pred, reconstruction_loss_factor):
    return reconstruction_loss_factor * reconstruction_loss(y_true, y_pred)

def vae_kullback_leibler_loss(mu, log_var):
    return -0.5*torch.sum(1 + log_var - mu**2 - torch.exp(log_var), axis=1)[0]

def vae_loss(y_true, y_pred, mu, log_var, reconstruction_loss_factor=1000):
    recon_loss = vae_reconstruction_loss(y_true, y_pred, reconstruction_loss_factor)
    kld_loss = vae_kullback_leibler_loss(mu, log_var)
    return recon_loss + kld_loss
```

En el caso de la componente de reconstrucción, para hacer más claro el código la variable `reconstruction_loss_factor` representa la constante $C$ de la función de pérdida mostrada anteriormente. La función `vae_kullback_leibler_loss` calcula $D_{KL}(\cal{N(\mu_x, \sigma_x)}, \cal{N(0, I)})$:

$$
\begin{align}
D_{KL}(\cal{N(\mu_x, \sigma_x)}, \cal{N(0, I)}) & = \mathop{\mathbb{E}} \left[\log \cal{N(\mu_x, \sigma_x)} - \log \cal{N(0, I)}\right] \\\\
& = \frac{1}{2} \left[\mu_x^2 + \sigma_x^2 - 1 - \log \sigma_x^2\right] \\\\\\\\
& = -\frac{1}{2} \left[1 + \log \sigma_x^2 - \mu_x^2 - \sigma_x^2 \right]
\end{align}
$$

Tuve que probar varios valores para el factor de error reconstrucción, al final los mejores resultados los obtuve con $C = 5000$. Para verificar el aprendizaje del modelo, fui monitoreando la función de costo en cada _epoch_. Entrené el modelo en `500` epochs.

<div align="center">

![vae-train](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/419061e6eacb5a988f1f0d34c08bffe05dd3e82f/vae-loss-training.png)

_Fig 4: Entrenamiento VAE en conjunto de datos._

</div>

Luego, tomé una muestra al azar, y inspeccioné las reconstrucciones que hace el modelo:

<div align="center">

![sample](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/419061e6eacb5a988f1f0d34c08bffe05dd3e82f/original.png)

_Fig 5: Ejemplo de datos en su versión original._

</div>

<div align="center">

![sample-recons](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d8d410828cf4816ae6e00daf17724c0b1a0849ed/vae-recons.png)

_Fig 6: Ejemplo de datos reconstruidos por el VAE._

</div>

Finalmente, la parte divertida, generar Pokémones a partir de ruido:

<div align="center">

![vae-gen](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d8d410828cf4816ae6e00daf17724c0b1a0849ed/generated-sprites.png)

_Fig 7: Sprites de Pokémones generados a partir de muestreo en espacio latente y decodificador._

</div>

La verdad, para mi sigue siendo mágico que a partir de ruido se puedan generar nuevos sprites (pixel art). Cuando probé con el conjunto de datos de las caras de celebridades o el MNIST, también, en el espacio latente podía hacer operaciones lineales, y generar datos nuevos, similares a los datos de entrenamiento.

#### ¿Es suficiente? Pixel-art es más complejo que imágenes convencionales

Podemos observar en los Pokémon generados anteriormente, que si bien tienen forma y coloreado similar a los datos vistos en el entrenamiento, estos siguen siendo borrosos y la representación considera un espacio continuo. Sin embargo, el pixel art y las imágenes en general, tienen un conjunto de colores limitados y en un espacio discreto.

Otro problema es que **estamos intentando predecir todos los pixeles considerándolos como independientes**, lo que es un supuesto demasiado simplista.

El pixel-art en general, utiliza una paleta de colores limitada, donde existen técnicas para coloreado, iluminación, aliasing que es diferente a la de una foto convencional (por ejemplo una fotografía "real"). Después de leer varios papers en el tema de pixel art, no encontré ninguna solución a mi problema. Sin embargo, investigando un poco más a fondo y expandiendo llegué a dos papers interesantes, que tratan los problemas con los que me topé:

1. [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)
2. [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)
3. [PixelVAE: A Latent Variable Model for Natural Image](https://arxiv.org/abs/1611.05013)

No quiero alargar el artículo más de la cuenta, así que no explicaré los modelos o los papers. En esencia, lo que intentan hacer estos modelos es, modelar el problema como un problema de predicción del siguiente pxiel (¿suena a algo parecido a lo que hacemos en NLP? 😊). Básicamente, queremos encontrar una distribución de probabilidad tal que:

$$p(x) = p(x_1, \ldots, x_n) = \displaystyle \prod_{i=1}^{n} p(x_i|x_1, \ldots, x_{i - 1})$$

Para lograr esto, se hace un modelamiento de imágenes _autorregresivo_, en este caso, el siguiente pixel, depende de los pixeles anteriores. Para ver detalles y un ejemplo de implementación simple, el siguiente tutorial es bastante completo:

* [Tutorial 12: Autoregressive Image Modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html)

La verdad, yo utilicé una pequeña variación del tutorial que acabo de mencionar.

<div align="center">

![pixelcnn-loss](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d0faec7873d6d4aa30a64d25204c5cbaeab12e51/pixelcnn-loss.png)

_Fig 8: Curva de aprendizaje de modelo PixelCNN con datos de sprites de Pokémon._

</div>

Intenté generar nuevos sprites con el modelo entrenado:

<div align="center">

![gen-pixelcnn](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d0faec7873d6d4aa30a64d25204c5cbaeab12e51/pixel-cnn-gen.png)

_Fig 9: Pokémones generados con PixelCNN._

</div>

También intenté hacer _autocompletado_ dada una parte de una imagen:

<div align="center">

![autocomplete1](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d0faec7873d6d4aa30a64d25204c5cbaeab12e51/sample-masked.png)

![autocomplete2](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/d0faec7873d6d4aa30a64d25204c5cbaeab12e51/sample-autocompleted.png)

_Fig 10: Autocompletado de Pokémones._

</div>

Para el muestreo, utilicé imágenes RGB, es decir 3 canales de color, y en este ejemplo se consideran los canales como independientes a la hora de muestrear. Esto en la práctica no es así, ya que el pixel-art utiliza una paleta de colores definida y limitada. La pregunta que me hago es ¿Existe alguna forma de considerar la paleta de colores en la entrada, para no tener que calcular `256 * n_canales` de intensidades de pixeles posible?

La verdad sigo pensando e investigando cómo lograr esto, pero hasta ahora no he tenido buenos resultados 😔.

#### Observaciones sobre implementación y parámetros

Algunas observaciones que recalco y que me causaron curiosidad:

* La tasa de aprendizaje y el tamaño de los lotes (_batches_) influyen en la convergencia de la red
    * Tuve explosión de gradiente cuando cualquiera de estos parámetros superaba ciertos umbrales.
    * Óptimos locales dependiendo del parámetro de momentum.
* Normalización en la retro-propagación. Interesante, ya que sin normalizar también tuve problemas de explosión de gradiente y divergencia
* En la deconvolución repetir pixeles (_Upsampling_) o jugar con el kernel y distintos tamaño de saltos (_Stride_), también son muy dependientes del problema.
* Repetir las arquitecturas de los tutoriales, siempre resulta para ese caso específico y para los conjuntos de datos en las evaluaciones comparativas (_Benchmark_); ejemplo: _CIFAR_, _MNIST_, _Celeb Faces_, etc.
* Probé otras múltiples arquitecturas: `StyleGAN`, `StyleGAN2`, `Pix2Pix`, `CycleGAN`. En el caso de las GAN tuve múltiples problemas como explosión de gradiente y desvanecimiento del mismo. Este tipo de redes son muy inestables al parecer. Por otro lado, las imágenes generadas no parecían pixel-art (manchas peores que las mostradas en este artículo 😂)
* La cantidad de epochs es básicamente lo más importante, de ahí que el tener disponibilidad de GPU y poder de cómputo es un factor diferenciador.

# Conclusiones

* Se deben tener claro los objetivos de la agilidad, y tratar de adoptar únicamente prácticas que habiliten trabajar de manera ágil y no simplemente seguir recetas. Empresa `X` tiene necesidades tipo `Y` en un contexto `Z`. No se puede llegar y repetir como loro una receta y esperar que funcione.
* Los profesionales somos suficientemente adultos para tomar decisiones que impacten en el tiempo y productividad. Cancelar reuniones no tiene nada de malo, siempre y cuando sea con tiempo y justificado.
* VAE es un modelo generativo que intenta ajustar una distribución a cada punto en un espacio latente, lo que permite generar nuevos datos a partir de muestreo en dicho espacio.
* Generar pixel art es una tarea mucho más compleja que generar imágenes, debido a la naturaleza del pixel art: Paleta de colores limitada
