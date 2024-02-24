---
layout: post
title:  "Generando Imágenes con VQ-VAE"
date:   2024-02-23 10:00:00 -0400
categories: python algorithms ai
---

# Introducción

En mi post _[Reflexiones y Jugando con Pixeles]({{ site.baseurl }}{% link _posts/2024-02-18-reflexiones-y-jugando-con-pixeles.markdown %})_ expliqué cómo generar imágenes con un Autocodificador variacional. Expliqué cómo las imágenes son codificadas a un espacio latente, donde en lugar de mapear a un "punto" como en el caso de los auto-codificadores, se mapea a una distribución de probabilidad. También expliqué que el problema de inferencia era en esencia aproximar la distribución condicional $p(z|x)$ a una distribución Gaussiana $q_x(z)$, tal que su media y covarianza están definidas por dos funciones, $g$ y $h$

$$
\begin{align}
 (g^*, h^*) & = \underset{(g, h) \in G\times H}{\mathrm{argmin}} D_{KL}(q_x(z), p(z|x))
\end{align}
$$

Y que esto llevado a una red neuronal, se vería como la figura 1. Donde la función de pérdida a minimizar estaría dada por:

$$Loss = C ||x - \hat{x}||^2 + D_{KL}(\cal{N(\mu_x, \sigma_x)}, \cal{N(0, I)})$$

**Nota: Para ver la derivación completa, y experimentos leer el post anterior.**

<div align="center">

![vae](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/2f72cc5476a22b888e00ab78ca11a29d1ff0c738/vae-arch.png)

_Fig 1: Arquitectura codificador-decodificador variacional_

</div>

En este post tocaré dos temas:

1. Algunas reflexiones de mi 2023, para algunos resonará, otros quizás no tanto
2. Hablaré sobre una mejora a este modelo VAE: VQ-VAE, que es la base de modelos sofisticados como DALL-E para generación de imágenes.

# Reflexiones Iniciales

Me he dado cuenta que los posts que escribo y que más enganche tienen son los que digo algo polémico. Y voy a ser honesto, me gusta _trolear_ a cierto tipo de personas, y también me gusta desafiar las creencias de la gente en general. Y cuando genero molestia, en verdad me causa cierto placer. Lo que ya no hago, y solía hacer, es enganchar mucho en la discusión, ya que termino perdiendo tiempo.

Lo que me causa un poco de tristeza, es que los artículos que escribo, donde intento explicar de forma "simple" y aterrizada cómo funcionan ciertos algoritmos y sistemas en el mundo actual (con fines de reducir el sensacionalismo), no tienen tanta recepción. Pero bueno, supongo que tengo que mejorar en "venderme a mi mismo" cosa que nunca he sido bueno, porque soy demasiado realista, objetivo y riguroso. Curiosamente, me he topado con personas similares en la industria y también se les hace difícil "venderse". Sin embargo, hay que aclarar que el tener miles de seguidores, no es sinónimo de conocimiento/veracidad de la información, es sólo publicidad y venderse. No niego que hay _influencers_ que crean muy buen material, sin embargo, en redes como LinkedIn esto no es el caso general.

Finalmente, me gustaría aclarar: **Yo no soy un _influencer_**. La verdad, mis artículos toman tiempo, necesito estar motivado, encontrar un buen tema y además intentar crear material de calidad, demostrando la veracidad de lo expuesto mediante el método científico. Ello toma tiempo y lamentablemente no vivo de esto. Si quisiera monetizar o ser _influencer_, escribiría mis artículos en lugares como **Medium**, y no en una web _chafa_ (me gusta esta palabra) como la que hago yo utilizando Github 🤣.

Finalmente, aclarar que soy un Ingeniero de Software/Machine Learning, promedio; He conocido personas mucho más inteligentes/mejores que yo, y personas peores. No obstante, siempre intento hacer las cosas de la mejor calidad que mi inteligencia me hace posible. No me conformo con una evaluación de desempeño de _"Meets All"_, tengo hambre de _"Greatly Exceed Expectations"_.

# Un poco más de IA generativa

Aún voy atrasado en temas de conocimiento de los fundamentos de las tecnologías actuales (recién voy por el 2017 😅), sin embargo de a poco agarro el vuelo. Se me ha hecho bastante difícil entender los papers y ¡luego implementarlos! pero ya vamos de a poco. Como dije anteriormente, no soy inteligente y de verdad me _impresionan los genios de GenAI que apenas sale una nueva tecnología se vuelven expertos_ (ejem... ChatGPT, Sora, DALL-E, LLAMA).

En esta sección explicaré el modelo fundamental utilizado en sistemas como DALL-E, en la parte de generación de imágenes. En particular, explicaré el modelo de Codificador Variacional con Cuantización Vectorial (VQ-VAE _Vector Quantised-Variational Autoencoder_), y una simple implementación. Por otro lado, también intentaremos generar imágenes nuevas utilizando este mismo modelo.

## Entendiendo el VQ-VAE

Este modelo fue introducido en el paper [_Neural Discrete Representation Learning_](https://arxiv.org/abs/1711.00937) y es una modificación que intenta resolver los problemas del VAE. En este caso, en lugar de tener una distribución continua en el espacio latente (distribuciones apriori y aposteriori se asumen Gausianas en este modelo), se obtiene una distribución discreta que se basa en cuantización vectorial, lo que implica distribuciones categóricas tanto apriori, como aposteriori. En palabras simples, se obtienen reconstrucciones más nítidas.

La arquitectura de la red neuronal para el modelo VQ-VAE se muestra en la figura 2:

<div align="center">

![vae](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/3ce75eda3bd933f8ceba04c2d9ff2db3546b89ef/Screenshot%25202024-02-24%2520at%25203.13.06%25E2%2580%25AFPM.png)

_Fig 2: Arquitectura VQ-VAE_

</div>

Se define un espacio latente de _embeddings_ $e \in \mathbb{R}^{K\times D}$, donde $K$ es la cantidad de categorías, es decir, el número de _embeddings_ y $D$ es la dimensión de cada vector latente. El codificador tiene una entrada $x$ y produce una salida $z_e(x)$. Las variables latentes discretas $z$ se calculan como el vector más cercano (_nearest neighbor_) del conjunto de vectores $e$. La entrada para el decodificador, es el vector $e_k$ obtenido en el paso previo:

$$z_q(x) = e_k, \quad \text{donde} \quad k = \text{argmin}_j ||z_e(x) - e_j||_2$$

La distribución categorica $q(z|x)$ (codificador) está definida como:

$$
q(z=k|x) =
	\begin{cases}
		1  & \mbox{for } k = \text{argmin}_j ||z_e(x) - e_j||_2 \\\\\\\\
		0 & \mbox{en caso contrario }
	\end{cases}
$$

En el paper, se propone $q(z = k | x)$ como determinista, y al definir $p(z)$ como una distribución uniforme. Si recordamos que un VAE intenta optimizar:

$$
\begin{align}
 (g^*, h^*) & = \underset{(g, h) \in G\times H}{\mathrm{argmin}} D_{KL}(q_x(z), p(z|x)) \\\\\\\\
 & = \underset{(g, h) \in G\times H}{\mathrm{argmin}} \left(\mathop{\mathbb{E}_{z\sim q_x}}(\log q_x(z)) - \mathop{\mathbb{E}_{z\sim q_x}}\left(\log \displaystyle \frac{p(x|z)p(z)}{p(x)}\right) \right) \\\\\\\\
 & = \underset{(g, h) \in G\times H}{\mathrm{argmin}} \left(\mathop{\mathbb{E}_{z\sim q_x}}(\log q_x(z)) - \mathop{\mathbb{E}_{z\sim q_x}}(\log p(z)) - \mathop{\mathbb{E}_{z\sim q_x}}(\log p(x|z)) + \mathop{\mathbb{E}_{z\sim q_x}}(\log p(x)) \right) \\\\\\\\
 & = \underset{(g, h) \in G\times H}{\mathrm{argmax}} \left(\mathop{\mathbb{E}_{z\sim q_x}} (\log p(x|z)) - D_{KL}(q_x(z), p(z)) \right) \\\\\\\\
 & = \underset{(g, h) \in G\times H}{\mathrm{argmax}} \left(\mathop{\mathbb{E}_{z\sim q_x}} \left(\displaystyle -\frac{||x - f(z)||}{2c}^2\right) - D_{KL}(q_x(z), p(z)) \right)
\end{align}
$$

Entonces, en este caso no tenemos una distribución Gaussiana $q_x(z)$, si no que una distribución categórica $q(z|x)$ como la descrita anteriormente, entonces, dado que la distribución $q(z|x)$ tiene un valor distinto de cero sólo en $q(z = k|x)$, se tiene que la divergencia de Kullback-Leibler es constante:

$$
\begin{align}
D_{KL}(q(z|x), p(z)) & = \sum_{z \in \mathop{Z}} q(z|x)\log\left(\frac{q(z|x)}{p(z)}\right) \\\\\\\\
& = q(k | x) \log\left(\frac{q(k|x)}{p(k)}\right) \\\\\\\\
& = 1 \cdot \log \left(\frac{1}{\frac{1}{K}}\right) \\\\\\\\
& = \log K
\end{align}
$$

En el caso de la ecuación $z_q(x) = e_k$ (entrada al decodificador), no tiene un gradiente definido, sin embargo se puede aproximar el gradiente de forma similar al estimador directo (_straight-through estimator_), y simplemente copiar los gradientes de la entrada del decodficador $z_q(x)$ a la salida del codificador $z_e(x)$. [Este post en Medium](https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0) tiene una explicación muy intuitiva de este estimador. [En este repositorio en Github](https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py) hay una implementación intuitiva de este estimador.

En el caso de cuantización vectorial:

```py
class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')
```

Se calculan las distancias entre el vector de entrada y los vectores del espacio de embeddings. En el caso de este código, lo que está haciendo es calcular el mínimo de la distancia al cuadrado (que sería lo mismo que minimizar la distancia, pero sin el costo computacional de calcular la raíz):

$$
\begin{align}
||X - Y||_2^2 & = ||X||_2^2 - 2 X \cdot Y + ||Y||_2^2 \\\\\\\\
& = ||X||_2^2 + ||Y||_2^2 - 2 X \cdot Y
\end{align}
$$

En el caso del estimador _straight-through_, puede implementarse como:

```py
class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)
```

En este caso, se aplica cuantización vectorial en la propagación hacia adelante y se guardan los vectores resultantes del espacio de embeddings y sus correspondientes índices en el diccionario de vectores. No se calculan los gradientes respecto a $z_q(x)$, y en el caso de los vectores del diccionario de embeddings, el objetivo es minimizar $||\text{sg}[z_e(x)] - e||_2^2$.

Finalmente para la función de costo del VQ-VAE tenemos que considerar 3 ingredientes:

1. La pérdida de reconstrucción $\log p(z|z_q(x))$.
2. Dado que los embeddings $e_i$ no reciben gradientes por reconstrucción, usamos un simple algoritmo: Cuantización vectorial. En este caso, como mencionamos previamente, el término a monimizar es $||\text{sg}[z_e(x)] - e||_2^2$
3. Finalmente, dado que el espacio de embeddings puede crecer arbitrariamente si los embeddings $e_i$ no se entrenan tan rápido como los parámetros del codificador, agregamos un término de regularización $\beta ||z_e(x) - \text{sg}[e]||_2^2$.

Finalmente, la función de pérdida a minimizar es:

$$Loss = p(z|z_q(x)) + ||\text{sg}[z_e(x)] - e||_2^2 + \beta ||z_e(x) - \text{sg}[e]||_2^2$$

Dado que el término que corresponde a la divergencia de Kullback-Leibler es constante dado los supuestos, este término se ignora, ya que no tiene efecto en la función de pérdida.

La distribución apriori sobre los vectores latentes $p(z)$ se asume como uniforme. Sin embargo, para el proceso generativo, se puede estimar otra distribución, por ejemplo utilizando un modelo autoregresivo como _PixelCNN_, y ello nos permitirá generar imágenes de acuerdo al estilo de los datos de entrenamiento.

### Entrenando VQ-VAE

En esta sección mostraré dos experimentos:

1. Utilizando un sub-conjunto del conjunto de datos [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
2. Generar Pokémones en base a pixel art

Primero definimos los embeddings del cuantizador vectorial:

```py
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar
```

La definición de este embedding requiere dos parámetros:

1. $K$ que es la cantidad de categorías o elementos que tendrá el diccionario de embeddings
2. $D$ que es la dimensionalidad de cada embedding.

Ambos parámetros afectan la reconstrucción, por lo que ajustarlos depende del problema. El método `straight_through` simplemente aplica la estrategia mencionada previamente para actualizar los embeddings vía el gradiente. La permutación de componentes, es debido a que la entrada de los vectores en procesamiento de imágenes es `(B, C, H, W)`, donde `C` es la cantidad de canales (ejemplo `RGB`) y queremos aplicar la misma multiplicación para todos los canales. Finalmente para computar los vectores latentes, estos se re-permutan para volver a las componentes originales. Finalmente, el codificador decodificador queda como:

```py
class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        # (B, D, H, W)
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
```

Debe notarse que ahora, en lugar de re-parametrizar y obtener una distribución continua, se usa este diccionario de vectores para reconstruir la entrada codificada. El entrenamiento, se ve como sigue, lo ejecuté por `100` epochs.

```py
train_loss = []
num_epochs = 100

# Another hyperparameter
for epoch in range(num_epochs):
    avg_loss = 0
    for img in train_loader:
        # Convertir tipo a float
        img = img.to(device)
        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(img)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, img)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + BETA * loss_commit
        loss.backward()
        optimizer.step()

        # Actualizar promedio de pérdida
        avg_loss += loss.item()

    avg_loss /= BATCH_SIZE
    train_loss.append(avg_loss)
    print(f"Epoch {epoch + 1}|{num_epochs}; Running loss {avg_loss}")
```

La curva de aprendizaje que obtuve:

<div align="center">

![vq-vae-learning](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/5bd96588b648795058582c5e8ade76024c6d68fc/vq-vae-learning.png)

_Fig 3: Curva de aprendizaje en CIFAR10 (aviones) para modelo VQ-VAE definido._

</div>

Ejemplo de imágenes originales y reconstrucción:

<div align="center">

![vq-vae-orig](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/5bd96588b648795058582c5e8ade76024c6d68fc/vq-vae-originals.png)

![vq-vae-recons](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/5bd96588b648795058582c5e8ade76024c6d68fc/vq-vae-recons.png)

_Fig 4: (arriba) Ejemplos de imágenes de aviones CIFAR 10 (abajo) Reconstrucción de los ejemplos con VQ-VAE_

</div>

### Re-Ajustando $p(z)$ con Modelo Auto-Regressivo PixelCNN

En esta sección estimaremos $p(z)$ utilizando el modelo auto-regresivo PixelCNN. En lugar de pixeles, en este caso predecimos los vectores diccionario de embeddings asociados a cada posición de la imagen en el espacio latente:

$$p(z) = \prod_i^K p(z_i|z_1, z_2, \ldots, z_{i-1})$$

El modelo PixelCNN a definir en `Pytorch`:

```py
class PixelCNN(nn.Module):

    def __init__(self, c_in, c_hidden):
        super().__init__()

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(c_in, c_hidden, mask_center=True)
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=4),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden)
        ])
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(c_hidden, c_in * 512, kernel_size=1, padding=0)

    def forward(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to K - 1 back to -1 to 1
        x = (x.float() / (K - 1)) * 2 - 1

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(out.shape[0], K, out.shape[1]//K, out.shape[2], out.shape[3])
        return out
```

Entrenando modelo PixelCNN para obtener $p(z)$:

```py
prior_train_loss = []
num_epochs = 100
for epoch in range(num_epochs):
    avg_loss = 0
    for img in train_loader:
        # Encode with the VQ-VAE
        with torch.no_grad():
            img = img.to(device)
            latents = model.encode(img)
            shape = latents.shape
            latents = latents.view(shape[0], 1, shape[1], shape[2])
            latents = latents.detach()

        # Calc likelihood
        pred = prior(latents)
        nll = F.cross_entropy(pred, latents, reduction='none')
        bpd = nll.mean(dim=[1,2,3]) * np.log2(np.exp(1))
        loss = bpd.mean()

        # Update weights
        prior_optimizer.zero_grad()
        loss.backward()
        prior_optimizer.step()

        # Update average loss
        avg_loss += loss.item()

    avg_loss /= batch_size
    prior_train_loss.append(avg_loss)
    print(f"Epoch {epoch + 1}|{num_epochs}; Running loss {avg_loss}")
```

Ahora con $p(z)$ re-calculada, podemos generar imágenes de nuevos aviones haciendo un muestreo en esta distribución:

```py
def sample(img_shape, out_img=None):
    with torch.no_grad():
        if out_img is None:
            out_img = torch.zeros(img_shape, dtype=torch.long).to(device) - 1
        # Generation loop
        for h in tqdm(range(img_shape[2]), leave=False):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # Skip if not to be filled (-1)
                    if (out_img[:,c,h,w] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    pred = prior.forward(out_img[:,:,:h+1,:])
                    probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                    out_img[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
    return out_img
```

Las imágenes generadas se muestran en la figura 5:

<div align="center">

![vq-vae-gen](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/5bd96588b648795058582c5e8ade76024c6d68fc/generated-samples.png)


_Fig 5: Imágenes generadas muestrando desde la distribución $p(z)$ y reconstruyendo con el decodificador._

</div>

#### ¿Qué pasó con mis Pokémon?

Sólo por diversión, quiero ver qué ocurriría con el conjunto de datos que utilicé en mi post previo (sprites de íconos de pokémon):

<div align="center">

![vq-vae-pk-orig](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/e33c649b57c0662ac2152ffa0df841d3c5a6e690/vq-vae-pkmn-orig.png)

![vq-vae-pk-recons](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/e33c649b57c0662ac2152ffa0df841d3c5a6e690/vq-vae-pkmn-recons.png)


_Fig 6: (arriba) Muestras de Pokémon del conjunto de datos. (abajo) Reconstrucciones con VQ-VAE._

</div>

Ahora, generemos nuevos Pokémones:

<div align="center">

![vq-vae-pk-gen](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/e33c649b57c0662ac2152ffa0df841d3c5a6e690/vq-vae-pkmn-gen.png)


_Fig 7: Generación de Pokémones muestreando de $p(z)$._

</div>

En este caso generamos nuevos íconos, curioso que también logramos muestrear pokémones cercanos a los que ya existían en el conjunto de entrenamiento.

# Reflexiones Finales

En este artículo expliqué cómo funciona uno de los modelos fundamentales en GenAI. Este modelo es la base del conocido DALL-E, claro que DALL-E utiliza otros trucos, en lugar de utilizar un VQ-VAE utiliza una adaptación llamada `dVAE` pero la idea es similar. Para más detalle ver paper [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092). En simples palabras, ellos logran dado una imagen $x$ y un texto $y$, logran estimar $p(x, y)$ (en realidad, en un espacio latente), y muestrean nuevas imágenes dado un texto.

¿Va la inteligencia artificial exterminarnos? De momento, creo que no. Si bien, en LinkedIn hay mucho _hype_ y sensacionalismo (también en algunos medios), ya observamos que en GenAI el patrón simple es encontrar una distribución de probabilidad y muestrear de la misma. Claro que hay varios trucos ingenieriles, los cuales desconozco, para lograr la calidad de contenido que generan estos sistemas de AI. El resto de mis opiniones, me las guardo.

Finalmente, sólo como dato, quise hacer experimentos rápidos para validar mi entendimiento en estos temas y modelos. No utilicé una cantidad de datos abismal, ya que tengo GPU limitada (y dinero limitado 😂), pero la industria y las empresas con mayor poder adquisitivo cuentan con una mejor infraestructura, un mejor equipo de ingenieros, mayor cantidad de datos y mucho mayor poder de cómputo.

# Conclusiones

* No soy un influencer 😤
* El modelo generativo VQ-VAE + PixelCNN es la base de sistemas como DALL-E
* Generar imágenes en este contexto de GenAI, es simplemente muestrear de una distribución $p(x)$, que se estima mediante modelos de redes neuronales
* La AI no nos va a exterminar por lo pronto, y mucho sensacionalismo y _posts_ de influencers no tienen base alguna
* Sigo sin entender qué es ser experto en GenAI
