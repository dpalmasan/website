---
layout: post
title:  "Reinforcement Learning: Multi-Armed Bandits"
date:   2024-07-04 20:37:00 -0400
categories: algorithms ai
---

# Introducción

En este post hablo sobre un algoritmo clásico de aprendizaje por refuerzo (o en inglés _Reinforcement Learning_), y el que más he visto en la práctica (experiencia personal). El algoritmo al cual hago referencia es el **bandido multibrazo**, o su nombre en inglés _Multi-Armed bandits_ (al cuál nos referiremos como MAB).

# Multi-Armed Bandits (MAB)

Tengamos la siguiente situación en la que estamos en un casino jugando la máquina traga-monedas y tenemos $k$ palancas para mover. Cada acción a tomar es mover una palanca y esta tendrá alguna probabilidad de ganar el premio. Idealmente queremos mover más la "mejor" palanca, es decir, la que más frecuentemente nos permita ganar (recompensa). Otra posible situación similar es tener una suite de pruebas (integración continua en software) y un conjunto de máquinas para ejecutar dichas pruebas. Queremos distribuir los tests de manera que la suite termine de ejecutarse en la menor cantidad de tiempo.

En un MAB tenemos $k$ acciones, y cada una de dichas acciones tiene una recompensa esperada, dada la acción a ejecutar. Llamemos a esta recompensa esperada el "valor" de una acción. Denotamos la acción escogida en el tiempo $t$ como $A_t$, y su recompensa correspondiente como $R_t$. El valor de una acción arbitraria $a$, se denota como $q_*(a)$, y la recompensa esperada si se elige dicha acción:

$$q_*(a) = \mathop{\mathbb{E}}\left[R_t|A_t=a\right]$$

Si supieramos el valor de cada acción, el problema sería trivial. Simplemente, elegiriamos siempre la acción con la mejor recompensa. Asumiremos que no conocemos los valores de cada acción, aunque podemos tener un valor estimado para la acción, $a$ en el tiempo $t$ que llamaremos $Q_t(a)$. Queremos que $Q_t(a)$ esté lo más cerca posible de $q_*(a)$.

Dado que tenemos un estimado del valor de cada acción en un momento dado, podríamos preguntarnos ¿Qué estrategia seguir para elegir la siguiente acción? Lo más _codicioso_ sería elegir la acción que estimamos, tendrá la mayor recompensa. En este caso, estamos **explotando** el conocimiento que tenemos, ya que escogemos la acción que con la información que tenemos, parece ser la mejor; a esta estrategia le llamaremos _greedy_. Por otro lado, si escogemos una acción de forma arbitraria, estaríamos **explorando**, ya que esto nos permitiría mejorar nuestra estimación del valor de dicha acción. Cuando queremos maximizar la recompensa esperada en un momento $t$, explotar la información es una buena estrategia. Sin embargo, esta es una estrategia corto-placista, ya que permitiéndonos escoger una acción no tan buena a primera vista, podríamos incluso mejorar nuestra recompensa en el largo plazo. Esto es en esencia el conflicto que existe entre exploración y explotación.

## Métodos de Acción-Valor

Supongamos por el momento que estamos frente a un problema _estacionario_, esto quiere decir que los valores $q_*(a)$ se mantienen en el tiempo. ¿Cómo podemos tener una buena estimación del valor de cada acción? Una forma natural de obtener esta estimación, es promediando las recompensas obtenidas en el tiempo:


$$Q_t(a) = \frac{\text{Suma de recompensas para la acción } a \text{ previo al paso } t}{\text{Número de veces que tomamos } a \text{ previo al paso } t}$$


$$
Q_t(a) = \frac{\sum\_{i=1}^{t-1}{R\_i\cdot\mathbb{1}_{A\_i=a}}}{\sum\_{i=1}^{t-1}\mathbb{1}\_{A_i=a}}
$$

Donde $\mathbb{1}$ denota una variable aleatoria que es 1 si la condición se cumple y 0 en caso contrario. Si el denominador es cero, entonces definimos $Q_t(a)$ a algún valor constante. Cuando el denominador tiende al infinito, debido a la ley de los grandes números, $Q_t(a)$ converge a $q_*(a)$. Llamamos a este método _muestreo-promedio_ porque cada estimado es un promedio de una muestra de recompensas. Claro que esta es sólo una forma de calcular este valor estimado y no necesariamente la mejor.

La regla más simple para escoger una acción es elegir la que tenga el mayor valor estimado, es decir:

$$A_t = \underset{a}{\mathrm{argmax}} \ Q_t(a)$$

Esta estrategia siempre aplicará explotación, ya que no intentará tomar acciones que tengan menos recompensa. Sin embargo, como mencionamos anteriormente, en algunos casos la acción que parece ser la mejor no es la que en el largo plazo nos dará la mejor recompensa. Una alternativa es definir un valor de probabilidad $\varepsilon$ pequeño de tal forma que con dicha probabilidad escogemos un valor aleatorio.

# 10-Arm Bandit

Consideremos el caso de un MAB de 10 "brazos". Supongamos que conocemos la funcion de valores para cada acción, cuya media fue obtenida muestreando una distribución Gaussiana y que \$q\_*(a)\$ sigue una distribución Gaussiana, tal como se muestra en la figura 1. El código para generar el bandido:

```py
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 200

n_actions = 10


def create_multiarmed_bandit(n_actions: int) -> Tuple[List[int], np.array]:
    mu = 0
    sigma = 1
    actions = [a for a in range(1, n_actions + 1)]
    q_a = np.random.normal(mu, sigma, n_actions)
    return actions, q_a
```

Para crear el gráfico de la figura 1:

```py
actions, q_a = create_multiarmed_bandit(n_actions)

rewards = [
    np.random.normal(q_a[a - 1], 1, 1000) for a in actions
]


plt.violinplot(rewards, )
plt.xlabel("Acción")
plt.ylabel("Recompensa")
plt.title("10-armed bandit")
plt.xticks(actions)
```

<div align="center">

![qa_mab](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/22b2f34c8fd05586ff71bde0437d74522e71feb2/mab1.png)

_Fig 1: Función de valor para cada acción del 10-arm bandit._

</div>

Ejecutemos un experimento en el cual tomamos 2000 ejecuciones de distintos MAB, considerando algoritmo descrito anteriormente para estimar $Q_t(a)$, y realizar esto para 1000 pasos de tiempo $t$. Consideremos también utilizar distintos valores de probabilidad de exploración $\varepsilon$.

```py
def stationary_experiment(
    epsilon: float, steps: int, n_actions: int
) -> Tuple[np.array, np.array]:
    average_rewards = np.zeros(steps + 1)
    actions, q_a = create_multiarmed_bandit(n_actions)
    optimal_action = np.argmax(q_a) + 1
    count_optimal_action = 0

    optimal_action_perc = np.zeros(steps + 1)

    # Q_t(a): Estimated expected reward. Initial estimate is 0
    q_t_a = np.zeros((n_actions, steps + 1))

    # Cumulative sum of rewards
    reward_cum_sum = np.zeros(n_actions)

    # Number of steps the action a was taken
    n_steps_a = np.ones(n_actions)
    for t in range(1, steps + 1):
        if np.random.random() <= epsilon:
            action = np.random.randint(1, n_actions + 1)
        else:
            # Take greedy action
            action = np.argmax(q_t_a[:, t - 1]) + 1

        reward_cum_sum[action - 1] += np.random.normal(q_a[action - 1], 1)

        # Estimating expected reward
        q_t_a[:, t] = reward_cum_sum / n_steps_a

        # The reward from the action taken
        average_rewards[t] = q_t_a[action - 1, t]
        count_optimal_action += int(action == optimal_action)
        optimal_action_perc[t] = count_optimal_action / t
        n_steps_a[action - 1] += 1

    return average_rewards, optimal_action_perc


def mab_run_experiment(
    n_experiments, experiment_func: Callable[[float, int, int], Tuple[np.array, np.array]], *args
) -> Tuple[np.array, np.array]:
    average_rewards = np.zeros(steps + 1)
    optimal_action_perc = np.zeros(steps + 1)

    for _ in range(n_experiments):
        average_rewards_, optimal_action_perc_ = experiment_func(
            *args
        )
        average_rewards += average_rewards_
        optimal_action_perc += optimal_action_perc_

    return average_rewards / n_experiments, optimal_action_perc / n_experiments


n_experiments = 2000
steps = 1000
rewards_e0, optimal_action_perc_e0 = mab_run_experiment(
    n_experiments, stationary_experiment, 0, steps, n_actions
)
rewards_e0_01, optimal_action_perc_e0_01 = mab_run_experiment(
    n_experiments, stationary_experiment, 0.01, steps, n_actions
)
rewards_e0_1, optimal_action_perc_e0_1 = mab_run_experiment(
    n_experiments, stationary_experiment, 0.1, steps, n_actions,
)
```

<div align="center">

![r_mab](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/22b2f34c8fd05586ff71bde0437d74522e71feb2/mab2.png)

![opt_mab](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/22b2f34c8fd05586ff71bde0437d74522e71feb2/mab3.png)

_Fig 2: Medidas de desempeño para experimento de 10-arm bandit.._

</div>

Podemos observar que siguiendo la estrategia _greedy_, sin exploración, el algoritmo converge rápido, pero en el largo plazo la recompensa esperada es menor que en el caso donde permitimos la exploración. Para $\varepsilon = 0.1$ observamos que su convergencia es más rápida que el caso de $\varepsilon = 0.01$, pero pareciera ser que en este último caso se podrá llegar a una recompensa esperada mayor. En fin, en ambos casos con exploración, tomando a veces acciones que no se veían prometedoras, se mejoró la recompensa en el largo plazo. Algo similar ocurre con la cantidad de veces que el agente (MAB) toma la acción óptima. En el caso sin exploración, el agente se queda estancado en un óptimo local tomando sólo ~1/3 de las veces la mejor acción. En los otros casos, con exploración el agente es capaz de encontrar la acción óptima un mayor porcentaje de las veces. Esto es porque a medida que obtenemos más información de las otras acciones, podemos elegir la que es realmente mejor en el largo plazo.

## ¿Función de valor no estacionaria?

En la simulación de la sección previa, asumimos que la función de valor para cada acción $q_*(a)$ se mantiene estacionaria a lo largo del tiempo. Sin embargo, la mayoría de los problemas en aprendizaje por refuerzo tienen una función de valor no estacionaria. Por completitud, simplifiquemos la ecuación para estimar $Q_t(a)$. Para el análisis, consideraremos una sóla acción:

Tenemos:

$$Q_n = \frac{R_1 + R_2 + \ldots + R_{n-1}}{n - 1}$$

Luego:

$$
\begin{align}
    Q_{n + 1} & = \frac{1}{n} \sum_{i=1}^n R_i \\\\
    & = \frac{1}{n} \left(R_n + \sum_{i=1}^n R_i\right) \\\\
    & = \frac{1}{n} \left(R_n + (n - 1)\frac{1}{n-1} \sum_{i=1}^n R_i\right) \\\\
    & = \frac{1}{n} \left(R_n + (n - 1)Q_n\right) \\\\
    & = Q_n + \frac{1}{n} \left[R_n - Q_n \right] \\\\
\end{align}
$$

Podemos observar en la ecuación, que estamos siempre ponderando todas las recompensas desde el inicio hasta el paso de tiempo $n$, via $\alpha(n) = \frac{1}{n}$. El problema, es que si $q_*(a)$ cambia con el tiempo, entonces las recompensas observadas por ejemplo inicialmente, dejarían de ser relevantes ya que la función de valores para las acciones cambió. Una forma de manejar esto es por ejemplo considerar un valor de $\alpha$ constante tal que $\alpha \in [0, 1)$.

La ecuación quedaría como:

$$
\begin{align}
    Q_{n + 1} & = \alpha \sum_{i=1}^n R_i \\\\
    & = \alpha R_n + (1 - \alpha) Q_n \\\\
    & = \alpha R_n + (1 - \alpha)[\alpha R_{n - 1} + (1 - \alpha) Q_{n - 1}] \\\\
    & = \alpha R_n + (1 - \alpha)\alpha R_{n - 1} + (1 - \alpha)^2 Q_{n - 1} \\\\
    & = \alpha R_n + (1 - \alpha)\alpha R_{n - 1} + (1 - \alpha)^2\alpha R_{n - 2} \\\\
    & \ldots + (1 - \alpha)^{n - 1}\alpha R_1 + (1 - \alpha)^n Q_1 \\\\
    & = (1 - \alpha)^n Q_1 + \sum_{i = 1}^{n} \alpha (1 - \alpha) ^ {n - i} R_i
\end{align}
$$

Se puede observar que hay un desconteo que aumenta con $n$, haciendo que la contribución, por ejemplo, de acciones pasadas como $Q_1$ sean menores a medida que avanza $n$.

Consideremos ahora dos experimentos, uno en que consideramos la estrategia de muestra-media para estimar $Q_n$, es decir, considerando las contribuciones de todas las acciones pasadas pero en el caso en que $q_*(a)$ es no estacionaria. También consideraremos un experimento en el que utilizamos un peso $\alpha = 0.1$ que se mantendrá constante. El código para este experimento:

```py
def non_stationary_experiment(
    epsilon: float, steps: int, n_actions: int, alpha: float = None
) -> Tuple[np.array, np.array]:
    average_rewards = np.zeros(steps + 1)
    actions = [i for i in range(1, n_actions + 1)]

    # Create the bandit with all q_a* equal
    q_a_0 = np.random.normal(0, 1)
    q_a = [q_a_0] * n_actions

    count_optimal_action = 0
    optimal_action_perc = np.zeros(steps + 1)

    # Q_n(a): Estimated expected reward. Initial estimate is 0
    q_n = np.zeros(n_actions)
    n_actions_step = np.ones(n_actions)

    rewards = np.zeros(n_actions)


    for t in range(1, steps + 1):
        # Update the real action values
        q_a = [q + np.random.normal(0, 0.01) for q in q_a]
        optimal_action = np.argmax(q_a) + 1
        if np.random.random() <= epsilon:
            action = np.random.randint(1, n_actions + 1)
        else:
            # Take greedy action
            action = np.argmax(q_n) + 1

        rewards[action - 1] = np.random.normal(q_a[action - 1], 1)
        if alpha is None:
            q_n = q_n + (rewards - q_n) / n_actions_step
        else:
            q_n = q_n + alpha*(rewards - q_n)

        # The reward from the action taken
        average_rewards[t] = q_n[action - 1]
        count_optimal_action += int(action == optimal_action)
        optimal_action_perc[t] = count_optimal_action / t
        n_actions_step[action - 1] += 1

    return average_rewards, optimal_action_perc
```

<div align="center">

![r_mab](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/2ed125da3e01259ce109bab7d577a759b45d0ffd/mab4.png)

![opt_mab](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/2ed125da3e01259ce109bab7d577a759b45d0ffd/mab5.png)

_Fig 3: Medidas de desempeño para experimento de 10-arm bandit, caso no estacionario y estrategia de muestreo-media_

</div>

En este caso se observa que el agente tomó muchos más pasos de tiempo para lograr una recompensa esperada similar a la del caso estacionario; tomó alrededor de `10x` más de tiempo e incluso inicialmente tomó peores acciones. Para el caso _greedy_ se observa que la recompensa esperada quedó estancada en alrededor de `0.1` en los otros casos fue mayor teniendo recompensas hasta `10x` mejores que el caso sin exploración. Se observa que para $\varepsilon = 0.01$ se obtuvo una mejor recompensa esperada. En este caso es porque toma mayor cantidad de veces la acción óptima debido a que explora con menor probabilidad. Por otro lado, para el caso de $\varepsilon = 0.1$, nuevamente encontró las acciones óptimas antes que los otros casos, pues explora más frecuentemente el espacio de búsqueda de acciones.

En la figura 4 se muestran los resultados para la ponderación $\alpha = 0.1$.

<div align="center">

![r_mab](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/884f0b501bb7a1456ed54c853d75d778ce16d7f1/mab6.png)

![opt_mab](https://gist.githubusercontent.com/dpalmasan/103d61ae06cfd3e7dee7888b391c1792/raw/884f0b501bb7a1456ed54c853d75d778ce16d7f1/mab7.png)

_Fig 4: Medidas de desempeño para experimento de 10-arm bandit, caso no estacionario y estrategia de ponderación con $\alpha = 0.1$_

</div>

Para este caso se puede observar que la explotación de información es más efectiva, pues para el caso _greedy_ con $\varepsilon = 0$, observamos que puede obtener mejores casos en el largo plazo que su contraparte con la estrategia muestreo-medio. Por otro lado, la recompensa al largo plazo es mayor para todos los agentes con respecto al caso previo.

# Conclusiones

En este artículo vimos una pincelada de _Reinforcement Learning_, discutimos sobre _Multi-Armed Bandits_ y revisamos algunas simulaciones. Algunas conclusiones:

* Existe un conflicto entre exploración y explotación. Explotando la información podemos escoger la mejor acción en el momento de manera de maximizar la recompensa en el corto plazo. Sin embargo, si se explora, puede encontrarse que existen acciones que inicialmente no se ven prometedoras pero en el largo plazo obtienen una mejor recompensa.
* Cómo se estima $q_*(a)$ es crucial para el desempeño del agente que aplique esta estrategia de _greedy_. Observamos que la convergencia es lenta en el caso no estacionario, respecto del estacionario y además que dependiendo de la estimación la explotación puede dar buenos resultados o converger en un óptimo local.

Algunas aplicaciones de MAB:

1. Sistemas de recomendación: MAB se pueden utilizar para hacer recomendaciones a los usuarios basándose en sus preferencias pasadas.
2. Anuncios Online: Optimización de anuncios a mostrar a los usuarios, eligiendo los más relevantes para cada usuario.
3. Administración de portafolios: MAB pueden usarse para optimizar la distribución de activos en un portafolio, eligiendo las inversiones más rentables.
4. Distribución de recursos: MAB pueden usarse para optimizar la distribución de recursos, por ejemplo recursos computacionales o ancho de banda, entre diferentes tareas/usuarios.
5. Precios dinámicos: MAB pueden utilizarse para optimizar el precio de productos o servicios en tiempo real, basándose en la demanda y otros factores.

Existen otras aplicaciones. Yo personalmente, lo he visto en dos casos de uso de los que menciono: anuncios y distribución de recursos computacionales.

Espero lector, que te haya gustado el artículo. Un saludo...
