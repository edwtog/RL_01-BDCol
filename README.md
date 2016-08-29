# RL_01-BDCol

# Reinforcement Learning

Aprendizaje por refuerzo (RL) trata de contestar la siguiente pregunta: ¿Cómo actuar de forma óptima en un ambiente desconocido?
Esta pregunta describe la intereacción de un agente, que es quien actúa, sobre un ambiente que cambia o evoluciona en la medida en la que el agente ejecuta una serie de acciones. Esto con el objetivo de optimizar alguna medida de satisfacción (recompensa) que el agente recibe y que se logra buscando configuraciones especificas del ambiente (estados).
El marco matemático usado para describir esta interacción agente-ambiente son los Procesos de Decisión de Markov (MDP)

## Elementos MDP


* Estados $s, s' \in \mathcal{S}$, el conjunto de estados representa todas las posibles configuraciones de un ambiente (Ej. Todas las configuraciones posibles y validas de las fichas de ajedrez sobre el tablero).
* Acciones $a \in \mathcal{A}$, el conjunto de acciones validas que el agente puede tomar en cada estado.
* Probabilidades de transición $p(s'\mid s,a)$, que describen la dinámica del ambiente.
* Recompensas $r = \mathcal{R} (s,a,s')$, que es una medida del desempeño en la transición de un estado a otro después de ejecutar una acción.

## Definiciones

La característica fundamental del MDP es que no poseen memoria, la dinámica del ambiente depende solamente del estado y la acción actual. (Es posible trabajar con procesos no Markovianos aplicando algún conjunto de restricciones o métodos que permitan la construcción una aproximación markoviana)
$$
    \begin{align}
      p(s'\mid s,a) = p(s'\mid s_0, s_1, \ldots , s, a)
    \end{align}
$$


Hay una política $\pi$ que genera un mapeo desde el conjunto de estados al conjunto de acciones. Esta política es usada por el agente para decidir que acción $a$ tomar cuando se encuentra en el estado $s$
$$
    \begin{align}
     \pi: \mathcal{S} \rightarrow \mathcal{A}
    \end{align}
$$

Se construye la función de retorno como la sumatoria de las recompensas acumuladas (con descuento) que el agente recibirá desde un estado dado.
$$
    \begin{align}
     R_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
    \end{align}
$$
El factor de descuento $\gamma$ se utiliza para hacer converger la sumatoria en casos de horizonte infinito (No episódico)


Debido a las probabilidades de transición es necesario calcular el valor esperado del retorno, lo que da origen a la función de valor:
$$
    \begin{align}
     Q^{\pi} (s,a) = \mathbf{E}\{R_t \mid s_t=s, a_t=a\}
    \end{align}
$$

## Objetivo

El problema del RL consisten en encontrar la política optima $\pi^{\ast}$ que maximiza $Q^{\pi} (s,a)$ entre todas las políticas posibles para todo $s \in \mathcal{S}$

En otras palabras el objetivo en cualquier problema de aprendizaje por refuerzo es encontrar la secuencia de acciones que maximice la suma de las recompensas a lo largo de la secuencia. Esto se reduce al objetivo de aproximar la función $Q$  que predecirá la suma de recompensas futuras (descontadas). Esta función depende de la tupla $(s,a)$ lo que nos permitirá seleccionar la $a$ que garantiza el mayor valor para $Q$

$$
    \begin{align}
     Q(s_t,a_t) \approx \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
    \end{align}
$$


\begin{align*}
\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} &= r_{t+1} + \sum_{k=1}^{\infty} \gamma^k r_{t+k+1} \\ 
&= r_{t+1} + \gamma\sum_{k=0}^{\infty} \gamma^k r_{t+k+2} \\
&\approx r_{t+1} + \gamma Q(s_{t+1},a_{t+1})
\end{align*}

este resultado se conoce como Temporal difference error (TD) y será utilizado para la actualizacion de $Q$.

## Representacion tabular de la funcion de valor $Q$

Dedido a que no conocemos las probabilidades de transición, podemos remover el operador de valor esperado y simplemente tomar muestras del ambiente para actualizar $Q$. Utilizando el TD generamos la funcion de actualización:

$$
    \begin{align}
     Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \Bigl( r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t) \Bigr)
    \end{align}
$$


## Ejemplo

Usaremos un ejemplo de "juguete" para mostrar los elementos básicos en la actualización de $Q$ para el caso en el que se representa como un tabla.

El agente se encuentra confinado en una cuadricula de 10X10 y debe llegar a la celda marcada como "G"

![Image](https://github.com/edwtog/RL_01-BDCol/blob/master/gw.png)

La recompensa $r$ es 1 al llegar a la  celda objetivo y 0 en otro caso.

Definimos la cuadricula como un arreglo de caracteres delimitado por '*', la celda objetivo se marca como 'G'

