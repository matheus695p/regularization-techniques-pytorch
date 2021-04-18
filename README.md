![Build Status](https://www.repostatus.org/badges/latest/active.svg)

# Types of neural network regularization in pytorch

Este repo lo haré con el objetivo de tener implementado, varios métodos de
regularización, con el fin de prevenir overfitting en los
entrenamientos de distintos tipos de redes neuronales.


Voy a tener un ejemplo de:


* LR scheduler
* Early stopping
* L1 regularization
* L2 Regularization
* Dropout
* Batchnormalization


Esto para hacerme más fácil la pega al hora de entrenar
redes en pytorch. Haciendo más modulor mi código !


# LR Scheduler 

Mientras entranai redes neuronales muy grandes/profundas (muchos parámetros),
el tiene más facilidad de sobreajustarse. Esto se convierte en un problema
durisimo cuando el conjunto de datos es pequeño y simple.

Podemos saber esto fácilmente cuando, durante el entrenamiento,
la pérdida de validación y la pérdida de entrenamiento
comienzan a divergir gradualmente (curva de validación sube, entrenamiento
sigue bajando).

Esto significa que el modelo está empezando a sobreajustarse (memorizarse los datos).
Además, el uso de una tasa de aprendizaje única y de alto valor puede hacer
que el modelo pierda por completo los óptimos
locales durante las últimas fases del entrenamiento. Durante las últimas fases,
los parámetros deben actualizarse gradualmente, a diferencia de las fases
iniciales de entrenamiento.


Entrenar una red neuronal grande mientras se usa una única tasa de
aprendizaje estática es realmente peluo. Aquí es donde ayuda para el dueño del código
de tasas de aprendizaje. Usando el programador de tasa de aprendizaje, podemos
disminuir gradualmente el valor de la tasa de aprendizaje de forma dinámica
durante el entrenamiento. Hay muchas maneras de hacer esto.
Pero el método más utilizado es cuando la pérdida de validación no mejora durante algunas épocas.
Digamos que observamos que la pérdida de validación no ha disminuido durante
5 épocas consecutivas. Entonces hay una gran probabilidad de que el modelo
comience a sobreajustarse. En ese caso, podemos comenzar a disminuir la
tasa de aprendizaje, digamos, en un factor de 0,5.
Podemos continuar con esto durante un cierto número de épocas.
Cuando estamos seguros de que la tasa de aprendizaje es tan baja que el
modelo no aprenderá nada, entonces podemos detener el entrenamiento. Esto es clasiquisimo
en problemas de optimización, estás llegando al minimo, muevete mas lento para no alejarte
de esa zona (suena raro pero es asi XD, como la vida misma diria un bro). 





## Instalar las librerías necesarias

```sh
$ git clone https://github.com/matheus695p/pytorch-regularization.git
$ cd pytorch-regularization
$ pip install -r requirements.txt
```

tree del proyecto

```sh

```


### Documentación

Documentación de los modulos con sphinx 

```sh
build/html/index.html
```




# Bibliografía


[1] https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

[2]

[3]

[4]