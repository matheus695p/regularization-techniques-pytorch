![Build Status](https://www.repostatus.org/badges/latest/active.svg)

# Different types of neural network regularization techniques in pytorch

Este repo lo haré con el objetivo de tener implementado, varios métodos de
regularización, en la construcción de redes neuronales en pytorch


Voy a tener un ejemplo de:


* LR scheduler --> torch.optim.lr_scheduler o en src/lr_scheduler.py
* Early stopping --> src/early_stopping.py
* Dropout --> torch.nn dropout 
* Batchnormalization --> torch.nn BatchNorm1d
* L1 regularization
* L2 Regularization

Esto para hacerme más fácil la pega al hora de entrenar
redes en pytorch. Haciendo más modulor mi código !

# Optimizer

Antes de empezar, con los regularizadores, analytics in diamag saco hace re poco [enero 2021] un super buen post sobre los optimizadores implementados en pytorch


Yo últimamente me he decaído por el Adamax, me ha resultado bastante bueno comparado con el que siempre usaba que era Adam


<p align="center">
  <img src="./images/optimizers.png">
</p>



* https://analyticsindiamag.com/ultimate-guide-to-pytorch-optimizers/


# LR Scheduler 

Mientras entranas redes neuronales muy grandes/profundas, estás tienen más facilidad de sobreajustarse. Esto se convierte en un problema durisimo cuando el conjunto de datos es pequeño y simple. podemos saber esto fácilmente cuando, durante el entrenamiento, la pérdida de validación y la pérdida de entrenamiento comienzan a divergir gradualmente (curva de validación sube, entrenamiento sigue bajando).

Esto significa que el modelo está empezando a sobreajustarse (memorizarse los datos). Además, el uso de una tasa de aprendizaje única y de alto valor puede hacer que el modelo pierda por completo los óptimos locales durante las últimas fases del entrenamiento. Durante las últimas fases, los parámetros deben actualizarse gradualmente, a diferencia de las fases iniciales de entrenamiento.


Entrenar una red neuronal grande mientras se usa una única tasa de aprendizaje estática es realmente peluo [los entrenamientos tiender a ser extramadamente ruidosos]. Aquí es donde ayuda tener un scheduler de tasas de aprendizaje, donde podemos disminuir gradualmente el valor de la tasa de aprendizaje de forma dinámica durante el entrenamiento, cumpliendo el criterio de ciertas condiciones. Hay muchas maneras de hacer esto torch.optim.lr_scheduler tiene muchas. Pero el método más utilizado es cuando la pérdida de validación no mejora durante algunas épocas, lo que se llama el ReduceLrOnPlateu.


Digamos que observamos que la pérdida de validación no ha disminuido durante **alpha** épocas consecutivas. Entonces hay una gran probabilidad de que el modelo comience a estancarse y no avanzar hacía otros mínimos. En ese caso, podemos comenzar a disminuir la tasa de aprendizaje, digamos, en un factor de **teta**. Podemos continuar con esto durante un cierto número de épocas. Cuando estamos seguros de que la tasa de aprendizaje es tan baja que el modelo no aprenderá nada, entonces podemos detener el entrenamiento. Esto es clasiquisimo en problemas de optimización, estás llegando al minimo, muevete mas lento para no alejarte de esa zona (suena raro pero es asi XD, como la vida misma diria un amigo). 


<p align="center">
  <img src="./images/schedulers.png">
</p>


<p align="center">
  <img src="./images/schedulers1.png">
</p>


Pytorch esto lo tiene bastante avanzado, donde tiene implementado los distintos reductores de tasa de aprendize en su modulo de:

**--> torch.optim.lr_scheduler:**


Acá todos los schedulers implementados a la fecha en pytorch

* torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=True)
* torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=True)
* torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=True)
* torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)
* torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False)
* torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                             threshold=0.0001, threshold_mode='rel',
                                             cooldown=0, min_lr=0, eps=1e-08,
                                             verbose=True)
* torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None,
                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                    cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1, verbose=False)
* torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None,
                                      pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
                                      base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                      final_div_factor=10000.0, three_phase=False,
                                      last_epoch=-1, verbose=False)
* torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1,
                                                       verbose=False)


Según [7] la útilización de schedulers ciclicos permité una mejor exploración de la función de costos, aumentando la probabilidad de descubrir mejores mínimos gloables, sin embargo, desde mis malas prácticas con tensorflow y keras, ReduceLROnPlateu, nos permité llegar más rápido a un mínimo local y hacer la convergencia más rápido, dado que soy un milenial, probaré esta implementación primero, pero dejo arriba como definiar cada uno de estas torch classes de optimizadores.


# Early Stopping

Un enfoque para encontrar un buen modelo de deep learning es tratar el número de épocas de entrenamiento como un hiperparámetro y entrenar el modelo varias veces con valores diferentes, luego seleccionar el número de épocas que dan como resultado el mejor rendimiento en el  conjunto de datos de test. Este es el enfoque más **gil**, dado que se requiere entrenar y descartar múltiples modelos por mucho rato. Esto es computacionalmente ineficiente lleva mucho tiempo y paja, especialmente para modelos grandes.

El concepto de early stopping lo dice todo xdd, para el entrenamiento cuando tu loss empiece a aumentar o guarda un checkpoint con el modelo que tenga mejor validación [ver imagen], después usa ese modelo para hacer la predicción en test.


<p align="center">
  <img src="./images/early stopping.png">
</p>


Una de las excusas que me ponía para no usar pytorch, era siempre esta, wn me da paja construir el early stopping por mi cuenta, tensorflow lo tiene listo, creo que esto era muy mala excusa, dado que no era tan complejo y las ventajas son caleta más.

```zh
src/early_stopping.py
```

# Dropout


# Batch Normalization





## Instalar las librerías necesarias

```sh
$ git clone https://github.com/matheus695p/pytorch-regularization.git
$ cd pytorch-regularization
$ pip install -r requirements.txt
```

tree del proyecto

```sh

│   .gitignore
│   README.md
│
├───codes
│   ├───galaxies
│   │       nn_galaxies.py
│   │       preprocessing_galaxies.py
│   │
│   └───wine
│           main_wine.py
│
├───data
│       galaxies.csv
│       glaxies_featured.csv
│       winequality-red.csv
│
├───images
│       pairplot.png
│       targets_pairplot.png
│
├───models
│       checkpoint.pt
│
└───src
        datasets.py
        early_stopping.py
        galaxiesConfig.py
        lr_scheduler.py
        metrics.py
        nn.py
        preprocessing_module.py
        visualizations.py
        wineConfig.py

```


### Documentación

Documentación de los modulos con sphinx 

```sh
build/html/index.html
```




# Bibliografía

De donde me base para hacer los códigos: [sin copiar]


[1] https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

[2] https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html

[3] https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

[4] https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler

[5] https://medium.com/@lipeng2/cyclical-learning-rates-for-training-neural-networks-4de755927d46

[6] https://www.jeremyjordan.me/nn-learning-rate/

[7] https://arxiv.org/abs/1702.04283
