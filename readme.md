# Indice
- [Funcion de Costo](#funcion-de-costo)
  * [L2-Norm](#l2-norm)
  * [Cross Entropy](#cross-entropy)
- [Gradiante Descendente](#gradiante-descendente)
  * [Gradiante Descendente Estocastico](#gradiante-descendente-estocastico)
- [Backpropagatation](#backpropagatation)
- [Regresin Lineal](#regresin-lineal)
- [SVM](#svm)
- [Random Forest](#random-forest)
- [Deep Learning y Redes Neuronales](#deep-learning-y-redes-neuronales)
- [Funcion de Activacion](#funcion-de-activacion)
  * [Umbral](#umbral)
  * [Sigmoid](#sigmoid)
  * [ReLU (Rectificador Unitario)](#relu--rectificador-unitario-)
  * [Hyper Tangent](#hyper-tangent)
- [Redes Neuronales Artificiales](#redes-neuronales-artificiales)
- [Redes Neuronales Convolucionales](#redes-neuronales-convolucionales)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Funcion de Costo

## L2-Norm

## Cross Entropy
<img src="images/1.png"><br />
<img src="images/2.png"><br />

# Gradiante Descendente
se necesita derivar para saber que inclunacion tiene la pendiente, tenemos que tratarr de llegar al punto minimo.

## Gradiante Descendente Estocastico
Se le da un poco de aleatoriedad para mejorar la optimizacion. Tiende a buscar el minimo global

# Backpropagatation
Propagacion hacia atras, vuelve hacia atras en las redes neuronales para correr nuevamente con los nuevos pesos

# Regresin Lineal

# SVM

# Random Forest

# Deep Learning y Redes Neuronales

# sklern

Normalizacion de datos, el valor mas pequeño lo transformamos en 0 y el mas grande en 1
```py
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
```

# Funcion de Activacion

## Umbral
<img src="images/3.png" />

## Sigmoid 
Devuelve un porcentaje <br />
<img src="images/4.png" />

## ReLU (Rectificador Unitario)
Transforma todo lo negativo a cero <br />
<img src="images/5.png" />

## Hyper Tangent
Tangente HIperbolica <br />
<img src="images/6.png" />

#  Redes Neuronales Artificiales
Utilizaremos Relu para que no le de importancia a los valores chicos <br />
Una recomendacion es elegir la media entre la capa entrada y salida para elegir los nodos<br />
El ejemplo es con 11 valores de entraada y 1 de salida
```py
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

###################
#   Contruccion   #
###################

# input_dim es la dimension de entrada, siempre lo colocamos en la primera capa
# units es el numero de nodos
# kernel_initializer con que distribucion inicializamos los datos
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  
                     activation = "relu", input_dim = 11))

# Solo tengo 1 salida, units = 1
# Utilizamos sigmoid para que me devuelva un porcentaje
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

###################
#   Compilacion   #
###################

# optimizer: Conjunto optimo de pesos en la nn (gd, gde, adam)
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#####################
#   Entrenamiento   #
#####################

# batch_size: procesar y corregir
classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)

#####################
#   Prediccion   #
#####################

y_pred  = classifier.predict(X_test)

#####################
#   Evaluacion   #
#####################

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Adaptamos el clasificador para que tome la evaluacion
def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
  classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
  classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1, verbose = 1)
mean = accuracies.mean()
variance = accuracies.std()

```

# Recurrent Neural Network
Aprendizaje Supervisado<br />
Se utiliza mucho paara speech, traduccion del lenguaje, para descubrir el contenido de una imagen<br />
Poder entender una secuencia, si digo 'I am a beautiful boy' entender como hacer la traduccion en secuencia (NO PALABRA POR PALABRA), si yo cambio una sola palabra 'I am a bautiful girl', cambian otras palabras en otros idiomas <br />
Podemos devolver una respuesta, o volver otra vez al nodo azul<br />
<img src="images/7.png" /><br /><br /><br />
En este caso, en el ultimo step vamos a tener todas las palabras de los anteriores steps<br />
<img src="images/8.png" /><br />

## Ejemplos
Nos puede devolver muchos detalles de una imagen<br />
<img src="images/9.png" /><br /><br />

Analisis de sentimientos, en base a un mensaje vemos el sentimiento de la persona<br />
<img src="images/10.png" /><br /><br />

Traduccion, no analiza palabras sueltas, traduccion en conjunto<br />
<img src="images/11.png" /><br /><br />

Traduccion, o subtitulos a las peliculas<br />
<img src="images/12.png" /><br /><br />

## Vanishing Gradient
Cuando propagamos un error hacia atras tenemos que mandarlo a todas las otras faces<br />
Si vamos multiplicando los valores del gradiante descendente para abajo vamos a seguir teniendo valores mas pequeños, tendiendo a cero, se va desvaneciendo<br />
<img src="images/13.png" /><br />

### Soluciones
- Exploding Gradient: propagacion hacia atras que se vaya truncando
- Vashing Gradient: corregir incializacion pesos, redes echo state, **LSTM** (Redes neuronales con corto plazo de memoria)


## LSTM (Long-Short Term Memory)
Memoria a corto plazo, la siguiente imagen es lo que sucede en cada nodo de la capa oculta<br />
<img src="images/14.png" /><br />
la **C** representa la memoria (fase memoria)<br />
la **h** representa los vaores de entrada y salida de la nn (fase prediccion)<br />
Van a ser todos vectores, por haber aplanado la nn<br />
<img src="images/15.png" /><br />
