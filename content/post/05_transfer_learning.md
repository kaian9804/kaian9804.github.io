---
layout:     post 
title:      "CNN + Transfer Learning"
subtitle:   "Some concepts and codings about CNN + Transfer Learning"
date:       2023-10-12T00:42:00+08:00
author:     "Ka Ian"
draft:      true
image:      "img/coding_bg.jpg"
tags:
  - "deep learning"
  - "keras"
  - "coding"
  - "python"
---

## The Working Concepts of Transfer Learning

* Use previous trained leverage knowledge (e.g. features, weights) to train new models

* Tackle problems like having less data for the newer task

* The **_more related_** the tasks is, the **_better_** the metrics of the hybrid model

* Generally, inputs are fed to the base model

* **_Usually omit the last layer_** when the base model features / weights are **_transferred_**

* The embeddings from the penultimate layer of the base model are then **_transferred to another model_** which extracts different parts according to its needs

## CNN using Transfer Learning for Tabular Data (input given to VGG16)

* Might need to reshape data

* Base model "include_top" argument is set to **_False_**

* "Trainable" attribute is set to **_False`_**

* The embeddings from the base models penultimate layer are then given as input to the fully-connected output Dense layer

## Other information

* **_Good_** to solve the problem about less data (**_Highly Recommended_**)

* While having less data, it is **_better to use simply model_** (= without complex architecture)

* Reshape data to simulate images (reshaping a 1-dim to 2-dims metrics)

* The reshaped data is fed to the transfer learning model

* Using a model that trained with **_similar models_** works **_better_** for transfer learning

## Coding - CNN + Transfer Learning

### 1. Load Library
```
import cv2
import numpy as np
import pandas as pd
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
```

### 2. Load Dataset
```
df = loadtxt('dataset.csv', delimiter=',')
```

### 3. Split into input and output variables
```
X = df[:, 0:8]
y = df[:, 8]
```

### 4. Resize Images RGB
```
def resize_img_rgb(X):
  tmp = np.array(X)
  tmp = tmp.reshape(-1, 2, 4) # 3 channels (R, G, B)
  X = pd.DataFrame(sum(map(list, tmp), []))

  tmp = []
  for i, g in X.groupby(np.arange(len(X)) // 2):
    tmp.append(g)
  
  tmp = np.array([i.to_numpy() for i in tmp])
  X = tmp.reshape(768, 2, 4, 1)

  X_rgb = []
  for i in X:
    i = cv2.resize(i, dsize=(64, 32), interpolation=cv2.INTER_CUBIC)
    X_rgb.append(np.stack((i, i, i), axis=2))

  return np.array(X_rgb)
```

```
X = resize_img_rgb(X) # resize X
```

### 5. Define the Keras Model
```
# base model
base_model = VGG16(
    weights = 'imagenet',
    include_top = False,
    input_shape = (32, 64, 3), # height 32, weight 64, 3 channels
)
base_model.trainable = False
```

```
# newer model
model = Sequential()
model.add(base_model)
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
```

### 6. Compile the Keras Model
```
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)
```

### 7. Fit the Keras Model on the Dataset
```
model.fit(X, y, epochs=20, batch_size=8)
```

### 8. Evaluate the Keras Model
```
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))
```