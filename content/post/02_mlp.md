---
layout:     post 
title:      "Coding for Multi-layer Perceptron (MLP)"
subtitle:   "This is only the coding introduction for MLP. No theory will be illustrated here."
date:       2023-10-10T00:42:00+08:00
author:     "Ka Ian"
draft:      true
image:      "img/coding_bg.jpg"
tags:
  - "deep learning"
  - "keras"
  - "coding"
  - "python"
---

## 1. Loading the Library
```
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense
```

## 2. Loading Dataset

```
df = loadtxt("dataset.csv", delimiter=',')
```

## 3. Spliting Features and label
For example, the data includes 5 features and 1 label.

```
x = df[:, 0:5]
y = df[:, 5]
```

## 4. Define Neural Network Architecture

```
model = Sequential()
model.add(Dense(12, input_dim=5, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
```

## 5. Compile the Model

```
model.compile(
    loss = 'binary_crossentropy', # this is a binary problem
    optimizer = 'adam',
    metrics = ['accuracy']
)
```

## 6. Fit the Model

> epochs means how many times it runs

```
model.fit(x, y, epochs=10, batch_size=100)
```

## 7. Evaluate the model (e.g. finding the accuracy)

```
_, accuracy = model.evaluate(x, y)
print(f"accuracy: %.2f" % (accuracy * 100))
```