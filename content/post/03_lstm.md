---
layout:     post 
title:      "Long Short-Term Memory (LSTM)"
subtitle:   "Some concepts and codings about LSTM"
date:       2023-10-11T10:03:00+08:00
author:     "Ka Ian"
draft:      true
image:      "img/coding_bg.jpg"
tags:
  - "deep learning"
  - "keras"
  - "coding"
  - "python"
---

## Long Short-Term Memory (LSTM) Architecture

LSTM uses gates to deal with calcualtions. It is used with **_time series_** data, but tabular is not time series data.

* Forget Gate: Will forget the useless information (no longer needed)

* Learn Gate: Combine the Event (= current input) and STM together, so the recently learnt necessary information (from STM) can apply to the current input

* Remember Gate: Have not forget the LTM information, STM and Event are combined in remember gate which work as updated LTM

* Use Gate: Collect the information from all three gates to do the prediction, use LTM, STM and Event to predict the output of the current event which works as an updated STM.

## Other information

* **_Tabular data is not time series data, so it is needed to reshape each row data to 2-dimensional matric (representing two time step) and put into LSTM_**

* number of input node is not needed to be the equal as number of features (can be as many as you want / **_as many as optimal_**)

* Should set the input shape argument of these layers of the input layer to be equal to the data that is using

* Set the input shape according to what the reshaped data shape is

* Complex function should probably need a LSTM stacking (= one LSTM on top of another)

## Coding - LSTM

### 1. Load the Necessary Library
```
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, LSTM
```

### 2. Load the Dataset
```
df = loadtxt('dataset.csv', delimiter=',') # type of df: numpy.ndarray
```

### 3. Split the Dataset into the Features and Label
```
x = df[:, 0:8]
y = df[:, 8]
```

### 4. Reshape the data to Simulate Two Time Steps
```
print(f'x.shape: {x.shape}') # (num_row, num_col)
x = x.reshape(x.shape[0], x.shape[1] // 2, 2)
```

### 5. Define the Neural Network Architecture
```
model = Sequential()
model.add(
    LSTM(
        1,
        activation = 'relu',
        input_shape = (x.shape[1], 2),
        dropout = 0.4,
        return_sequences = True, # allow for LSTM stacking, allow another LSTM to be stacked on top of this one
    )
)
model.add(BatchNormalization())
model.add(LSTM(1))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
```

### 6. Compile the Model
```
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'],
)
```

### 7. Fit the Model

* epochs: one complete pass through the entire training dataset

* batch_size: the number of samples that used for training each time
```
model.fit(x, y, epochs=20, batch_size=8)
```

### 8. Evaluate the Model
```
_, accuracy = model.evaluate(x, y)
print('accuracy: %.2f' % (accuracy * 100))
```