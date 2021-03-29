# Tensorflow Keras 模型儲存
Tensorflow 有三種模型儲存方式。第一種是存成 checkpoint 檔(.ckpt)，使用時機是訓練過程中欲保存目前 session 狀態。第二種是存成 pb 檔(.pb)，如果模型架構已確定或是訓練已結束，準備匯出應用時，可以直接存成 pb 檔。第三種是 Keras (目前已合併到 TF2.0) 的 `save()` 直接存成 HDF5 檔(.h5)，HDF 是設計用來儲存和組織大量資料的一組檔案格式，其內容包含了模型架構與權重。本篇文章透過波士頓房價預測資料集，訓練一個 DNN 模型並示範如何匯出與載入 `.pb` 和 `.h5` 模型檔。


## 建構模型
4) 建構模型
使用 Tensorflow Keras API 建立 Sequential Model

```py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(16, activation='relu',
                       input_shape=(x_train.shape[1],)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='linear'))

model.summary()
```

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 16)                224       
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 136       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 9         
=================================================================
Total params: 369
Trainable params: 369
Non-trainable params: 0
_________________________________________________________________
```

```py
# 編譯模型用以訓練 (設定 optimizer, loss function, metrics, 等等)
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.05))

# 設定訓練參數
batch_size = 32  # 每次看 batch_size 筆的資料就更新權重
epochs = 50      # 一個 epoch 會看過一次所有的資料

# 訓練模型
model_history = model.fit(x=x_train, y=y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_valid, y_valid),
                          shuffle=True)
```

## 儲存模型
#### HDF5 檔
```py
model.save('./weights/boston_model.h5') 
```

#### pb 檔
```py
model.save('./checkpoints/boston_model.pb')
```

## 載入模型
#### HDF5 檔
```py
myModel = keras.models.load_model('./weights/boston_model.h5')
```

#### pb 檔
```py
myModel = keras.models.load_model('./checkpoints/boston_model.pb')
```