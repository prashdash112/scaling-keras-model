from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

housing=fetch_california_housing()

df=pd.DataFrame(data=housing.data,columns=housing.feature_names)
target=pd.DataFrame(data=housing.target,columns=['target'])
df=pd.concat([df,target],sort=True,axis=1)

X_train_full,X_test, y_train_full,y_test=train_test_split(
    housing.data, housing.target)

X_train,X_valid, y_train,y_valid=train_test_split(
    X_train_full,y_train_full)

scaler1=StandardScaler()
X_train=scaler1.fit_transform(X_train)
X_valid=scaler1.transform(X_valid)
x_test=scaler1.transform(X_test)

#Functional API model

input_=tf.keras.layers.Input(shape=X_train.shape[1:])
hidden1=tf.keras.layers.Dense(30,activation='relu')(input_)
hidden2=tf.keras.layers.Dense(10,activation='relu')(hidden1)
concat=tf.keras.layers.concatenate(inputs=[input_ , hidden2])
output=tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_], outputs=[output])

model.compile(loss='mse',optimizer=tf.keras.optimizers.SGD(lr=1e-3))
model.fit(X_train,y_train,epochs=10,validation_data=(X_valid,y_valid))

# Saving the model
model.save('mlp_reg_model.h5')

#Loading the model
model = tf.keras.models.load_model("mlp_reg_model.h5")

#For a deep learning model, callbacks are very important

#EarlyStopping for reducing the waste of time and resources & checkpoint for saving the model after every n epochs so system crash won't affect our model

checkpoint_cb1=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("mlp_reg_model.h5")

model.fit(X_train,y_train,epochs=40,validation_data=(X_valid,y_valid),callbacks=[checkpoint_cb1,checkpoint_cb])

model.save('mlp_reg_model.h5')

log_directory = 'logs\\fit'
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_directory)
history = model.fit(X_train, y_train, epochs=5,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])
print(os.listdir(path=r'file_path'))
                    
# After reaching the location where your .py or .ipynb file is saved in cmd, run this command 'tensorboard --logdir logs\fit' and browse http://localhost:6006/ . 
