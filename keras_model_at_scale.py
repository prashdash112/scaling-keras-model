from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import pprint

housing= fetch_california_housing()

df=pd.DataFrame(data=housing.data,columns=housing.feature_names)
target=pd.DataFrame(data=housing.target,columns=['target'])
df=pd.concat([df,target],sort=True,axis=1)

X_train_full,X_test, y_train_full,y_test=train_test_split(
    housing.data, housing.target)

X_train,X_valid, y_train,y_valid=train_test_split(
    X_train_full,y_train_full)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_valid=scaler.transform(X_valid)

# Function to pass through keras regressor wrapper
def build_model(n_hidden=1,n_neurons=30,input_shape=[8],learning_rate=1e-3):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(units=n_hidden,activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='mse',optimizer=optimizer)
return model
    
keras_reg=tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)
checkpoint_cb1=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

keras_reg.fit(X_train,y_train,epochs=50,
              validation_data=(X_valid,y_valid),
              callbacks=[checkpoint_cb1])

param_distribs={
    'n_hidden':[0,1,2,3],
    'n_neurons':np.arange(1,100),
    'learning_rate': reciprocal(1e-4,1e-2)
}

# RandomsearchCV object is used to test various parameters and to tune hyperparameters of the production model for best scores

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
                  
print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)

model = rnd_search_cv.best_estimator_.model
model.save('final_production.h5')
pprint.pprint(model.get_config())
   
