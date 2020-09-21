import tensorflow as tf

model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(30,activation='relu',input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mean_squared_error,optimizer='sgd'
             )
history=model.fit(X_train,y_train,epochs=60,
                  validation_data=(X_valid,y_valid))

mse_test = model.evaluate(X_test, y_test)
print(mse_test)

X_new = X_test[:3] # pretend these are new instances
print(X_new)
y_pred = model.predict(X_new)
print(y_pred)
