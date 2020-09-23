# INTRODUCTION

The goal is to make a production level deep learning model which can produce good results on any sample labelled data. In this case, we've used California housing data.

# Sample data
![Sample data](https://github.com/prashdash112/scaling-keras-model/blob/master/Images/sample%20data.png)


# Artificial neural networks

ANNs are MLP network used for a variety of task. Here we've used it as a regressor to predict the target for the sample data. ANNs can be implemented using several  libraries. We've used keras here for its implementation.

ANNs can be implemented by Sequential API,functional API & keras subclassing. Different model implementation gives different loss graph results.

![ANN](https://miro.medium.com/max/482/1*qcqm1_dqyuRdfEPQGWMQcA.jpeg)

Article to understand neural networks- https://medium.com/analytics-vidhya/neural-network-from-scratch-ed75e5e14cd

## Sequential API loss graph
![Loss graph for sequential api](https://github.com/prashdash112/scaling-keras-model/blob/master/Images/Sequential.png)

## Functiona API loss graph
![Loss graph for functional api](https://github.com/prashdash112/scaling-keras-model/blob/master/Images/functional.png)

## Wide and deep model loss graph
![Loss graph for Wide and deep model](https://github.com/prashdash112/scaling-keras-model/blob/master/Images/wide%20and%20deep%20.png)

## Wide and deep with auxillary output
![Loss graph for Wide and deep with aux output model](https://github.com/prashdash112/scaling-keras-model/blob/master/Images/wide%20and%20deep%20with%20aux%20output.png)

## Keras subclassing API loss graph
![Loss graph for Keras subclassing api](https://github.com/prashdash112/scaling-keras-model/blob/master/Images/keras%20subclassing%20.png)

Different models give different results for sample data(California housing in our case).

# Hyperparameter tuning

Tuning the hyperparameters of a model is very important for fruitful results. Tuning can be done manually but it's not feasible and infact a very tedious task. So here we've used RandomSearchCV - a cross validation technique which itself picks the parameters given in param_distribs{}, trains the model using different combinations of them & yeilds the best result for production level deployment.

For this to happen, we need to write our keras model as a python function so that we can fit it as an estimator to keras wrapper regressor and later on randomsearchCV can pick and put various hyper-parameters in the model to evaluate its performance.

# Callbacks

Callbacks are also a very important feature of a production model. EarlyStopping is used to stop the training process just before the model starts to overfit the data, in this way we save the time and computational resources which is principal for production environment. Modelcheckpoint is used to save a checkpoint after every n epochs so that CPU crash or cloud server error won't affect out model.

# Model summary
![Model summary](https://github.com/prashdash112/scaling-keras-model/blob/master/Images/model%20summary.png)
