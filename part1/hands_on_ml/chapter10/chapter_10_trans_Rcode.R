library(devtools)
library(tensorflow.examples.tutorials.mnist)
library(reticulate)

setwd('/Users/syleeie/Downloads/tensorbreak/part1/hands_on_ml/chapter10')

tf$reset_default_graph()

np <- import("numpy")
sklearn <- import("sklearn")

Perceptron <- sklearn$linear_model$Perceptron
data(iris)
iris
head(iris)
X = iris[,c(3:4)]
y = ifelse(as.numeric(iris$Species) == 1, 1, 0)

per_clf = Perceptron(random_state=as.integer(42))

per_clf$fit(as.array(as.matrix(X)), 
            as.array(as.vector(y)))

per_clf
y_pred = per_clf$predict(matrix(c(2, 0.5), ncol=2))
y_pred



### Training an MLP with TensorFlow’s high level API
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

X_train = mnist$train$images
X_test = mnist$test$images
y_train = as.integer(mnist$train$labels)
y_test = as.integer(mnist$test$labels)

config = tf$contrib$learn$RunConfig(tf_random_seed=as.integer(42)) # not shown in the config

feature_cols = tf$contrib$learn$infer_real_valued_columns_from_input(X_train2)
feature_cols

dnn_clf = tf$contrib$learn$DNNClassifier(hidden_units=c(300,100),
                                         n_classes=10,
                                         feature_columns=feature_cols, 
                                         config=config)

dnn_clf
dnn_clf = tf$contrib$learn$SKCompat(dnn_clf)

dnn_clf

dnn_clf$fit(X_train, y_train, batch_size=50, steps = 40000)


### Training a DNN using plain TensorFlow¶
n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

tf$reset_default_graph()

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

?sqrt
sqrt(4)
sqrt(784)
np$sqrt(X$get_shape()[[1]])

sqrt(as.vector(X$get_shape()[[1]]))

neuron_layer <- function(X, n_neurons, name, activation=NULL){
  with(tf$name_scope('name') %as% scope, {
    n_inputs = 784 #X$get_shape()[[1]]
    stddev = 2 / 28 #sqrt(n_inputs)
    init = tf$truncated_normal(shape(n_inputs, n_neurons), stddev=stddev)
    W = tf$Variable(init, name="kernel")
    b = tf$Variable(tf$zeros(n_neurons), name="bias")
    Z = tf$matmul(X, W) + b
    return(ifelse(activation != NULL, activation(Z), Z))
    })
}

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf$nn$relu)
  hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf$nn$relu)
  logits = neuron_layer(hidden2, n_outputs, name="outputs")
})

