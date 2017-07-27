library(devtools)
library(reticulate)
library(tensorflow)

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
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = FALSE, dtype = 'float32')
np <- import("numpy", convert = FALSE)
X_train = mnist$train$images
X_test = mnist$test$images
y_train = as.matrix(mnist$train$labels,1)
dim(y_train)
head(y_train)
y_test = as.matrix(mnist$test$labels,1)

config = tf$contrib$learn$RunConfig(tf_random_seed=as.integer(42)) # not shown in the config

feature_cols = tf$contrib$learn$infer_real_valued_columns_from_input(X_train)
feature_cols
dnn_clf = tf$contrib$learn$DNNClassifier(hidden_units=c(300L,100L),
                                         n_classes=10L,
                                         feature_columns=feature_cols, 
                                         config=config)

dnn_clf
dnn_clf = tf$contrib$learn$SKCompat(dnn_clf)

dnn_clf

dnn_clf$fit(X_train, y_train, batch_size=50L, steps = 1000L) #error..TT

y_pred = dnn_clf$predict(X_test)

sklearn$metrics$accuracy_score(y_pred = y_pred$classes, y_true  = y_test)
sklearn$metrics$log_loss(y_pred = y_pred$probabilities, y_true = y_test)

### Training a DNN using plain TensorFlow¶
n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

tf$reset_default_graph()

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

neuron_layer <- function(X, n_neurons, name, activation=NULL){
  with(tf$name_scope('name') %as% scope, {
    n_inputs = dim(X)[[2]]
    stddev = 2 / sqrt(n_inputs) 
    init = tf$truncated_normal(shape(n_inputs, n_neurons), stddev=stddev)
    W = tf$Variable(init, name="kernel")
    b = tf$Variable(tf$zeros(n_neurons), name="bias")
    Z = tf$matmul(X, W) + b
    if(!is.null(activation)){
      activation(Z)
    }
    else{
      Z
    }
  })
}

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf$nn$relu)
  hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf$nn$relu)
  logits = neuron_layer(hidden2, n_outputs, name="outputs")
})

hidden1
hidden2
logits

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

xentropy
loss

learning_rate = 0.01

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$GradientDescentOptimizer(learning_rate)
  training_op = optimizer$minimize(loss) 
})

optimizer
training_op

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32))
})

correct
accuracy

n_epochs = 5
batch_size = 50
n_batches = 55000 / batch_size
n_batches

init = tf$global_variables_initializer()
saver = tf$train$Saver()

library(caret)

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      #print(dim(X_train[folds[[batch_index]],]))
      #print(length(y_train[folds[[batch_index]]]))
      
      sess$run(training_op, feed_dict=dict(X = X_train[folds[[batch_index]],],
                                           y = y_train[folds[[batch_index]]]))
    }
    #print(dim(X_train[folds[[batch_index]],]))
    #print(length(y_train[folds[[batch_index]]]))
    acc_train = accuracy$eval(feed_dict = dict(X = X_train[folds[[batch_index]],], 
                                             y = y_train[folds[[batch_index]]]))
    acc_test = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    
    cat(paste(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test, sep=' '), sep='\t', fill = T)
  }
  save_path = saver$save(sess, "./my_model_final.ckpt")
})

with(tf$Session() %as% sess, {
  saver$restore(sess, "./my_model_final.ckpt") 
  X_new_scaled = mnist$test$images[1:20,]
  Z = logits$eval(feed_dict=dict(X = X_new_scaled))
  y_pred = apply(Z, 1, which.max)
})

cbind(pred=y_pred-1, label=mnist$test$labels[1:20])

### Using dense() instead of neuron_layer()
n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

tf$reset_default_graph()

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, name="hidden1", activation=tf$nn$relu)
  hidden2 = tf$layers$dense(hidden1, n_hidden2, name="hidden2", activation=tf$nn$relu)
  logits = tf$layers$dense(hidden2, n_outputs, name="outputs")
})

hidden1
hidden2
logits

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

xentropy
loss

learning_rate = 0.01

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$GradientDescentOptimizer(learning_rate)
  training_op = optimizer$minimize(loss) 
})

optimizer
training_op

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32))
})

correct
accuracy

n_epochs = 20
batch_size = 50
n_batches = 55000 / batch_size
n_batches

init = tf$global_variables_initializer()
saver = tf$train$Saver()

library(caret)

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      #print(dim(X_train[folds[[batch_index]],]))
      #print(length(y_train[folds[[batch_index]]]))
      
      sess$run(training_op, feed_dict=dict(X = X_train[folds[[batch_index]],],
                                           y = y_train[folds[[batch_index]]]))
    }
    #print(dim(X_train[folds[[batch_index]],]))
    #print(length(y_train[folds[[batch_index]]]))
    acc_train = accuracy$eval(feed_dict = dict(X = X_train[folds[[batch_index]],], 
                                               y = y_train[folds[[batch_index]]]))
    acc_test = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    
    cat(paste(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test, sep=' '), sep='\t', fill = T)
  }
  save_path = saver$save(sess, "./my_model_final2.ckpt")
})