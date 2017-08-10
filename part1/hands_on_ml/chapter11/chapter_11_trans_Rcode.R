#Hands on Machine learning scikit_learn_and_Tensorflow
#Training Deep Neural Nets 책 11장 (약 70% 완료)
#Transfer Learning 부분이랑 일부 에러가 있음

library(devtools)
library(reticulate)
library(tensorflow)

setwd('/Users/syleeie/Downloads/tensorbreak/part1/hands_on_ml/chapter11')

tf$reset_default_graph()

np <- import("numpy")
X_train = mnist$train$images
X_test = mnist$test$images

#Xavier and He initialization
n_inputs = 28 * 28  # MNIST
n_hidden1 = 300

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")

he_init = tf$contrib$layers$variance_scaling_initializer()

hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu,
                          kernel_initializer=he_init, name="hidden1")

he_init
hidden1

z = seq(-5, 5, length.out = 200)

leaky_relu <- function(z, alpha=0.01){
  return(pmax(alpha*z, z))
}

plot(z, leaky_relu(z, 0.05), ylim=c(-0.5,4.2), xlim=c(-5,5))
abline(h=0, col="black", lty=1)
abline(v=0, col="black", lty=1)
abline(h=4, col="black", lty=3)


tf$reset_default_graph()

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")

leaky_relu <- function(z, name=NULL){
  return(tf$maximum(0.01*z, z, name=name))
}

hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu,
                          kernel_initializer=he_init, name="hidden1")

tf$reset_default_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")


with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, name="hidden1", activation=leaky_relu)
  hidden2 = tf$layers$dense(hidden1, n_hidden2, name="hidden2", activation=leaky_relu)
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

n_epochs = 40
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
      
      sess$run(training_op, feed_dict=dict(X = X_train[folds[[batch_index]],],
                                           y = y_train[folds[[batch_index]]]))
    }
    if (epoch %% 5 == 0)
    {
      acc_train = accuracy$eval(feed_dict = dict(X = X_train[folds[[batch_index]],], 
                                                 y = y_train[folds[[batch_index]]]))
      acc_test = accuracy$eval(feed_dict = dict(X = mnist$validation$images, y = mnist$validation$labels))
      
      cat(paste(epoch, "Batch accuracy:", acc_train, "Validation accuracy:", acc_test, sep=' '), sep='\t', fill = T)
      
    } 
  }
  save_path = saver$save(sess, "./my_model_final_R.ckpt")
})

###Elu Function
elu <- function(z, alpha=1){
  return(ifelse(z<0, alpha * (exp(z) - 1), z))
}

plot(z, elu(z), xlim=c(-5,5), ylim=c(-2.2,3.2))
abline(h=0, col="black", lty=1)

abline(v=0, col="black", lty=1)
abline(h=-1, col="black", lty=3)
abline(h=3, col="black", lty=3)


selu <- function(z,  scale=1.0507009873554804934193349852946,
                 alpha=1.6732632423543772848170429916717){
  return(scale * elu(z, alpha))
}

plot(z, selu(z), xlim=c(-5,5), ylim=c(-2.2,3.2))
abline(h=0, col="black", lty=1)

abline(v=0, col="black", lty=1)
abline(h=-2, col="black", lty=3)
abline(h=3, col="black", lty=3)

selu <- function(z,  scale=1.0507009873554804934193349852946,
                 alpha=1.6732632423543772848170429916717){
  return(scale * tf$where(z >= 0.0, z, alpha * tf$nn$elu(z)))
}

tf$reset_default_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, name="hidden1", activation=selu)
  hidden2 = tf$layers$dense(hidden1, n_hidden2, name="hidden2", activation=selu)
  logits = tf$layers$dense(hidden2, n_outputs, name="outputs")
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

learning_rate = 0.01

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$GradientDescentOptimizer(learning_rate)
  training_op = optimizer$minimize(loss) 
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32))
})

n_epochs = 40
batch_size = 50
n_batches = 55000 / batch_size

init = tf$global_variables_initializer()
saver = tf$train$Saver()

means <- apply(X_train, 2, mean)
stds <- apply(X_train, 2, sd) + 1e-10

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      X_batch_scaled = (X_train[folds[[batch_index]],] - means) / stds
      sess$run(training_op, feed_dict=dict(X = X_batch_scaled,
                                           y = y_train[folds[[batch_index]]]))
    }
    if (epoch %% 5 == 0)
    {
      acc_train = accuracy$eval(feed_dict = dict(X = X_batch_scaled, 
                                                 y = y_train[folds[[batch_index]]]))
      X_val_scaled = (mnist$validation$images - means) / stds
      acc_test = accuracy$eval(feed_dict = dict(X = X_val_scaled, y = mnist$validation$labels))
      
      cat(paste(epoch, "Batch accuracy:", acc_train, "Validation accuracy:", acc_test, sep=' '), sep='\t', fill = T)
      
    } 
  }
  save_path = saver$save(sess, "./my_model_final_selu_R.ckpt")
})
#결과가 이상하게 나오네요..ㅜㅜ 

#Implementing Batch Normalization with TensorFlow
tf$reset_default_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
training = tf$placeholder_with_default(F,shape(), name="training")

hidden1 = tf$layers$dense(X, n_hidden1, name="hidden1")
bn1 = tf$layers$batch_normalization(hidden1, training=training, momentum=0.9)
bn1_act = tf$nn$elu(bn1)

hidden2 = tf$layers$dense(bn1_act, n_hidden2, name="hidden2")
bn2 = tf$layers$batch_normalization(hidden2, training=training, momentum=0.9)
bn2_act = tf$nn$elu(bn2)

logits_before_bn = tf$layers$dense(bn2_act, n_outputs, name="outputs")
logits = tf$layers$batch_normalization(logits_before_bn, training=training,
                                       momentum=0.9)

tf$reset_default_graph()

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
training = tf$placeholder_with_default(F,shape(), name="training")

functools <- import("functools")

my_batch_norm_layer = functools$partial(tf$layers$batch_normalization,
                              training=training, momentum=0.9)

hidden1 = tf$layers$dense(X, n_hidden1, name="hidden1")
bn1 = my_batch_norm_layer(hidden1)
bn1_act = tf$nn$elu(bn1)
hidden2 = tf$layers$dense(bn1_act, n_hidden2, name="hidden2")
bn2 = my_batch_norm_layer(hidden2)
bn2_act = tf$nn$elu(bn2)

bn2_act
logits_before_bn = tf$layers$dense(bn2_act, n_outputs, name="outputs")
logits = my_batch_norm_layer(logits_before_bn)


tf$reset_default_graph()

batch_norm_momentum = 0.9

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")
training = tf$placeholder_with_default(F,shape(), name="training")

with(tf$name_scope('dnn') %as% scope, {
  he_init = tf$contrib$layers$variance_scaling_initializer()
  
  my_batch_norm_layer = functools$partial(tf$layers$batch_normalization,
                                          training=training, momentum=batch_norm_momentum)
  
  my_dense_layer = functools$partial(tf$layers$dense,
                                     kernel_initializer=he_init)
  
  hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
  bn1 = tf$nn$elu(my_batch_norm_layer(hidden1))
  hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
  bn2 = tf$nn$elu(my_batch_norm_layer(hidden2))
  logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
  logits = my_batch_norm_layer(logits_before_bn)
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$GradientDescentOptimizer(learning_rate)
  training_op = optimizer$minimize(loss) 
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32))
})

init = tf$global_variables_initializer()
saver = tf$train$Saver()

n_epochs = 20
batch_size = 200
n_batches = 55000 / batch_size
n_batches

learning_rate = 0.01

extra_update_ops = tf$get_collection(tf$GraphKeys$UPDATE_OPS)

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(tuple(training_op,extra_update_ops), 
               feed_dict=dict(training = TRUE,
                              X = X_train[folds[[batch_index]],],
                              y = y_train[folds[[batch_index]]]))
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
      
  }
  save_path = saver$save(sess, "./my_model_final_R2.ckpt")
})

#Gradient clipping (기울기 제한)
tf$reset_default_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")


with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, name="hidden1")
  hidden2 = tf$layers$dense(hidden1, n_hidden2, activation=tf$nn$relu, name="hidden2")
  hidden3 = tf$layers$dense(hidden2, n_hidden3, activation=tf$nn$relu, name="hidden3")
  hidden4 = tf$layers$dense(hidden3, n_hidden4, activation=tf$nn$relu, name="hidden4")
  hidden5 = tf$layers$dense(hidden4, n_hidden5, activation=tf$nn$relu, name="hidden5")
  logits = tf$layers$dense(hidden5, n_outputs, name="outputs")
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

learning_rate = 0.01
threshold = 1.0

optimizer = tf$train$GradientDescentOptimizer(learning_rate)

grads_and_vars = optimizer$compute_gradients(loss)

capped_gvs = NULL

for(i in 1:12){
  grad = grads_and_vars[[i]][1][[1]]
  var = grads_and_vars[[i]][2][[1]]
  capped_gvs = c(capped_gvs, tuple(tf$clip_by_value(grad, -1, 1), var))
}

training_op = optimizer$apply_gradients(unlist(capped_gvs))

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})


init = tf$global_variables_initializer()
saver = tf$train$Saver()

n_epochs = 20
batch_size = 200
n_batches = 55000 / batch_size
n_batches

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
                              X = X_train[folds[[batch_index]],],
                              y = y_train[folds[[batch_index]]]))
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final_R3.ckpt")
})

#Reusing pretrained layers
tf$reset_default_graph()

saver = tf$train$import_meta_graph("./my_model_final.ckpt.meta")

for (op in tf$get_default_graph()$get_operations()){
  print(op$name)
}


X = tf$get_default_graph()$get_tensor_by_name("X:0")
y = tf$get_default_graph()$get_tensor_by_name("y:0")

accuracy = tf$get_default_graph()$get_tensor_by_name("eval/accuracy:0")

training_op = tf$get_default_graph()$get_operation_by_name("GradientDescent")

for (op in c(X, y, accuracy, training_op)){
  tf$add_to_collection("my_important_ops", op)
}

my_important_ops_tf <- tf$get_collection("my_important_ops")

X <- my_important_ops_tf[[1]]
y <- my_important_ops_tf[[2]]
accuracy <- my_important_ops_tf[[3]]
training_op <- my_important_ops_tf[[4]]

X
y
accuracy
training_op

with(tf$Session() %as% sess, {
  saver$restore(sess, "./my_model_final.ckpt") 
})

with(tf$Session() %as% sess, {
  saver$restore(sess, "./my_model_final.ckpt") 
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
        X = X_train[folds[[batch_index]],],
        y = y_train[folds[[batch_index]]]))
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final_R4.ckpt")
})


tf$reset_default_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_outputs = 10

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, name="hidden1")
  hidden2 = tf$layers$dense(hidden1, n_hidden2, activation=tf$nn$relu, name="hidden2")
  hidden3 = tf$layers$dense(hidden2, n_hidden3, activation=tf$nn$relu, name="hidden3")
  hidden4 = tf$layers$dense(hidden3, n_hidden4, activation=tf$nn$relu, name="hidden4")
  hidden5 = tf$layers$dense(hidden4, n_hidden5, activation=tf$nn$relu, name="hidden5")
  logits = tf$layers$dense(hidden5, n_outputs, name="outputs")
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

learning_rate = 0.01
threshold = 1.0

optimizer = tf$train$GradientDescentOptimizer(learning_rate)

grads_and_vars = optimizer$compute_gradients(loss)

capped_gvs = NULL

for(i in 1:12){
  grad = grads_and_vars[[i]][1][[1]]
  var = grads_and_vars[[i]][2][[1]]
  capped_gvs = c(capped_gvs, tuple(tf$clip_by_value(grad, -1, 1), var))
}

training_op = optimizer$apply_gradients(unlist(capped_gvs))

init = tf$global_variables_initializer()
saver = tf$train$Saver()

with(tf$Session() %as% sess, {
  saver$restore(sess, "./my_model_final.ckpt") 
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
        X = X_train[folds[[batch_index]],],
        y = y_train[folds[[batch_index]]]))
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final.ckpt")
})




tf$reset_default_graph()

n_hidden4 = 20 # new layer
n_outputs = 10

saver = tf$train$import_meta_graph("./my_model_final.ckpt.meta")

X = tf$get_default_graph()$get_tensor_by_name("X:0")
y = tf$get_default_graph()$get_tensor_by_name("y:0")

hidden3 = tf$get_default_graph()$get_tensor_by_name("dnn/hidden4/Relu:0")

new_hidden4 = tf$layers$dense(hidden3, n_hidden4, activation=tf$nn$relu, name="new_hidden4")
new_logits = tf$layers$dense(new_hidden4, n_outputs, name="new_outputs")

with(tf$name_scope('new_loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

with(tf$name_scope('new_eval') %as% scope, {
  correct = tf$nn$in_top_k(new_logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

with(tf$name_scope('new_train') %as% scope, {
  optimizer = tf$train$GradientDescentOptimizer(learning_rate)
  training_op = optimizer$minimize(loss)
})

init = tf$global_variables_initializer()
new_saver = tf$train$Saver()


with(tf$Session() %as% sess, {
  init$run()
  saver$restore(sess, "./my_model_final.ckpt") 
  
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
        X = X_train[folds[[batch_index]],],
        y = y_train[folds[[batch_index]]]))
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final.ckpt")
})

tf$reset_default_graph()
n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, name="hidden1")
  hidden2 = tf$layers$dense(hidden1, n_hidden2, activation=tf$nn$relu, name="hidden2")
  hidden3 = tf$layers$dense(hidden2, n_hidden3, activation=tf$nn$relu, name="hidden3")
  hidden4 = tf$layers$dense(hidden3, n_hidden4, activation=tf$nn$relu, name="hidden4") #new
  logits = tf$layers$dense(hidden4, n_outputs, name="outputs")
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$GradientDescentOptimizer(learning_rate)
  training_op = optimizer$minimize(loss)
})

reuse_vars = tf$get_collection(tf$GraphKeys$GLOBAL_VARIABLES,
                               scope="hidden[123]") # regular expression

reuse_vars

var = NULL
reuse_vars_dict = NULL

for(i in 1:6){
  var = reuse_vars[[i]]
  reuse_vars_dict = c(reuse_vars_dict, tuple(var$op$name, var))
}

reuse_vars_dict

#restore_saver = tf$train$Saver(reuse_vars_dict) # to restore layers 1-3 error...TT

#Reusing models from other frameworks
tf$reset_default_graph()
n_inputs = 2  # MNIST
n_hidden1 = 3 

original_w = list(c(1.0, 2.0, 3.0), c(4.0, 5.0, 6.0)) # Load the weights from the other framework
original_b = list(7.0, 8.0, 9.0)              # Load the biases from the other framework
original_w
original_w
original_b

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
X
hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, name="hidden1")
# [...] Build the rest of the model

# Get a handle on the assignment nodes for the hidden1 variables
graph = tf$get_default_graph()
assign_kernel = graph$get_operation_by_name("hidden1/kernel/Assign")
assign_bias = graph$get_operation_by_name("hidden1/bias/Assign")

init_kernel = assign_kernel$inputs[[1]]
init_bias = assign_bias$inputs[[1]]

init = tf$global_variables_initializer()

with(tf$Session() %as% sess, {
    sess$run(init, feed_dict=dict(init_kernel = original_w, init_bias = original_b))
    print(hidden1$eval(feed_dict=dict(X = list(10.0, 11.0))))    
}) ##error...TT


tf$reset_default_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, name="hidden1")
  hidden2 = tf$layers$dense(hidden1, n_hidden2, activation=tf$nn$relu, name="hidden2")
  hidden3 = tf$layers$dense(hidden2, n_hidden3, activation=tf$nn$relu, name="hidden3")
  hidden4 = tf$layers$dense(hidden3, n_hidden4, activation=tf$nn$relu, name="hidden4")
  logits = tf$layers$dense(hidden4, n_outputs, name="outputs")
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

with(tf$name_scope("train") %as% scope, {                                      # not shown in the book
  optimizer = tf$train$GradientDescentOptimizer(learning_rate)     # not shown
  train_vars = tf$get_collection(tf$GraphKeys$TRAINABLE_VARIABLES,
                               scope="hidden[34]|outputs")
  training_op = optimizer$minimize(loss, var_list=train_vars)
})

init = tf$global_variables_initializer()
new_saver = tf$train$Saver()

reuse_vars = tf$get_collection(tf$GraphKeys$GLOBAL_VARIABLES,
                               scope="hidden[123]") # regular expression

reuse_vars

var = NULL
reuse_vars_dict = NULL

for(i in 1:6){
  var = reuse_vars[[i]]
  reuse_vars_dict = c(reuse_vars_dict, tuple(var$op$name, var))
}

reuse_vars_dict

restore_saver = tf$train$Saver(reuse_vars_dict) # to restore layers 1-3 error...TT

init = tf$global_variables_initializer()
new_saver = tf$train$Saver()


#Learning rate scheduling
tf$reset_default_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, name="hidden1")
  hidden2 = tf$layers$dense(hidden1, n_hidden2, activation=tf$nn$relu, name="hidden2")
  logits = tf$layers$dense(hidden2, n_outputs, name="outputs")
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

with(tf$name_scope('train') %as% scope, {
  initial_learning_rate = 0.1
  decay_steps = 10000
  decay_rate = 1/10
  global_step = tf$Variable(0, trainable=FALSE, name="global_step")
  learning_rate = tf$train$exponential_decay(initial_learning_rate, global_step,
                                             decay_steps, decay_rate)
  optimizer = tf$train$MomentumOptimizer(learning_rate, momentum=0.9)
  training_op = optimizer$minimize(loss, global_step=global_step)
})

init = tf$global_variables_initializer()
saver = tf$train$Saver()

n_epochs = 5
batch_size = 50
n_batches = 55000 / batch_size
n_batches

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
        X = X_train[folds[[batch_index]],],
        y = y_train[folds[[batch_index]]]))
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final.ckpt")
})

#Avoiding Overfitting Through Regularization¶
tf$reset_default_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_outputs = 10

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, name="hidden1")
  logits = tf$layers$dense(hidden1, n_outputs, name="outputs")
})

W1 = tf$get_default_graph()$get_tensor_by_name("hidden1/kernel:0")
W2 = tf$get_default_graph()$get_tensor_by_name("outputs/kernel:0")

scale = 0.001 # l1 regularization hyperparameter

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
  base_loss = tf$reduce_mean(xentropy, name="avg_xentropy")
  reg_losses = tf$reduce_sum(tf$abs(W1)) + tf$reduce_sum(tf$abs(W2))
  loss = tf$add(base_loss, scale * reg_losses, name="loss")
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

learning_rate = 0.01

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$GradientDescentOptimizer(learning_rate)
  training_op = optimizer$minimize(loss) 
})

init = tf$global_variables_initializer()
saver = tf$train$Saver()

n_epochs = 20
batch_size = 200
n_batches = 55000 / batch_size
n_batches

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
        X = X_train[folds[[batch_index]],],
        y = y_train[folds[[batch_index]]]))
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final.ckpt")
})


tf$reset_default_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

my_dense_layer <- functools$partial(
  tf$layers$dense, activation=tf$nn$relu,
  kernel_regularizer=tf$contrib$layers$l1_regularizer(scale))


with(tf$name_scope('dnn') %as% scope, {
  hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
  hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
  logits = my_dense_layer(hidden2, n_outputs,  activation=NULL,
                          name="outputs")
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
  base_loss = tf$reduce_mean(xentropy, name="avg_xentropy")
  reg_losses = tf$get_collection(tf$GraphKeys$REGULARIZATION_LOSSES)
  loss = tf$add(base_loss, reg_losses, name="loss")
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

learning_rate = 0.01

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$GradientDescentOptimizer(learning_rate)
  training_op = optimizer$minimize(loss) 
})

init = tf$global_variables_initializer()
saver = tf$train$Saver()

n_epochs = 20
batch_size = 200
n_batches = 55000 / batch_size
n_batches

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
        X = X_train[folds[[batch_index]],],
        y = y_train[folds[[batch_index]]]))
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final.ckpt")
})

#Dropout
tf$reset_default_graph()

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

training = tf$placeholder_with_default(FALSE, shape(), name='training')

dropout_rate = 0.5  # == 1 - keep_prob
X_drop = tf$layers$dropout(X, dropout_rate, training=training)

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, name="hidden1")
  hidden1_drop = tf$layers$dropout(hidden1, dropout_rate, training=training)
  hidden2 = tf$layers$dense(hidden1_drop, n_hidden2, activation=tf$nn$relu, name="hidden2")
  hidden2_drop = tf$layers$dropout(hidden2, dropout_rate, training=training)
  logits = tf$layers$dense(hidden2_drop, n_outputs, name="outputs")
})


with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$MomentumOptimizer(learning_rate, momentum=0.9)
  training_op = optimizer$minimize(loss)
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

n_epochs = 20
batch_size = 50
n_batches = 55000 / batch_size
n_batches

init = tf$global_variables_initializer()
saver = tf$train$Saver()

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
        training = TRUE,
        X = X_train[folds[[batch_index]],],
        y = y_train[folds[[batch_index]]]))
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final.ckpt")
})


#Max-norm regularization
tf$reset_default_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

learning_rate = 0.01
momentum = 0.9

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, name="hidden1")
  hidden2 = tf$layers$dense(hidden1, n_hidden2, activation=tf$nn$relu, name="hidden2")
  logits = tf$layers$dense(hidden2, n_outputs, name="outputs")
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$MomentumOptimizer(learning_rate, momentum)
  training_op = optimizer$minimize(loss)
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

threshold = 1.0
weights = tf$get_default_graph()$get_tensor_by_name("hidden1/kernel:0")

weights

clipped_weights = tf$clip_by_norm(weights, clip_norm=threshold)
clip_weights = tf$assign(weights, clipped_weights)

weights2 = tf$get_default_graph()$get_tensor_by_name("hidden2/kernel:0")
clipped_weights2 = tf$clip_by_norm(weights2, clip_norm=threshold)
clip_weights2 = tf$assign(weights2, clipped_weights2)
  
init = tf$global_variables_initializer()
saver = tf$train$Saver()

n_epochs = 20
batch_size = 50
n_batches = 55000 / batch_size

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
        X = X_train[folds[[batch_index]],],
        y = y_train[folds[[batch_index]]]))
      clip_weights$eval()
      clip_weights2$eval()   
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final.ckpt")
})


max_norm_regularizer <- function(threshold, name="max_norm",
                                 collection="max_norm") {
  max_norm <- function(weights) {
    clipped = tf$clip_by_norm(weights, clip_norm=threshold)
    clip_weights = tf$assign(weights, clipped, name=name)
    tf$add_to_collection(collection, clip_weights)
  }
  return(max_norm)
}

tf$reset_default_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

learning_rate = 0.01
momentum = 0.9

X = tf$placeholder(tf$float32, shape(NULL, n_inputs), name="X")
y = tf$placeholder(tf$int64, shape(NULL), name="y")


max_norm_reg = max_norm_regularizer(threshold=1.0)

with(tf$name_scope('dnn') %as% scope, {
  hidden1 = tf$layers$dense(X, n_hidden1, activation=tf$nn$relu, kernel_regularizer = max_norm_reg, name="hidden1")
  hidden2 = tf$layers$dense(hidden1, n_hidden2, activation=tf$nn$relu, kernel_regularizer = max_norm_reg, name="hidden2")
  logits = tf$layers$dense(hidden2, n_outputs, name="outputs")
})

with(tf$name_scope('loss') %as% scope, {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
})

with(tf$name_scope('train') %as% scope, {
  optimizer = tf$train$MomentumOptimizer(learning_rate, momentum)
  training_op = optimizer$minimize(loss)
})

with(tf$name_scope('eval') %as% scope, {
  correct = tf$nn$in_top_k(logits, y, 1)
  accuracy = tf$reduce_mean(tf$cast(correct, tf$float32), name="accuracy")
})

init = tf$global_variables_initializer()
saver = tf$train$Saver()

n_epochs = 20
batch_size = 50
n_batches = 55000 / batch_size

clip_all_weights = tf$get_collection("max_norm")

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(y_train, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op,feed_dict=dict(
        X = X_train[folds[[batch_index]],],
        y = y_train[folds[[batch_index]]]))
      sess$run(clip_all_weights)   
    }
    accuracy_val = accuracy$eval(feed_dict = dict(X = mnist$test$images, y = mnist$test$labels))
    cat(paste(epoch, "Test accuracy:", accuracy_val, sep=' '), sep='\t', fill = T)
    
  }
  save_path = saver$save(sess, "./my_model_final.ckpt")
})
