#devtools::install_github("r-lib/processx") 
#devtools::update_packages()
#install.packages("reticulate")
#devtools::install_github("rstudio/keras")
#Sys.setenv(TENSORFLOW_PYTHON="/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5")
#devtools::install_github("rstudio/tensorflow", force=T)

#스터디 (Hands-on 9장) Up and running with TensorFlow  (7/11)
#R 코드로 바꾸기 (약 95% 완료했음)

library(devtools)
library(tensorflow)
sys <- import("sys")
sys$path

x = tf$Variable(3, name="x")
y = tf$Variable(4, name="y")
f <- x * x + y + y + 2
x
y
f

sess = tf$Session()
sess$run(x$initializer)
sess$run(y$initializer)
result = sess$run(f)
result
sess$close()

with(tf$Session() %as% sess, {
  x$initializer$run()
  y$initializer$run()
  result = f$eval()
})

init = tf$global_variables_initializer()
with(tf$Session() %as% sess, {
  init$run()
  result = f$eval()
})

sess = tf$InteractiveSession()
init$run()
result = f$eval()
print(result)
sess$close()

hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)
reset_default_graph()

#Managing graphs
tf$reset_default_graph()
x1 = tf$Variable(1)
x1$graph == tf$get_default_graph()
graph = tf$Graph()

with(graph$as_default() %as% sess, {
  x2 = tf$Variable(2)
})
x2$graph == graph
x2$graph == tf$get_default_graph()


#Lifecycle of a node value
w = tf$constant(3) 
x = w + 2 
y = x + 5 
z = x * 3

with(tf$Session() %as% sess, {
  print(y$eval()) # 10 
  print(z$eval())
})

with(tf$Session() %as% sess, {
  val = sess$run(c(y, z))
  print(val[1])
  print(val[2])
})

#Linear Regression with TensorFlow
dataset <- import("sklearn.datasets")
housing <- dataset$fetch_california_housing()
dim(housing$data)
np <- import("numpy")
m = dim(housing$data)[1]
n = dim(housing$data)[2]
housing_data_plus_bias = cbind(matrix(rep(1,m)), housing$data)
housing$target

tf$constant()
X = tf$constant(housing_data_plus_bias, name="X")
y = tf$constant(housing$target, shape=c(m,1), name="y")
X
y
XT = tf$transpose(X)
XT
theta = tf$matmul(tf$matmul(tf$matrix_inverse(tf$matmul(XT, X)), XT), y)
with(tf$Session() %as% sess, {
  theta_value = theta$eval()
})

theta_value

n_epochs = 1000
learning_rate = 0.01

scaled_housing_data_plus_bias = cbind(matrix(rep(1,m)), scale(housing$data))
tf$reset_default_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf$constant(scaled_housing_data_plus_bias, name="X", dtype = 'float32')
y = tf$constant(housing$target, shape=c(m,1), name="y", dtype = 'float32')
theta = tf$Variable(tf$random_uniform(shape(n+1,1), minval = -1.0, maxval = 1.0, 
                                      seed=42, dtype = 'float32'), name='theta')
X
y

theta
y_pred = tf$matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf$reduce_mean(tf$square(error), name='mse')
gradients = 2/m * tf$matmul(tf$transpose(X), error) 
training_op = tf$assign(theta, theta - learning_rate * gradients)
training_op

init = tf$global_variables_initializer()

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    if (epoch %% 100 == 0)
    {
      cat(paste("Epoch", epoch, "MSE =", mse$eval(), sep=' '), sep='\t', fill = T)
    }
    sess$run(training_op)
  }
best_theta = theta$eval()
})

#using autodiff
tf$reset_default_graph()

a = tf$Variable(0.2, name="a", dtype = 'float32')
b = tf$Variable(0.3, name="b", dtype = 'float32')
z = tf$constant(0.0, name="z0", dtype = 'float32')

for (i in 0:99)
{
  z = tf$add(tf$multiply(a, tf$cos(z + i)), tf$multiply(z, tf$sin(tf$subtract(b, i))))
}

z

grads = tf$gradients(z, c(a, b))
init = tf$global_variables_initializer()
grads
init

#Using an optimizer
tf$reset_default_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf$constant(scaled_housing_data_plus_bias, name="X", dtype = 'float32')
y = tf$constant(housing$target, shape=c(m,1), name="y", dtype = 'float32')

theta = tf$Variable(tf$random_uniform(shape(n+1,1), minval = -1.0, maxval = 1.0, 
                                      seed=42, dtype = 'float32'), name='theta')
y_pred = tf$matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf$reduce_mean(tf$square(error), name='mse')

optimizer = tf$train$GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer$minimize(mse)
init = tf$global_variables_initializer()

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    if (epoch %% 100 == 0)
    {
      cat(paste("Epoch", epoch, "MSE =", mse$eval(), sep=' '), sep='\t', fill = T)
    }
    sess$run(training_op)
  }
  best_theta = theta$eval()
})

print("Best theta:")
print(best_theta)

#Feeding data to the training algorithm
tf$reset_default_graph()
A = tf$placeholder(tf$float32, shape(NULL, 3))
B = A + 5

with(tf$Session() %as% sess, {
  B_val_1 = B$eval(feed_dict = feed_data)
})

feed_data = dict(A=c(1,2,3))
feed_data


tf$reset_default_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf$placeholder(tf$float32, shape(NULL, n + 1), name="X")
y = tf$placeholder(tf$float32, shape(NULL, 1), name="y")

theta = tf$Variable(tf$random_uniform(shape(n + 1, 1), minval= -1.0, maxval = 1.0, seed=42), name="theta", dtype = 'float32')
y_pred = tf$matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf$reduce_mean(tf$square(error), name="mse")
optimizer = tf$train$GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer$minimize(mse)

init = tf$global_variables_initializer()

n_epochs = 10
batch_size = 100
n_batches = (np$ceil(m / batch_size))

library(caret)

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(housing$target, k = n_batches)
    for (batch_index in 1:n_batches){
      sess$run(training_op, feed_dict=dict(X = scaled_housing_data_plus_bias[folds[[batch_index]],],
                                           y = matrix(housing$target[folds[[batch_index]]])))
    }
  }
  best_theta = theta$eval()
})


### Saving and restoring models
n_epochs = 1000                                                                       # not shown in the book
learning_rate = 0.01                                                                  # not shown

tf$reset_default_graph()

X = tf$constant(scaled_housing_data_plus_bias, name="X", dtype = 'float32')
y = tf$constant(housing$target, shape=c(m,1), name="y", dtype = 'float32')

theta = tf$Variable(tf$random_uniform(shape(n+1,1), minval = -1.0, maxval = 1.0, 
                                      seed=42, dtype = 'float32'), name='theta')

y_pred = tf$matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf$reduce_mean(tf$square(error), name='mse')

optimizer = tf$train$GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer$minimize(mse)
init = tf$global_variables_initializer()
saver = tf$train$Saver()

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    if (epoch %% 100 == 0)
    {
      cat(paste("Epoch", epoch, "MSE =", mse$eval(), sep=' '), sep='\t', fill = T)
      save_path = saver$save(sess, "./tmp/my_model.ckpt")
    }
    sess$run(training_op)
  }
  best_theta = theta$eval()
  save_path = saver$save(sess, "./tmp/my_model_final.ckpt")
})

with(tf$Session() %as% sess, {
  saver$restore(sess, "./tmp/my_model_final.ckpt")
  best_theta_restored = theta$eval()
})

np$allclose(best_theta, best_theta_restored)

tf$reset_default_graph()

saver = tf$train$import_meta_graph("./tmp/my_model_final.ckpt.meta")  # this loads the graph structure
theta = tf$get_default_graph()$get_tensor_by_name("theta:0") # not shown in the book

with(tf$Session() %as% sess, {
  saver$restore(sess, "./tmp/my_model_final.ckpt")  # this restores the graph's state
  best_theta_restored = theta$eval()
})

np$allclose(best_theta, best_theta_restored)

###Visualizing the graph and training curves using TensorBoard
tf$reset_default_graph()
datetime <- import("datetime")

now = datetime$datetime$utcnow()$strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"

logdir = sprintf("%s/run-%s/", root_logdir, now)
n_epochs = 1000
learning_rate = 0.01

X = tf$placeholder(tf$float32, shape(NULL, n + 1), name="X")
y = tf$placeholder(tf$float32, shape(NULL, 1), name="y")

theta = tf$Variable(tf$random_uniform(shape(n + 1, 1), minval= -1.0, maxval = 1.0, seed=42), name="theta", dtype = 'float32')
y_pred = tf$matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf$reduce_mean(tf$square(error), name="mse")
optimizer = tf$train$GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer$minimize(mse)

init = tf$global_variables_initializer()
mse_summary = tf$summary$scalar('MSE', mse)
file_writer = tf$summary$FileWriter(logdir, tf$get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = (np$ceil(m / batch_size))

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(housing$target, k = n_batches)
    for (batch_index in 1:n_batches){
      if (batch_index %% 10 == 0){
        summary_str = mse_summary$eval(feed_dict=dict(X = scaled_housing_data_plus_bias[folds[[batch_index]],],
                                                      y = matrix(housing$target[folds[[batch_index]]])))
        step = epoch * n_batches + batch_index
        file_writer$add_summary(summary_str, step)
      }
    sess$run(training_op, feed_dict=dict(X = scaled_housing_data_plus_bias[folds[[batch_index]],],
                                         y = matrix(housing$target[folds[[batch_index]]])))
    }
  }
  best_theta = theta$eval()
})
file_writer$close()
logdir
#terminal에서 tensorboard --logdir tf_logs/ 실행 접속
best_theta

###Name scopes
tf$reset_default_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf$placeholder(tf$float32, shape(NULL, n + 1), name="X")
y = tf$placeholder(tf$float32, shape(NULL, 1), name="y")

theta = tf$Variable(tf$random_uniform(shape(n + 1, 1), minval= -1.0, maxval = 1.0, seed=42), name="theta", dtype = 'float32')
y_pred = tf$matmul(X, theta, name="predictions")

with(tf$name_scope('loss') %as% scope, {
  error = y_pred - y
  mse = tf$reduce_mean(tf$square(error), name="mse")
})

error$op$name
mse$op$name

optimizer = tf$train$GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer$minimize(mse)

init = tf$global_variables_initializer()

mse_summary = tf$summary$scalar('MSE', mse)
file_writer = tf$summary$FileWriter(logdir, tf$get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = (np$ceil(m / batch_size))

with(tf$Session() %as% sess, {
  init$run()
  for (epoch in 1:n_epochs){
    folds <- createFolds(housing$target, k = n_batches)
    for (batch_index in 1:n_batches){
      if (batch_index %% 10 == 0){
        summary_str = mse_summary$eval(feed_dict=dict(X = scaled_housing_data_plus_bias[folds[[batch_index]],],
                                                      y = matrix(housing$target[folds[[batch_index]]])))
        step = epoch * n_batches + batch_index
        file_writer$add_summary(summary_str, step)
      }
      sess$run(training_op, feed_dict=dict(X = scaled_housing_data_plus_bias[folds[[batch_index]],],
                                           y = matrix(housing$target[folds[[batch_index]]])))
    }
  }
  best_theta = theta$eval()
})

file_writer$flush()
file_writer$close()
paste("Best theta:", best_theta)

##name scope, simple example
tf$reset_default_graph()

with(tf$name_scope('scope_1') %as% scope, {
  a = tf$add(1,2)
  b = tf$multiply(a,3)
})

with(tf$name_scope('scope_2') %as% scope, {
  x = tf$add(7,8)
  y = tf$multiply(x,9)
})

o = tf$add(b,y,name='output')
sess = tf$Session()
writer = tf$summary$FileWriter('./output01', graph=sess$graph)
writer$close()
#터미널 확인 tensorboard --logdir output01
sess$close()

a1 = tf$Variable(0, name="a")      # name == "a"
a2 = tf$Variable(0, name="a")      # name == "a_1"

with(tf$name_scope('param') %as% scope, {
  a3 = tf$Variable(0, name="a") 
})

with(tf$name_scope('param') %as% scope, {
  a4 = tf$Variable(0, name="a") 
})

for (node in c(a1, a2, a3, a4)){
  print(node$op$name)
}

### Modularity
tf$reset_default_graph()

n_features = 3
X = tf$placeholder(tf$float32, shape(NULL, n_features), name="X")

w1 = tf$Variable(tf$random_normal(shape(n_features, 1)), name="weights1")
w2 = tf$Variable(tf$random_normal(shape(n_features, 1)), name="weights2")
b1 = tf$Variable(0.0, name="bias1")
b2 = tf$Variable(0.0, name="bias2")

z1 = tf$add(tf$matmul(X, w1), b1, name="z1")
z2 = tf$add(tf$matmul(X, w2), b2, name="z2")

relu1 = tf$maximum(z1, 0., name="relu1")
relu2 = tf$maximum(z1, 0., name="relu2")  # Oops, cut&paste error! Did you spot it?

output = tf$add(relu1, relu2, name="output")

tf$reset_default_graph()

relu <- function(X){
  #w_shape = X$get_shape()[[1]]
  w = tf$Variable(tf$random_normal(shape(3,1)), name="weights")
  b = tf$Variable(0.0, name="bias")
  z = tf$add(tf$matmul(X, w), b, name="z")
  return(tf$maximum(z, 0., name="relu"))
}

n_features = 3
X = tf$placeholder(tf$float32, shape(NULL, n_features), name="X")

relus = NULL
for (i in 1:5){
  relus = c(relus,relu(X))
}
output = tf$add_n(relus, name="output")
outputfile_writer = tf$summary$FileWriter("logs/relu1", tf$get_default_graph())
#터미널 확인 tensorboard --logdir logs/relu1
tf$reset_default_graph()

relu <- function(X){
  with(tf$name_scope('relu') %as% scope, {
    #w_shape = X$get_shape()[[1]]
    w = tf$Variable(tf$random_normal(shape(3,1)), name="weights")
    b = tf$Variable(0.0, name="bias")
    z = tf$add(tf$matmul(X, w), b, name="z")
    return(tf$maximum(z, 0., name="relu"))
  })
}

n_features = 3
X = tf$placeholder(tf$float32, shape(NULL, n_features), name="X")

relus = NULL
for (i in 1:5){
  relus = c(relus,relu(X))
}
output = tf$add_n(relus, name="output")
file_writer = tf$summary$FileWriter("logs/relu2", tf$get_default_graph())
file_writer$close()
#터미널 확인 tensorboard --logdir logs/relu2


###Sharing Variables
tf$reset_default_graph()

relu <- function(X, threshold){
  with(tf$name_scope('relu') %as% scope, {
    #w_shape = X$get_shape()[[1]]
    w = tf$Variable(tf$random_normal(shape(3,1)), name="weights")
    b = tf$Variable(0.0, name="bias")
    z = tf$add(tf$matmul(X, w), b, name="z")
    return(tf$maximum(z, threshold, name="max"))
  })
}

n_features = 3
threshold = tf$Variable(0.0, name="threshold")
X = tf$placeholder(tf$float32, shape(NULL, n_features), name="X")

relus = NULL
for (i in 1:5){
  relus = c(relus,relu(X, threshold))
}
output = tf$add_n(relus, name="output")

#if not hasattr 모르겠음 ...ㅜㅜ

tf$reset_default_graph()

with(tf$variable_scope('relu') %as% scope, {
  threshold = tf$get_variable("threshold", shape(),
                              initializer=tf$constant_initializer(0.0))
})

with(tf$variable_scope('relu') %as% scope, {
  scope$reuse_variables()
  threshold = tf$get_variable("threshold")
})

tf$reset_default_graph()

relu <- function(X){
  with(tf$variable_scope('relu') %as% scope, {
    scope$reuse_variables()
    threshold = tf$get_variable("threshold")
    #w_shape = X$get_shape()[[1]]
    w = tf$Variable(tf$random_normal(shape(3,1)), name="weights")
    b = tf$Variable(0.0, name="bias")
    z = tf$add(tf$matmul(X, w), b, name="z")
    return(tf$maximum(z, threshold, name="relu"))
  })
}

n_features = 3
X = tf$placeholder(tf$float32, shape(NULL, n_features), name="X")

with(tf$variable_scope('relu') %as% scope, {
  threshold = tf$get_variable("threshold", shape(),
                              initializer=tf$constant_initializer(0.0))
})

relus = NULL
for (i in 1:5){
  relus = c(relus,relu(X))
}
output = tf$add_n(relus, name="output")

file_writer = tf$summary$FileWriter("logs/relu6", tf$get_default_graph())
file_writer$close()
#터미널 확인 tensorboard --logdir logs/relu6

tf$reset_default_graph()

relu <- function(X){
  with(tf$variable_scope('relu') %as% scope, {
    threshold = tf$get_variable("threshold", shape(),
                                initializer=tf$constant_initializer(0.0))
    #w_shape = X$get_shape()[[1]]
    w = tf$Variable(tf$random_normal(shape(3,1)), name="weights")
    b = tf$Variable(0.0, name="bias")
    z = tf$add(tf$matmul(X, w), b, name="z")
    return(tf$maximum(z, threshold, name="relu"))
  })
}

n_features = 3
X = tf$placeholder(tf$float32, shape(NULL, n_features), name="X")
relus = NULL

with(tf$variable_scope('') %as% scope, {
  first_relu = relu(X)
  scope$reuse_variables()
  for (i in 1:5){
    relus = c(first_relu, relus, relu(X))
  }
})

output = tf$add_n(relus, name="output")

file_writer = tf$summary$FileWriter("logs/relu8", tf$get_default_graph())
file_writer$close()
#터미널 확인 tensorboard --logdir logs/relu6

#tf$reset_default_graph()

#relu <- function(X){
#  with(tf$variable_scope('relu') %as% scope, {
#    threshold = tf$get_variable("threshold", shape(),
#                                initializer=tf$constant_initializer(0.0))
#    #w_shape = X$get_shape()[[1]]
#    w = tf$Variable(tf$random_normal(shape(3,1)), name="weights")
#    b = tf$Variable(0.0, name="bias")
#    z = tf$add(tf$matmul(X, w), b, name="z")
#    return(tf$maximum(z, threshold, name="relu"))
#  })
#}

#n_features = 3
#X = tf$placeholder(tf$float32, shape(NULL, n_features), name="X")
#relus = NULL

#for (relu_index in 1:5){
#  with(tf$variable_scope('relu', reuse=ifelse(relu_index >=1, T, F)) %as% scope, {
#    relus = c(relus,relu(X))
#  })
#}
#output = tf$add_n(relus, name="output")
#file_writer = tf$summary$FileWriter("logs/relu9", tf$get_default_graph())
#file_writer$close()
#터미널 확인 tensorboard --logdir logs/relu9

