#devtools::install_github("r-lib/processx") 
#devtools::update_packages()
#install.packages("reticulate")
#devtools::install_github("rstudio/keras")
#Sys.setenv(TENSORFLOW_PYTHON="/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5")
#devtools::install_github("rstudio/tensorflow", force=T)

#스터디 (Hands-on 9장) Up and running with TensorFlow  (7/11)
#R 코드로 바꾸기 (약 70% 완료)

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
n_batches
m
np$random$randint()

#fetch_batch <- function(epoch, batch_index, batch_size){
#  np$random$seed(epoch * n_batches + batch_index)  # not shown in the book
#  indices = np$random$randint(m, batch_size=batch_size)  # not shown
#  X_batch = scaled_housing_data_plus_bias[indices,] # not shown
#  y_batch = housing$target$reshape(shape(-1, 1))[indices] # not shown
#  return(X_batch, y_batch)
#}

#with(tf$Session() %as% sess, {
#  init$run()
#  for (epoch in 1:n_epochs){
#    for (batch_index in 1:n_batches){
#      X_batch = fetch_batch(epoch, batch_index, batch_size)[[1]]
#      y_batch = fetch_batch(epoch, batch_index, batch_size)[[2]]
#      sess$run(training_op, feed_dict=dict(x= X_batch, y= y_batch))
#    }
#  }
#  best_theta = theta$eval()
#})