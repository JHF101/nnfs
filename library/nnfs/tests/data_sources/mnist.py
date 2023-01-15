from data_sources.mnist import MNIST

mnist = MNIST()
mnist.download_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data(directory="/mnist")
print(y_test)
