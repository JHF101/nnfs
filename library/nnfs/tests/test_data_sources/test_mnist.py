import unittest
from nnfs.data_sources.mnist import MNIST


class TestMNIST(unittest.TestCase):
    def setUp(self):
        self.mnist = MNIST()

    def test_mnist_download_and_load(self):
        self.mnist.download_data()
        (x_train, y_train), (x_test, y_test) = self.mnist.load_data()
        self.assertIsNotNone(x_train)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(x_test)
        self.assertIsNotNone(y_test)

    def tearDown(self):
        self.mnist.clean_datasets()
