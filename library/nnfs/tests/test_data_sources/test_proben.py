import unittest
from nnfs.data_sources.proben1 import Proben1


class TestProben1(unittest.TestCase):
    def setUp(self):
        self.proben = Proben1()

    def test_get_dataset_dirs(self):
        dataset_dirs = self.proben.get_dataset_dirs()
        self.assertIsNotNone(dataset_dirs)

    def test_load_data(self):
        (
            (x_train, y_train),
            (x_validate, y_validate),
            (x_test, y_test),
        ) = self.proben.load_data(data_set_name="cancer")[2]
        self.assertIsNotNone(x_train)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(x_validate)
        self.assertIsNotNone(y_validate)
        self.assertIsNotNone(x_test)
        self.assertIsNotNone(y_test)

    def test_get_filenames(self):
        filename = self.proben.get_filenames(data_set_name="cancer")[2]
        self.assertIsNotNone(filename)

    def tearDown(self):
        self.proben.clean_datasets()


if __name__ == "__main__":
    unittest.main()
