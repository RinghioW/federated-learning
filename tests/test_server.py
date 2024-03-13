import unittest
from server import Server

class TestServer(unittest.TestCase):
    def setUp(self):
        # Create a Server instance with a dummy dataset
        self.server = Server("cifar10")

    def test_aggregate_updates(self):
        # TODO: Write test case for aggregate_updates method
        pass

    def test_evaluate(self):
        # TODO: Write test case for evaluate method
        pass

    def test_send_adaptive_scaling_factor(self):
        # TODO: Write test case for send_adaptive_scaling_factor method
        pass

    def test_select_users(self):
        # TODO: Write test case for select_users method
        pass

if __name__ == '__main__':
    unittest.main()
