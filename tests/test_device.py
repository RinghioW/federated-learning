import unittest
from device import Device
import torch

class DeviceTest(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset and validation set
        dataset = [
            {"img": torch.tensor([0, 0, 0]), "label": 0},
            {"img": torch.tensor([1, 1, 1]), "label": 1},
            {"img": torch.tensor([2, 2, 2]), "label": 0},
            {"img": torch.tensor([3, 3, 3]), "label": 1},
            {"img": torch.tensor([4, 4, 4]), "label": 0},
            {"img": torch.tensor([5, 5, 5]), "label": 1},
        ]
        valset = [
            {"img": torch.tensor([6, 6, 6]), "label": 0},
            {"img": torch.tensor([7, 7, 7]), "label": 1},
        ]
        config = {"uplink_rate": 1, "downlink_rate": 1, "compute": 1}
        self.device = Device(config, dataset, valset)

    def test_initialize_transition_matrix(self):
        self.device.initialize_transition_matrix(3)
        expected_matrix = [[0, 0, 0], [0, 0, 0]]
        self.assertEqual(self.device.transition_matrix.tolist(), expected_matrix)

    def test_data_imbalance(self):
        imbalance = self.device.data_imbalance()
        self.assertAlmostEqual(imbalance, 0.0, places=2)

    def test_latency(self):
        devices = [self.device, self.device, self.device]
        latency = self.device.latency(0, devices, 5)
        self.assertEqual(latency, 90)

    def test_train(self):
        self.device.model = torch.nn.Linear(3, 2)
        self.device.train(epochs=2, verbose=False)
        self.assertIsNotNone(self.device.model)

    def test_remove_data(self):
        samples = self.device.remove_data(0, 0.5)
        self.assertEqual(len(samples), 3)
        self.assertEqual(len(self.device.dataset), 3)

    def test_add_data(self):
        samples = [
            {"img": torch.tensor([8, 8, 8]), "label": 0},
            {"img": torch.tensor([9, 9, 9]), "label": 1},
        ]
        self.device.add_data(samples)
        self.assertEqual(len(self.device.dataset), 8)

    def test_cluster_data(self):
        self.device.cluster_data(0.5)
        self.assertEqual(len(self.device.datset_clusters), 6)

if __name__ == '__main__':
    unittest.main()