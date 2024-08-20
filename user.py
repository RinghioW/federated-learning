import torch
import datasets
from config import DEVICE, LABEL_NAME
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from optimize import optimize_transmission_matrices
from shuffle import shuffle_data
from statistics import fmean
from copy import deepcopy
import math

CIFAR100_CLASSES = [
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
    'bottles', 'bowls', 'cans', 'cups', 'plates',
    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
    'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple', 'oak', 'palm', 'pine', 'willow',
    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
    'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor',
]

class User:
    def __init__(self, id, devices, testset, model, sampling) -> None:
        self.id = id
        self.devices = devices
        self.model = model

        self.testset = testset
        self.kd_dataset = None
        self.sampling = sampling

        self.log = []
        self.latency = 0

    def _aggregate_updates(
        self,
        epochs,
        learning_rate=0.001,
        T=2,
        soft_target_loss_weight=0.4,
        ce_loss_weight=0.6,
    ):
        # Train server model on the dataset using kd
        student = self.model().to(DEVICE)
        student.load_state_dict(torch.load("checkpoints/server.pth"))
        student.train()
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

        # TODO : Test 2 options
        # option 1 -> each device acts as a teacher separately (takes more time)
        # option 2 -> average the teacher logits from all devices (can be done in parallel)
        teachers = []
        for device in self.devices:
            teacher = device.model().to(DEVICE)
            teacher.load_state_dict(torch.load(f"checkpoints/device_{device.id}.pth"))
            teacher.eval()
            teachers.append(teacher)

        train_loader = torch.utils.data.DataLoader(
            self.kd_dataset, shuffle=True, drop_last=True, batch_size=32, num_workers=3
        )
        ce_loss = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        running_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0
        running_accuracy = 0.0
        for _ in range(epochs):
            for batch in train_loader:
                inputs, labels = batch["img"].to(DEVICE), batch[LABEL_NAME].to(DEVICE)

                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    # Keep the teacher logits for the soft targets
                    teacher_targets = []
                    for teacher in teachers:
                        logits = teacher(inputs)
                        targets = torch.nn.functional.softmax(logits / T, dim=-1)
                        teacher_targets.append(targets)


                # Forward pass with the student model
                student_logits = student(inputs)

                soft_targets = torch.mean(torch.stack(teacher_targets), dim=0)
                soft_prob = torch.nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = kl_loss(soft_prob, soft_targets) * (T ** 2)
                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = (soft_target_loss_weight * soft_targets_loss) + (
                    ce_loss_weight * label_loss
                )

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_kd_loss += soft_targets_loss.item()
                running_ce_loss += label_loss.item()
                running_accuracy += (
                    (torch.max(student_logits, 1)[1] == labels).sum().item()
                )
        torch.save(student.state_dict(), f"checkpoints/user_{self.id}.pth")



    # Train all the devices belonging to the user
    # Steps 11-15 in the ShuffleFL Algorithm
    def train(self, kd_epochs, on_device_epochs):
        if self.kd_dataset is None:
            for device in self.devices:
                torch.save(
                    device.model().state_dict(), f"checkpoints/device_{device.id}.pth"
                )
                self.create_kd_dataset()
                device.update_model(self.model, self.kd_dataset)
        else:
            for device in self.devices:
                device.update_model(self.model, self.kd_dataset)

        for device in self.devices:
            device.train(on_device_epochs)

        self._aggregate_updates(epochs=kd_epochs)

        self.test()

    def n_samples(self):
        return len(self.kd_dataset)

    def test(self):
        net = self.model().to(DEVICE)
        net.load_state_dict(torch.load(f"checkpoints/user_{self.id}.pth"))
        net.eval()
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False, num_workers=3
        )
        correct, total = 0, 0
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"].to(DEVICE), batch[LABEL_NAME].to(DEVICE)
                outputs = net(images)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                total += labels.size(0)
        print(f"USER {self.id} test accuracy: {correct / total}")
        self.log.append(correct / total)


    # Functions to determine if there are non-iid data on the devices
    # Types of non-iid data:
    # Feature distribution skew: detect by comparing the mean and variance of the features for corresponding classes
    # Label distribution skew: detect by comparing the number of samples for each class
    # Quantity skew: detect by comparing the number of samples for each class
    # Concept drift: detect by comparing the performance of the model on the devices
    def create_kd_dataset(self):
        # Sample a dataset from the devices
        # Types of sampling:
        # Full
        # Size-proportional
        # Fair
        # Upload-proportional
        # Balance-proportional
        # Balanced
        # Adaptive
        # Shuffle-optimized
        if self.kd_dataset is not None:
            return
        if self.sampling == "full":
            self.kd_dataset = self._sample_full()
        elif self.sampling == "size-proportional":
            self.kd_dataset = self._sample_size_proportional(0.4)
        elif self.sampling == "fair":
            self.kd_dataset = self._sample_fair(200)
        elif self.sampling == "balance-proportional":
            self.kd_dataset = self._sample_balance_proportional(1000)
        elif self.sampling == "upload-proportional":
            self.kd_dataset = self._sample_upload_proportional(1000)
        elif self.sampling == "balanced":
            self.kd_dataset = self._sample_balanced(10)
        elif self.sampling == "adaptive":
            self.kd_dataset = self._sample_adaptive()
        elif self.sampling == "shuffle-optimized":
            self.kd_dataset = self._sample_shuffle_optimized()
        else:
            raise ValueError(f"Sampling method {self.sampling} not recognized")



    # Functions to sample the devices
    def _sample_devices(self, fractions):
        dataset = None
        for device, f in zip(self.devices, fractions):
            if dataset is None:
                dataset = device.sample(f)
            else:
                dataset = datasets.concatenate_datasets(
                    [dataset, device.sample(f)]
                )
        return dataset
    
    def _sample_devices_amount(self, amounts):
        dataset = None
        for device, amount in zip(self.devices, amounts):
            if dataset is None:
                dataset = device.sample_amount(amount)
            else:
                dataset = datasets.concatenate_datasets(
                    [dataset, device.sample_amount(amount)]
                )
        return dataset
    
    def _sample_full(self):
        # Each device contributes all of its dataset
        self.latency = (max([device.uplink * len(device.dataset) for device in self.devices]))
        return self._sample_devices([1 for _ in self.devices])
    
    def _sample_size_proportional(self, p):
        # Each device contributes a percentage of its dataset
        fractions = [p for _ in self.devices]
        self.latency = max([device.uplink * len(device.dataset) * p for device in self.devices])
        return self._sample_devices(fractions)

    def _sample_fair(self, k):
        # Each device contributes the same number of samples
        amounts = [k for _ in self.devices]
        self.latency = (max([device.uplink for device in self.devices]) * k)
        return self._sample_devices_amount(amounts)

    def _sample_balance_proportional(self, dataset_length):
        # Take the inverse of the imbalance
        balances = [1 / (device.imbalance() + 1e-12) for device in self.devices]

        # Sample a higher percentage for the devices with less imbalance
        fractions = [b / sum(balances) for b in balances]
        amounts = [f * dataset_length for f in fractions]
        self.latency= (max([device.uplink * amount for device, amount in zip(self.devices, amounts)]))
        return self._sample_devices_amount(amounts)
    
    def _sample_upload_proportional(self, dataset_length):
        # Sample more from the devices with faster upload speeds
        upload_speeds = [1 / device.uplink for device in self.devices]
        fractions = [u / sum(upload_speeds) for u in upload_speeds]
        amounts = [f * dataset_length for f in fractions]
        self.latency = (max([device.uplink * amount for device, amount in zip(self.devices, amounts)]))
        return self._sample_devices_amount(amounts)

    def _sample_balanced(self, samples_per_class):
        # Sample in a greedy way (starting with the device with faster upload) so that the final dataset's labels are balanced
        devices_by_upload = sorted(self.devices, key=lambda x: x.uplink)
        n_classes = 100
        s = [samples_per_class for _ in range(n_classes)]
        dataset = None
        latencies = []
        # Get the label distribution for each device
        for device in devices_by_upload:
            # Sample as much as possible from the device
            device_latency = 0
            for class_id in range(n_classes):
                if s[class_id] <= 0:
                    continue
                samples = device.sample_amount_class(s[class_id], CIFAR100_CLASSES[class_id])
                if dataset is None:
                    dataset = samples
                else:
                    dataset = datasets.concatenate_datasets([dataset, samples])
                s[class_id] -= len(samples)
                device_latency += device.uplink * len(samples)
            latencies.append(device_latency)
        self.latency = (max(latencies))
        return dataset

    def _sample_adaptive(self):
        # Identify the kind of non-iid data on the devices
        # Use a sampling technique for the devices based on the identified non-iid data
        if self._label_distribution_skew():
            return self._sample_balance_proportional(1000)
        elif self._bottleneck():
            return self._sample_upload_proportional(1000)
        elif self._quantity_skew():
            return self._sample_size_proportional(0.3)
        else:
            return self._sample_fair(200)

    def _sample_shuffle_optimized(self):
        # User shuffles according tifo the ShuffleFL algorithm
        # Cluster distributions 
        self._shuffle()
        # Use kmeans to cluster the devices
        return self._sample_devices([0.4 for _ in self.devices])
    
    # Implements section 4.4 from ShuffleFL
    def _reduce_dimensionality(self):
        lda_estimator, kmeans_estimator = self._compute_centroids()
        for device in self.devices:
            device.cluster(lda_estimator, kmeans_estimator)
    
    def _shuffle(self):
        # Reduce dimensionality of the transmission matrices
        # ShuffleFL step 7, 8
        self._reduce_dimensionality()
        # User optimizes the transmission matrices
        # ShuffleFL step 9
        adaptive_scaling_factor = 1
        cluster_distributions = [device.cluster_distribution() for device in self.devices]
        uplinks = [device.uplink for device in self.devices]
        downlinks = [device.uplink for device in self.devices]
        computes = [device.compute for device in self.devices]
        transition_matrices = optimize_transmission_matrices(adaptive_scaling_factor, cluster_distributions, uplinks, downlinks, computes)

        # Shuffle the data and update the transition matrices
        # Implements Equation 1 from ShuffleFL
        datasets = [device.dataset for device in self.devices]
        clusters = [device.clusters for device in self.devices]
        res_datasets, n_transferred_samples = shuffle_data(datasets, clusters, cluster_distributions, transition_matrices)
        print("Shuffle complete")
        self.latency = np.mean([device.uplink * device.compute * n_transferred_samples for device in self.devices])
        print(self.latency)
        # Update average capability
        # Update the devices with the new datasets
        for device, dataset in zip(self.devices, res_datasets):
            device.dataset = dataset


    def _compute_centroids(self):
        # Sample based on the uplink rate of the devices
        dataset = self._sample_devices([0.1 for _ in self.devices])
        dataset = dataset.shuffle()
        features = np.array(dataset["img"]).reshape(len(dataset), -1)
        labels = np.array(dataset["fine_label"])

        lda = LinearDiscriminantAnalysis(n_components=4).fit(features, labels)
        feature_space = lda.transform(features)
        # Cluster datapoints to k classes using KMeans
        n_clusters = math.floor(0.05*100)
        kmeans = KMeans(n_clusters=n_clusters).fit(feature_space)

        return lda, kmeans
    
    # Functions to determine data heterogeneity on the devices
    def _quantity_skew(self) -> bool:
        # Check the number of samples the devices
        quantities = np.array([device.n_samples() for device in self.devices])
        # Check the Z score
        z = np.abs((quantities - np.mean(quantities)) / np.std(quantities))
        return np.any(z > 2)

    def _label_distribution_skew(self) -> bool:
        # Check the number of samples for each class
        labels = np.array([device.imbalance() for device in self.devices])
        # Determine if the number of samples is skewed using the coefficient of variation
        z = np.abs((labels - np.mean(labels)) / (np.std(labels)) + 10e-6)
        return np.any(z > 2)

    def _bottleneck(self) -> bool:
        # Check the performance of the model on the devices
        latencies = np.array([device.uplink for device in self.devices])
        # Check the Z score
        z = np.abs((latencies - np.mean(latencies)) / (np.std(latencies)) + 10e-6)
        return np.any(z > 2)
    
    def _quality_skew(self) -> bool:
        # Sample a small amount of data from each device
        return False

    def _feature_skew(self) -> bool:
        # Sample a small amount of data from each device
        return False
        