import torch
import datasets
from config import DEVICE, LABEL_NAME
import numpy as np


class User:
    def __init__(self, id, devices, testset, model, sampling) -> None:
        self.id = id
        self.devices = devices
        self.model = model

        self.testset = testset
        self.kd_dataset = None
        self.sampling = sampling

        self.log = []
        self.latencies = []

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
                    teacher_logits = []
                    for teacher in teachers:
                        logits = teacher(inputs)
                        teacher_logits.append(logits)

                # Forward pass with the student model
                student_logits = student(inputs)

                averaged_teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0)
                soft_targets = torch.nn.functional.softmax(
                    averaged_teacher_logits / T, dim=-1
                )
                soft_prob = torch.nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = (
                    -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)
                )

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
        else:
            for device in self.devices:
                device.update_model(self.model, self.kd_dataset)

        for device in self.devices:
            device.train(on_device_epochs)

        self.create_kd_dataset()
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

    def flush(self, results_dir):
        with open(f"{results_dir}/user_{self.id}.log", "w") as f:
            for accuracy in self.log:
                f.write(f"{accuracy}\n")

        with open(f"{results_dir}/user_{self.id}_latencies.log", "w") as f:
            for latency in self.latencies:
                f.write(f"{latency}\n")

        import matplotlib.pyplot as plt

        plt.plot(self.log)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(f"{results_dir}/user_{self.id}.svg")
        plt.close()

        plt.plot(self.latencies)
        plt.xlabel("Epoch")
        plt.ylabel("Latency")
        plt.savefig(f"{results_dir}/user_{self.id}_latencies.svg")
        plt.close()

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
        if self.sampling == "full":
            self.kd_dataset = self._sample_full()
            latency = max([device.uplink * len(device.dataset) for device in self.devices])
        elif self.sampling == "size-proportional":
            p = 0.2
            self.kd_dataset = self._sample_size_proportional(p)
            latency = max([device.uplink * len(device.dataset) for device in self.devices]) * p
        elif self.sampling == "fair":
            k = 50
            self.kd_dataset = self._sample_fair(k)
            latency = max([device.uplink for device in self.devices]) * k
        elif self.sampling == "balance-proportional":
            self.kd_dataset = self._()
        elif self.sampling == "upload-proportional":
            self.kd_dataset = self._sa
        elif self.sampling == "balanced":
            self.kd_dataset = self._sample_balanced(100)
        elif self.sampling == "adaptive":
            self.kd_dataset = self._sample_adaptive()
        elif self.sampling == "shuffle-optimized":
            self.kd_dataset = self._sample_shuffle_optimized()

        self.latencies.append(latency)

    # Functions to sample the devices
    def _sample_devices(self, fractions):
        dataset = None
        for device, f in zip(self.devices, fractions):
            if dataset is None:
                dataset = device.sample(f)
            dataset = datasets.concatenate_datasets(
                [dataset, device.sample(f)]
            )
        return dataset
    
    def _sample_full(self):
        # Each device contributes all of its dataset
        return self._sample_devices([1 for _ in self.devices])
    
    def _sample_size_proportional(self, p):
        # Each device contributes a percentage of its dataset
        fractions = [p for _ in self.devices]
        return self._sample_devices(fractions)

    def _sample_fair(self, k):
        # Each device contributes the same number of samples
        fractions = [k / device.n_samples() for device in self.devices]
        return self._sample_devices(fractions)

    def _sample_balance_proportional(self):
        # Take the inverse of the imbalance, normalized by the number of samples
        total_samples = sum([device.n_samples() for device in self.devices])
        balances = [
            (1 / (device.imbalance() + 1e-12)) * (device.n_samples() / total_samples)
            for device in self.devices
        ]

        # Sample a higher percentage for the devices with less imbalance
        fractions = [b / sum(balances) for b in balances]
        return self._sample_devices(fractions)
    
    def _sample_upload_proportional(self):
        # Sample more from the devices with faster upload speeds
        upload_speeds = [device.uplink for device in self.devices]
        fractions = [u / sum(upload_speeds) for u in upload_speeds]
        return self._sample_devices(fractions)

    def _sample_balanced(self, samples_per_class):
        # Sample in a greedy way (starting with the device with faster upload) so that the final dataset's labels are balanced
        devices_by_upload = sorted(self.devices, key=lambda x: x.uplink, reverse=True)
        n_classes = 100
        s = [samples_per_class] * n_classes
        dataset = None
        # Get the label distribution for each device
        for device in devices_by_upload:
            # Sample as much as possible from the device
            for class_id in range(n_classes):
                if s[class_id] <= 0:
                    break
                samples = device.sample_amount_class(s[class_id], class_id)
                if dataset is None:
                    dataset = samples
                else:
                    dataset = datasets.concatenate_datasets([dataset, samples])
                s[class_id] -= len(samples)

    def _sample_adaptive(self):
        # Identify the kind of non-iid data on the devices
        # Use a sampling technique for the devices based on the identified non-iid data
        if self._label_distribution_skew():
            return self._()
        elif self._bottleneck():
            return self._sample_upload_proportional()
        elif self._quantity_skew():
            return self._sample_fair(50)
        else:
            return self._sample_size_proportional(0.2)

    def _sample_shuffle_optimized(self):
        # Integrate with shuffle optimization
        pass


    # Functions to determine if there are non-iid data on the devices
    def _quantity_skew(self) -> bool:
        # Check the number of samples the devices
        quantities = np.array([device.n_samples() for device in self.devices])
        # Check the Z score
        z = np.abs((quantities - np.mean(quantities)) / np.std(quantities))
        return np.any(z > 2)

    def _label_distribution_skew(self) -> bool:
        # Check the number of samples for each class
        labels = [device.labels() for device in self.devices]
        # Determine if the number of samples is skewed using the coefficient of variation
        cv = np.std(labels) / np.mean(labels)
        return cv > 0.1
    
    def _quality_skew(self) -> bool:
        # Sample a small amount of data from each device
        return False

    def _feature_skew(self) -> bool:
        # Sample a small amount of data from each device
        return False

    def _bottleneck(self) -> bool:
        # Check the performance of the model on the devices
        quantities = np.array([device.uplink for device in self.devices])
        # Check the Z score
        z = np.abs((quantities - np.mean(quantities)) / np.std(quantities))
        return np.any(z > 2)
        