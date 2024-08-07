import torch
import datasets
from config import DEVICE, LABEL_NAME
import numpy as np

class User():
    def __init__(self, id, devices, testset, model) -> None:
        self.id = id
        self.devices = devices
        self.model = model

        self.testset = testset
        self.kd_dataset = None
        
        self.log = []

    
    def _aggregate_updates(self, epochs, learning_rate=0.001, T=2, soft_target_loss_weight=0.4, ce_loss_weight=0.6):
        
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
        
        train_loader = torch.utils.data.DataLoader(self.kd_dataset, shuffle=True, drop_last=True, batch_size=32, num_workers=3)
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
                soft_targets = torch.nn.functional.softmax(averaged_teacher_logits / T, dim=-1)
                soft_prob = torch.nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = (soft_target_loss_weight * soft_targets_loss) + (ce_loss_weight * label_loss)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_kd_loss += soft_targets_loss.item()
                running_ce_loss += label_loss.item()
                running_accuracy += (torch.max(student_logits, 1)[1] == labels).sum().item()
        torch.save(student.state_dict(), f"checkpoints/user_{self.id}.pth")

    def create_kd_dataset(self):
        # Sample a dataset from the devices
        # self.kd_dataset = self._sample_devices_uniform()
        self.kd_dataset = self._sample_devices_uniform()

    # Train all the devices belonging to the user
    # Steps 11-15 in the ShuffleFL Algorithm
    def train(self, kd_epochs, on_device_epochs):

        if self.kd_dataset is not None:
            for device in self.devices:
                device.update_model(self.model, self.kd_dataset)
        else:
            for device in self.devices:
                torch.save(device.model().state_dict(), f"checkpoints/device_{device.id}.pth")

        for device in self.devices:
            device.train(on_device_epochs)
        
        self.create_kd_dataset()
        self._aggregate_updates(epochs=kd_epochs)

        self.test()


    
    def _sample_devices(self, percentages):
        dataset = None
        for device, percentage in zip(self.devices, percentages):
            if dataset is None:
                dataset = device.sample(percentage)
            dataset = datasets.concatenate_datasets([dataset, device.sample(percentage)])
        return dataset

    def n_samples(self):
        return len(self.kd_dataset)

    def test(self):
        net = self.model().to(DEVICE)
        net.load_state_dict(torch.load(f"checkpoints/user_{self.id}.pth"))
        net.eval()
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=32, shuffle=False, num_workers=3)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"].to(DEVICE), batch[LABEL_NAME].to(DEVICE)
                outputs = net(images)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                total += labels.size(0)
        print(f"USER {self.id} test accuracy: {correct / total}")
        self.log.append(correct / total)
    
    def flush(self):
        with open(f"results/user_{self.id}.log", "w") as f:
            for accuracy in self.log:
                f.write(f"{accuracy}\n")

        import matplotlib.pyplot as plt
        plt.plot(self.log)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(f"results/user_{self.id}.svg")
        plt.close()
    
    # Functions to determine if there are non-iid data on the devices
    # Types of non-iid data:
    # Feature distribution skew: detect by comparing the mean and variance of the features for corresponding classes
    # Label distribution skew: detect by comparing the number of samples for each class
    # Quantity skew: detect by comparing the number of samples for each class
    # Concept drift: detect by comparing the performance of the model on the devices

    def _quantity_skew(self) -> bool:
        # Check the number of samples the devices
        quantities = [device.n_samples() for device in self.devices]
        # Determine if the number of samples is skewed using the coefficient of variation
        cv = np.std(quantities) / np.mean(quantities)
        return cv > 0.1

    def _label_distribution_skew(self) -> bool:
        # Check the number of samples for each class
        labels = [device.labels() for device in self.devices]
        # Determine if the number of samples is skewed using the coefficient of variation
        cv = np.std(labels) / np.mean(labels)
        return cv > 0.1
    
    # Functions to sample the devices
    def _sample_devices_proportional(self):
        # Each device contributes a percentage of its dataset
        p = 0.1
        percentages = [p for _ in self.devices]
        return self._sample_devices(percentages)
    
    def _sample_devices_fixed(self):
        # Each device contributes the same number of samples
        n_samples = 100
        percentages = [n_samples / device.n_samples() for device in self.devices]
        return self._sample_devices(percentages)
    
    def _sample_devices_imbalance(self):
        # Take the inverse of the imbalance, normalized by the number of samples
        total_samples = sum([device.n_samples() for device in self.devices])
        balances = [(1 / (device.imbalance() + 1e-12)) * (device.n_samples()/total_samples) for device in self.devices]
        
        # Sample a higher percentage for the devices with less imbalance
        percentages = [b / sum(balances) for b in balances]
        return self._sample_devices(percentages)
    
    def  _sample_devices_upload(self):
        # Take the inverse of the imbalance, normalized by the number of samples
        total_samples = sum([device.n_samples() for device in self.devices])
        balances = [device.upload * (device.n_samples()/total_samples) for device in self.devices]
        
        # Sample a higher percentage for the devices with less imbalance
        percentages = [b / sum(balances) for b in balances]
        return self._sample_devices(percentages)