import torch
import datasets
from config import DEVICE, LABEL_NAME

class User():
    def __init__(self, id, devices, testset) -> None:
        self.id = id
        self.devices = devices
        self.model = None

        self.testset = testset
        self.kd_dataset = None

    
    def _aggregate_updates(self, epochs, learning_rate=0.0001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75):
        
        # Train server model on the dataset using kd
        student = self.model.to(DEVICE)
        student.train()
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

        # TODO : Test 2 options
        # option 1 -> each device acts as a teacher separately (takes more time)
        # option 2 -> average the teacher logits from all devices (can be done in parallel)
        teachers = [] 
        for device in self.devices:
            teacher = device.model.to(DEVICE)
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
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_kd_loss += soft_targets_loss.item()
                running_ce_loss += label_loss.item()
                running_accuracy += (torch.max(student_logits, 1)[1] == labels).sum().item()

    def create_kd_dataset(self):
        # Sample a dataset from the devices
        percentages = [1 / len(self.devices) for _ in range(len(self.devices))]
        kd_dataset = self._sample_devices(percentages)
        self.kd_dataset = kd_dataset

    # Train all the devices belonging to the user
    # Steps 11-15 in the ShuffleFL Algorithm
    def train(self, kd_epochs, on_device_epochs):

        # TODO : Adapt model to the devices (KD on the same dataset)
        if self.kd_dataset is not None:
            for device in self.devices:
                device.update_model(self.model, self.kd_dataset)

        for device in self.devices:
            device.train(on_device_epochs)
        
        self.create_kd_dataset()
        self._aggregate_updates(epochs=kd_epochs)

        self.test()

    
    def _sample_devices(self, percentages):
        # Assemble the entire dataset from the devices
        dataset = None
        for device, percentage in zip(self.devices, percentages):
            if dataset is None:
                dataset = device.sample(percentage)
            dataset = datasets.concatenate_datasets([dataset, device.sample(percentage)])
        return dataset

    def n_samples(self):
        return len(self.kd_dataset)

    def test(self):
        net = self.model
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