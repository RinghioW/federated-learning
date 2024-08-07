import torch
import math
import datasets
from config import DEVICE, LABEL_NAME, NUM_CLASSES
import numpy as np
from scipy.spatial.distance import jensenshannon

class Device():
    def __init__(self, id, trainset, valset) -> None:
        self.id = id
        self.dataset = trainset
        self.valset = valset
        self.model =None
        self.log = []

    def __repr__(self) -> str:
        return f"Device({self.config}, 'samples': {len(self.dataset)})"
    
    def sample(self, percentage):
        amount = math.floor(percentage * len(self.dataset))
        return datasets.Dataset.shuffle(self.dataset).select([i for i in range(amount)])
    
    def n_samples(self):
        return len(self.dataset)
    
    def update_model(self, user_model, kd_dataset):
        # Use knowledge distillation to adapt the model to the device
        # Train server model on the dataset using kd
        student = self.model().to(DEVICE)
        student.train()
        optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

        # TODO : Test 2 options
        # option 1 -> each device acts as a teacher separately (takes more time)
        # option 2 -> average the teacher logits from all devices (can be done in parallel)
        teacher = user_model().to(DEVICE)
        teacher.load_state_dict(torch.load("checkpoints/server.pth"))
        teacher.eval()
        
        train_loader = torch.utils.data.DataLoader(kd_dataset, shuffle=True, drop_last=True, batch_size=32, num_workers=3)
        ce_loss = torch.nn.CrossEntropyLoss()
        
        running_loss = 0.0
        running_accuracy = 0.0
        num_samples = 0
        epochs = 10
        for _ in range(epochs):

            for batch in train_loader:
                inputs, labels = batch["img"].to(DEVICE), batch[LABEL_NAME].to(DEVICE)

                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    # Keep the teacher logits for the soft targets
                    teacher_logits = teacher(inputs)

                # Forward pass with the student model
                student_logits = student(inputs)
                T = 2
                soft_targets = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = torch.nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                soft_target_loss_weight = 0.75
                ce_loss_weight = 0.25
                loss = (soft_target_loss_weight * soft_targets_loss) + (ce_loss_weight * label_loss)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_samples += labels.size(0)
                running_accuracy += (torch.max(student_logits, 1)[1] == labels).sum().item()
        torch.save(student.state_dict(), f"checkpoints/device_{self.id}.pth")

    # Perform on-device learning on the local dataset. This is simply a few rounds of SGD.
    def train(self, epochs):
        print(f"DEVICE {self.id} - Training")
        if len(self.dataset) == 0:
            return
        
        # Load the model
        net = self.model().to(DEVICE)
        net.load_state_dict(torch.load(f"checkpoints/device_{self.id}.pth"))
        net.train()

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=3)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for batch in trainloader:
                images, labels = batch["img"].to(DEVICE), batch[LABEL_NAME].to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        torch.save(net.state_dict(), f"checkpoints/device_{self.id}.pth")
        self.validate()

    def validate(self):
        if self.valset is None:
            return 0.
        net = self.model().to(DEVICE)
        net.load_state_dict(torch.load(f"checkpoints/device_{self.id}.pth"))
        net.eval()
        valloader = torch.utils.data.DataLoader(self.valset, batch_size=32, shuffle=False, num_workers=3)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in valloader:
                images, labels = batch["img"].to(DEVICE), batch[LABEL_NAME].to(DEVICE)
                outputs = net(images)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                total += labels.size(0)
        print(f"DEVICE {self.id} - Validation Accuracy: {correct / total}")
        self.log.append(correct / total)

    def update(self, trainset, valset):
        self.dataset = trainset
        self.valset = valset

    def flush(self):
        with open(f"results/device_{self.id}.log", "w") as f:
            for line in self.log:
                f.write(f"{line}\n")
    
        import matplotlib.pyplot as plt
        plt.plot(self.log)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Device {self.id} Validation Accuracy")
        plt.savefig(f"results/device_{self.id}.svg")
        plt.close()
    
    def imbalance(self):
        if len(self.dataset) == 0:
            return np.float64(0.)
        distribution = np.bincount(self.dataset[LABEL_NAME], minlength=NUM_CLASSES)
        n_samples = sum(distribution)
        n_classes = len(distribution)
        avg_samples = n_samples / n_classes
        balanced_distribution = [avg_samples] * n_classes
        js = jensenshannon(balanced_distribution, distribution)
        return js