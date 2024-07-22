import torch
import math
import datasets
from config import DEVICE, LABEL_NAME

class Device():
    def __init__(self, id, trainset, valset) -> None:
        self.id = id
        self.dataset = trainset
        self.valset = valset

        self.model = None

    def __repr__(self) -> str:
        return f"Device({self.config}, 'samples': {len(self.dataset)})"
    
    def sample(self, percentage):
        amount = math.floor(percentage * len(self.dataset))
        return datasets.Dataset.shuffle(self.dataset).select([i for i in range(amount)])
    
    def update_model(self, user_model, kd_dataset):
        
        # Train server model on the dataset using kd
        student = self.model.to(DEVICE)
        student.train()
        optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
        teacher = user_model.to(DEVICE)
        teacher.eval()
        
        train_loader = torch.utils.data.DataLoader(kd_dataset, shuffle=True, drop_last=True, batch_size=32, num_workers=3)
        ce_loss = torch.nn.CrossEntropyLoss()
        
        running_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0
        running_accuracy = 0.0
        epochs = 10
        T = 2
        soft_target_loss_weight = 0.75 # Priority on trasferring the knowledge
        ce_loss_weight = 0.25 # No priority on learning the labels
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

                soft_targets = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
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

    # Perform on-device learning on the local dataset. This is simply a few rounds of SGD.
    def train(self, epochs):
        print(f"DEVICE {self.id} - Training")
        if len(self.dataset) == 0:
            return
        
        # Load the model
        net = self.model.to(DEVICE)
        net.train()

        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

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
        
        self.validate()

    def validate(self):
        if self.valset is None:
            return 0.
        net = self.model.to(DEVICE)
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

    def update(self, trainset, valset):
        self.dataset = trainset
        self.valset = valset
    
