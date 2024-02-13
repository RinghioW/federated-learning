import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils.data as tdata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class User():
    def __init__(self, devices, kd_dataset=None):
        self.devices = devices
        self.kd_dataset = kd_dataset
        self.model = None

    # Adapt the model to the devices
    def adapt_model(self, model):
        self.model = model
        for device in self.devices:
            if device.config["compute"] < 5:
                device.model = models.quantization.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT, quantize=False)
            elif device.config["compute"] < 10:
                device.model = models.quantization.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, quantize=False)
            else:
                device.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                
    # Devices shuflle the data and create a knowledge distillation dataset
    def shuffle_data(self):
        for devices in self.devices:
            # Shuffle the data
            pass
        # Create a knowledge distillation dataset
        kd_dataset = []
        for device in self.devices:
            kd_dataset.append(device.valset)
        self.kd_dataset = kd_dataset

    # Train the user model using knowledge distillation
    def aggregate_updates(self, learning_rate=0.001, epochs=3, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75):
        student = self.model
        for i, device in enumerate(self.devices):
            teacher = device.model
            train_loader = torch.utils.data.DataLoader(self.kd_dataset[i], shuffle=True, batch_size=32)
            ce_loss = nn.CrossEntropyLoss()
            optimizer = optim.Adam(student.parameters(), lr=learning_rate)

            teacher.eval()  # Teacher set to evaluation mode
            student.train() # Student to train mode

            for epoch in range(epochs):
                running_loss = 0.0
                for batch in train_loader:
                    inputs, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)

                    optimizer.zero_grad()

                    # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                    with torch.no_grad():
                        teacher_logits = teacher(inputs)

                    # Forward pass with the student model
                    student_logits = student(inputs)

                    #Soften the student logits by applying softmax first and log() second
                    soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                    soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

                    # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                    soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

                    # Calculate the true label loss
                    label_loss = ce_loss(student_logits, labels)

                    # Weighted sum of the two losses
                    loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")