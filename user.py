import torchvision.models as models
import torch
import numpy as np
from config import DEVICE, NUM_CLASSES, STD_CORRECTION
from scipy.optimize import minimize
import math
import random
import torchvision
import datasets
class User():
    def __init__(self, devices, classes=NUM_CLASSES) -> None:
        self.devices = devices
        self.kd_dataset = []
        self.model = None

        # SHUFFLE-FL

        # Transition matrix of ShuffleFL of size (floor{number of classes * shrinkage ratio}, number of devices + 1)
        # The additional column is for the kd_dataset
        # Also used by equation 7 as the optimization variable for the argmin
        # Shrinkage ratio for reducing the classes in the transition matrix
        self.shrinkage_ratio = 0.3
        self.transition_matrices = [np.full(shape=(math.floor(classes*self.shrinkage_ratio), len(devices)), fill_value=0.1, dtype=float)] * len(devices)
        
        # System latencies for each device
        self.adaptive_scaling_factor = 1.0

        # Staleness factor
        self.staleness_factor = 0.0

        # Average capability beta
        self.diff_capability = 1. + STD_CORRECTION*random.random()
        self.average_power = 1. + STD_CORRECTION*random.random()
        self.average_bandwidth = 1. + STD_CORRECTION*random.random()


    # Adapt the model to the devices
    # Uses quantization
    # TODO : ditch quantization. Instead, use torch.nn.utils.prune and torch.pca_lowrank
    def adapt_model(self, model):
        self.model = model
        for device in self.devices:
            # Adaptation is based on the device resources
            if device.config["compute"] < 5:
                device.model = models.quantization.mobilenet_v2()
            else:
                device.model = models.mobilenet_v3_small()
    
    # Train the user model using knowledge distillation
    def aggregate_updates(self, learning_rate=0.001, epochs=3, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75):
        student = self.model
        # TODO : Train in parallel, not sequentially (?)
        teachers = [device.model for device in self.devices]
        to_tensor = torchvision.transforms.ToTensor()
        kd_dataset = datasets.Dataset.from_list(self.kd_dataset)
        kd_dataset = kd_dataset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        train_loader = torch.utils.data.DataLoader(kd_dataset, shuffle=True, batch_size=32, num_workers=3)
        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for teacher in teachers:
                teacher.eval()  # Teacher set to evaluation mode
            student.train() # Student to train mode

            running_loss = 0.0
            for batch in train_loader:
                inputs, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)

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

                # Soften the student logits by applying softmax first and log() second
                # Compute the mean of the teacher logits received from all devices
                # TODO: Does the mean make sense?
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
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader.dataset)}")
        self.model = student

    # Train all the devices belonging to the user
    # Steps 11-15 in the ShuffleFL Algorithm
    def train_devices(self, epochs=5, verbose=True):
        for device in self.devices:
            device.train(epochs, verbose)
    
    def get_latencies(self, epochs):
        t_communication = [0.] * len(self.devices)
        t_computation = [0.] * len(self.devices)
        latencies = [0.] * len(self.devices)
        
        # For each device, compute communication and computation time
        for device_idx, device in enumerate(self.devices):
            transition_matrix = self.transition_matrices[device_idx]
            # Compute the communication time
            for data_class_idx, _ in enumerate(transition_matrix):
                for other_device_idx, other_device in enumerate(self.devices):
                    other_transition_matrix = self.transition_matrices[other_device_idx]
                    if device_idx != other_device_idx:
                        # Transmitting
                        num_transmitted_samples = len(device.dataset) * transition_matrix[data_class_idx][other_device_idx]
                        t_communication[device_idx] += num_transmitted_samples * ((1/device.config["uplink_rate"]) + (1/other_device.config["downlink_rate"]))
                        # Receiving
                        num_received_samples = len(other_device.dataset) * other_transition_matrix[data_class_idx][device_idx]
                        t_communication[device_idx] += num_received_samples * ((1/device.config["downlink_rate"]) + (1/other_device.config["uplink_rate"]))
            # Compute the computation time
            t_computation[device_idx] += 3 * epochs * len(device.dataset) * device.config["compute"]
            
            # Compute the latency as the sum of both
            latencies[device_idx] = t_communication[device_idx] + t_computation[device_idx]
        return latencies
    
    def get_data_imbalances(self):
        data_imbalances = []
        for device in self.devices:
            data_imbalances.append(device.data_imbalance())
        return data_imbalances
    
    def send_data(self, sender_idx, receiver_idx, cluster, percentage_amount):
        # Identify sender and receiver
        sender = self.devices[sender_idx]
        receiver = self.devices[receiver_idx]

        # If the receiver is the same as the sender, add the samples to the kd_dataset
        if sender_idx == receiver_idx:
            sender.remove_data(cluster=cluster, percentage_amount=percentage_amount, add_to_kd_dataset=True)
        else:
            # Sender removes some samples
            samples, clusters = sender.remove_data(cluster, percentage_amount)
            # Receiver adds those samples
            receiver.add_data(samples, clusters)

    # Shuffle data between devices according to the transition matrices
    # Implements the transformation described by Equation 1 from ShuffleFL
    def shuffle_data(self, transition_matrices):
        # Each device sends data according to the respective transition matrix
        for device_idx, transition_matrix in enumerate(transition_matrices):
            for cluster_idx in range(len(transition_matrix)):
                for other_device_idx in range(len(transition_matrix[0])):
                        # Send data from cluster i to device j
                        print(f"Sending data from device {device_idx} to device {other_device_idx} in cluster {cluster_idx}")
                        self.send_data(sender_idx=device_idx, receiver_idx=other_device_idx, cluster=cluster_idx, percentage_amount=transition_matrix[cluster_idx][other_device_idx])

    # Function to implement the dimensionality reduction of the transition matrices
    # The data is embedded into a 2-dimensional space using t-SNE
    # The classes are then aggregated into k groups using k-means
    # Implements section 4.4 from ShuffleFL
    def reduce_dimensionality(self):
        for device in self.devices:
            device.cluster_data(self.shrinkage_ratio)

    # Function for optimizing equation 7 from ShuffleFL
    def optimize_transmission_matrices(self):
        # Define the objective function to optimize
        # Takes as an input the transfer matrices
        # Returns as an output the result of Equation 7
        def objective_function(x):
            # Parse args
            transfer_matrices = x.reshape((len(self.devices), math.floor(NUM_CLASSES*self.shrinkage_ratio), len(self.devices)))
            print(f"Current transfer matrices:\n {transfer_matrices}")

            # Store the current status of the devices
            current_datasets = []
            current_kd_datasets = []
            current_dataset_clusters = []
            current_kd_dataset_clusters = []
            for device in self.devices:
                current_datasets.append(device.dataset)
                current_kd_datasets.append(device.kd_dataset)
                current_dataset_clusters.append(device.dataset_clusters)
                current_kd_dataset_clusters.append(device.kd_dataset_clusters)
                # Reset the number of transferred samples for each device
                device.num_transferred_samples = 0
            
            # Transfer the data according to the matrices
            self.shuffle_data(transfer_matrices)
            # Compute the resulting system latencies and data imbalances
            latencies = self.get_latencies(epochs=1)
            data_imbalances = self.get_data_imbalances()
            print(f"Result - Latencies: {latencies}, Data imbalances: {data_imbalances}")
            # Restore the original state of the devices
            for device_idx, device in enumerate(self.devices):
                device.dataset = current_datasets[device_idx]
                device.kd_dataset = current_kd_datasets[device_idx]
                device.dataset_clusters = current_dataset_clusters[device_idx]
        
            # Compute the loss function
            # The factor of 10 was introduced to increase by an order of magnitude the importance of the time std
            # Time std is usually very small and the max time is usually very large
            # But a better approach would be to normalize the values or take the square of the std
            return STD_CORRECTION*np.std(latencies) + np.max(latencies) + self.adaptive_scaling_factor*np.max(data_imbalances)

        # Define the constraints for the optimization
        # Row sum represents the probability of data of each class that is sent
        # Sum(row) <= 1
        # Equivalent to [1 - Sum(row)] >= 0
        # Note that in original ShuffleFL the constraint is Sum(row) = 1
        # But in this case, we can use the same column as an additional dataset
        def one_minus_sum_rows(variables, num_devices, num_clusters):
            # Reshape the flat variables back to the transition matrices shape
            transition_matrices = variables.reshape((num_devices, num_clusters, num_devices))

            # Calculate the row sums ford each matrix and ensure they sum to 1
            # Because each row is the distribution of the data of a class for a device
            row_sums = []
            for matrix in transition_matrices:
                # Compute Sum(row)
                row_sum = np.sum(matrix, axis=1)
                # Now compute [1 - Sum(row)]
                row_sums.extend(1. - row_sum)
            return row_sums
        
        # Constraint to make sure that at least some elements are used for the kd dataset
        def non_zero_self_column(variables, num_devices, num_clusters):
            # Reshape the flat variables back to the transition matrices shape
            transition_matrices = variables.reshape((num_devices, num_clusters, num_devices))
            self_column = np.array([])
            for i, matrix in enumerate(transition_matrices):
                # Extract the column of the matrix that corresponds to the same device
                self_column = np.append(self_column, [row[i] for row in matrix])
            # Subtract a small value to ensure the column is non-zero
            self_column = self_column.flatten()
            return np.subtract(self_column, 10 ** -3)
        
        num_devices = len(self.devices)
        num_clusters = math.floor(NUM_CLASSES*self.shrinkage_ratio)
        num_variables = num_devices * (num_clusters * num_devices)
        # Each element in the matrix is a probability, so it must be between 0 and 1
        bounds = [(0.,1.)] * num_variables
        # If the sum is less than one, we can use same-device column as additional dataset
        constraints = [{'type': 'ineq', 'fun': lambda variables: one_minus_sum_rows(variables, num_devices, num_clusters)},
                       {'type': 'ineq', 'fun': lambda variables: non_zero_self_column(variables, num_devices, num_clusters)},]
        
        # Run the optimization
        current_transition_matrices = np.array(self.transition_matrices).flatten()
        result = minimize(objective_function,
                          x0=current_transition_matrices,
                          method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 50, 'ftol': 1e-03})
        # Update the transition matrices
        self.transition_matrices = result.x.reshape((num_devices, num_clusters, num_devices))
        print(f"Final result of optimization:\n {self.transition_matrices}")
    
    
    # Compute the difference in capability of the user compared to last round
    # Implements Equation 8 from ShuffleFL 
    def update_average_capability(self):
        # Compute current average power and bandwidth and full dataset size
        average_power = sum([device.config["compute"] for device in self.devices]) / len(self.devices)
        average_bandwidth = sum([(device.config["uplink_rate"] + device.config["downlink_rate"]) / 2 for device in self.devices]) / len(self.devices)
        
        # Equation 8 in ShuffleFL
        self.diff_capability = self.staleness_factor * (average_power / self.average_power) + (1. - self.staleness_factor) * (average_bandwidth / self.average_bandwidth)
        
        # Update the average power and bandwidth
        self.average_power = average_power
        self.average_bandwidth = average_bandwidth
    
    # Implements Equation 9 from ShuffleFL
    def compute_staleness_factor(self):
        # Compute the dataset size and number of transferred samples
        dataset_size = sum([len(device.dataset) for device in self.devices])
        num_transferred_samples = sum([device.num_transferred_samples for device in self.devices])

        # Compute the staleness factor
        self.staleness_factor = (3 * dataset_size) / ((3 * dataset_size) + num_transferred_samples)
    
    def create_kd_dataset(self):
        # Create the knowledge distillation dataset
        # The dataset is created by sampling from the devices
        # The dataset is then used to train the user model
        kd_dataset = []
        for device in self.devices:
            kd_dataset.extend(device.kd_dataset)
        self.kd_dataset = kd_dataset
