import torch
import torch.nn as nn
import numpy as np
from config import DEVICE, NUM_CLASSES, STD_CORRECTION
from scipy.optimize import minimize
import math
import random
import torchvision
import datasets
from adaptivenet import AdaptiveNet
from config import Style
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from copy import deepcopy
class User():
    def __init__(self, devices, classes=NUM_CLASSES) -> None:
        self.devices = devices
        self.kd_dataset = []
        self.model = None
        self.model_state_dict = None
        self.optimizer_state_dict = None

        # SHUFFLE-FL

        # Transition matrix of ShuffleFL of size (floor{number of classes * shrinkage ratio}, number of devices + 1)
        # The additional column is for the kd_dataset
        # Also used by equation 7 as the optimization variable for the argmin
        # Shrinkage ratio for reducing the classes in the transition matrix
        self.shrinkage_ratio = 0.3
        initial_transition_matrices = np.random.rand(len(devices), math.floor(classes*self.shrinkage_ratio), len(devices))
        sums = np.sum(initial_transition_matrices, axis=2)
        self.transition_matrices = initial_transition_matrices / sums[:, :, np.newaxis]
        
        # System latencies for each device
        self.adaptive_scaling_factor = 1.0

        # Staleness factor
        self.staleness_factor = 0.0

        # Average capability beta
        self.diff_capability = 1. + random.random()
        self.average_power = 1. + random.random()
        self.average_bandwidth = 1. + random.random()


    # Adapt the model to the devices
    # Implements the adaptation step from ShuffleFL Novelty
    # Constructs a function s.t. device_model = f(user_model, device_resources, device_data_distribution)
    def adapt_model(self, model):
        self.model = model
        for idx, device in enumerate(self.devices):
            # If compute is low, better to quantize the network
            # If memory is low, better to prune the network
            # If communication is low, does not really matter as to the network
            # If energy is low, better to prune the network
            resources = device.config["compute"] + device.config["memory"] + device.config["energy_budget"]
            # Adaptation is based on the device resources
            if resources < 5:
                device.model = AdaptiveNet(quantize=True)
                device.model.qconfig = torch.quantization.default_qconfig
                torch.quantization.prepare(device.model, inplace=True)
            elif resources < 10:
                device.model = AdaptiveNet(pruning_factor=0.5)
            elif resources < 20:
                device.model = AdaptiveNet(pruning_factor=0.3)
            else:
                device.model = AdaptiveNet(pruning_factor=0.)
            self.devices[idx] = device
        return self

    # Train the user model using knowledge distillation
    def aggregate_updates(self, learning_rate=0.001, epochs=10, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75):
        student = self.model
        if self.model_state_dict is not None:
            student.load_state_dict(self.model_state_dict)

        # TODO : Train in parallel, not sequentially (?)
        teachers = [device.model for device in self.devices]
        to_tensor = torchvision.transforms.ToTensor()
        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)
        student.train() # Student to train mode
        for teacher_idx, teacher in enumerate(teachers):
            train_loader = torch.utils.data.DataLoader(self.kd_dataset[teacher_idx].map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch"), shuffle=True, drop_last=True)
            teacher.eval()
            for epoch in range(epochs):
                running_loss = 0.0
                running_kd_loss = 0.0
                running_ce_loss = 0.0
                running_accuracy = 0.0
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
                    soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

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

                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader.dataset)} (KD: {running_kd_loss / len(train_loader.dataset)}, CE: {running_ce_loss / len(train_loader.dataset)}), Accuracy: {running_accuracy / len(train_loader.dataset)}")

        self.model = student
        self.model_state_dict = deepcopy(self.model.state_dict())
        self.optimizer_state_dict = deepcopy(optimizer.state_dict())
        return self

    # Train all the devices belonging to the user
    # Steps 11-15 in the ShuffleFL Algorithm
    def train_devices(self, epochs=5, verbose=True):
        for device_idx, device in enumerate(self.devices):
            device = device.train(epochs, verbose)
            self.devices[device_idx] = device
        return self
    
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

        # Sender removes some samples
        sender, samples, clusters = sender.remove_data(cluster, percentage_amount)
        # Receiver adds those samples
        receiver = receiver.add_data(samples, clusters)

        self.devices[sender_idx] = sender
        self.devices[receiver_idx] = receiver

        return self
    
    def mock_send_data(self, sender_idx, receiver_idx, cluster, percentage_amount):
        # Identify sender and receiver
        sender = self.devices[sender_idx]
        receiver = self.devices[receiver_idx]

        # Sender removes some samples
        sender, clusters = sender.mock_remove_data(cluster, percentage_amount)
        # Receiver adds those samples
        receiver = receiver.mock_add_data(clusters)

        self.devices[sender_idx] = sender
        self.devices[receiver_idx] = receiver
        return self

    def mock_get_data_imbalances(self):
        data_imbalances = []
        for device in self.devices:
            data_imbalances.append(device.mock_data_imbalance())
        return data_imbalances
    
    def mock_get_latencies(self, transition_matrices):
        t_communication = [0.] * len(self.devices)
        t_computation = [0.] * len(self.devices)
        latencies = [0.] * len(self.devices)
        
        # For each device, compute communication and computation time
        for device_idx, device in enumerate(self.devices):
            transition_matrix = transition_matrices[device_idx]
            # Compute the communication time
            for data_class_idx, _ in enumerate(transition_matrix):
                for other_device_idx, other_device in enumerate(self.devices):
                    other_transition_matrix = transition_matrices[other_device_idx]
                    if device_idx != other_device_idx:
                        # Transmitting
                        num_transmitted_samples = len(device.dataset_clusters) * transition_matrix[data_class_idx][other_device_idx]
                        t_communication[device_idx] += num_transmitted_samples * ((1/device.config["uplink_rate"]) + (1/other_device.config["downlink_rate"]))
                        # Receiving
                        num_received_samples = len(other_device.dataset_clusters) * other_transition_matrix[data_class_idx][device_idx]
                        t_communication[device_idx] += num_received_samples * ((1/device.config["downlink_rate"]) + (1/other_device.config["uplink_rate"]))
            # Compute the computation time
            t_computation[device_idx] += 3 * len(device.dataset_clusters) * device.config["compute"]
            
            # Compute the latency as the sum of both
            latencies[device_idx] = t_communication[device_idx] + t_computation[device_idx]
        return latencies
    


    # Mock function to shuffle data between devices
    # Only use the dataset_clusters to simulate the data shuffling
    # Does not actually send the data
    def mock_shuffle_data(self, transition_matrices):
        for device_idx, transition_matrix in enumerate(transition_matrices):
            for cluster_idx in range(len(transition_matrix)):
                for other_device_idx in range(len(transition_matrix[0])):
                    if other_device_idx != device_idx:
                        # Send data from cluster i to device j
                        self = self.mock_send_data(sender_idx=device_idx, receiver_idx=other_device_idx, cluster=cluster_idx, percentage_amount=transition_matrix[cluster_idx][other_device_idx])

        # Compute the latencies
        latencies = self.mock_get_latencies(transition_matrices)
        # Compute the data imbalances
        data_imbalances = self.mock_get_data_imbalances()
        return latencies, data_imbalances

    # Shuffle data between devices according to the transition matrices
    # Implements the transformation described by Equation 1 from ShuffleFL
    def shuffle_data(self, transition_matrices):
        # Each device sends data according to the respective transition matrix
        for device_idx, transition_matrix in enumerate(transition_matrices):
            for cluster_idx in range(len(transition_matrix)):
                for other_device_idx in range(len(transition_matrix[0])):
                    if other_device_idx != device_idx:
                        # Send data from cluster i to device j
                        self = self.send_data(sender_idx=device_idx, receiver_idx=other_device_idx, cluster=cluster_idx, percentage_amount=transition_matrix[cluster_idx][other_device_idx])
        return self
    # Function to implement the dimensionality reduction of the transition matrices
    # The data is embedded into a 2-dimensional space using t-SNE
    # The classes are then aggregated into k groups using k-means
    # Implements section 4.4 from ShuffleFL
    def reduce_dimensionality(self):
        for idx, device in enumerate(self.devices):
            device = device.cluster_data(self.shrinkage_ratio)
            self.devices[idx] = device
        return self
    
    def user_reduce_dimensionality(self):
        self.user_cluster_data(self.shrinkage_ratio)
        return self

    # Function for optimizing equation 7 from ShuffleFL
    def optimize_transmission_matrices(self):
        # Define the objective function to optimize
        # Takes as an input the transfer matrices
        # Returns as an output the result of Equation 7
        def objective_function(x):
            # Parse args
            transfer_matrices = x.reshape((len(self.devices), math.floor(NUM_CLASSES*self.shrinkage_ratio), len(self.devices)))
            
            # Save the state of the clusters of each device
            devices_dataset_clusters = [device.dataset_clusters for device in self.devices]
                
            # Simulate the transferring of the data according to the matrices
            latencies, data_imbalances = self.mock_shuffle_data(transfer_matrices)

            # Restore the state of the clusters of each device
            for idx, device in enumerate(self.devices):
                device.dataset_clusters = devices_dataset_clusters[idx]
                device.num_transferred_samples = 0
                self.devices[idx] = device

            # Compute the loss function
            # The factor of STD_CORRECTION was introduced to increase by an order of magnitude the importance of the time std
            # Time std is usually very small and the max time is usually very large
            # But a better approach would be to normalize the values or take the square of the std
            system_latency = STD_CORRECTION*np.std(latencies) + np.max(latencies)
            data_imbalance = self.adaptive_scaling_factor*np.max(data_imbalances)
            obj_func = system_latency + data_imbalance
            return obj_func

        # Define the constraints for the optimization
        # Row sum represents the probability of data of each class that is sent
        # Sum(row) <= 1
        # Equivalent to [1 - Sum(row)] >= 0
        # Note that in original ShuffleFL the constraint is Sum(row) = 1
        # But in this case, we can use the same column as an additional dataset
        def one_minus_sum_rows(variables, num_devices, num_clusters):
            # Reshape the flat variables back to the transition matrices shape
            transition_matrices = variables.reshape((num_devices, num_clusters, num_devices))
            return (1. - np.sum(transition_matrices, axis=2)).flatten()
        
        num_devices = len(self.devices)
        num_clusters = math.floor(NUM_CLASSES*self.shrinkage_ratio)
        num_variables = num_devices * (num_clusters * num_devices)
        # Each element in the matrix is a probability, so it must be between 0 and 1
        bounds = [(0.,1.)] * num_variables
        # If the sum is less than one, we can use same-device column as additional dataset
        # constraints = [{'type': 'ineq', 'fun': lambda variables: one_minus_sum_rows(variables, num_devices, num_clusters)}]
        constraints = [{'type': 'ineq', 'fun': lambda variables: one_minus_sum_rows(variables, num_devices, num_clusters)}]
        
        # Run the optimization
        current_transition_matrices = np.array(self.transition_matrices).flatten()
        result = minimize(objective_function,
                          x0=current_transition_matrices,
                          method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 50, 'ftol': 1e-01, 'eps': 1e-01})
        # Update the transition matrices
        if not result.success:
            print(f"{Style.RED}[ERROR]{Style.RESET} Optimization did not converge after {result.nit} iterations. Status: {result.status} Message: {result.message}")
        self.transition_matrices = result.x.reshape((num_devices, num_clusters, num_devices))
        return self
    
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
        return self
    # Implements Equation 9 from ShuffleFL
    def compute_staleness_factor(self):
        # Compute the dataset size and number of transferred samples
        dataset_size = sum([len(device.dataset) for device in self.devices])
        num_transferred_samples = sum([device.num_transferred_samples for device in self.devices])

        # Compute the staleness factor
        self.staleness_factor = (3 * dataset_size) / ((3 * dataset_size) + num_transferred_samples)
        return self
    
    def create_kd_dataset(self, percentage_amount=0.5):
        # Create the knowledge distillation dataset
        # The dataset is created by sampling from the devices
        # The dataset is then used to train the user model
        kd_dataset = [None for _ in range(len(self.devices))]
        for device_idx, device in enumerate(self.devices):
            dataset = np.array(device.dataset)
            dataset_idxs = np.random.choice(a=dataset.shape[0], size=math.floor(percentage_amount*len(dataset)), replace=False)
            kd_dataset[device_idx] = datasets.Dataset.from_list([dataset[idx] for idx in dataset_idxs])
        self.kd_dataset = kd_dataset
        dataset_len = sum(len(kd_dataset[i]) for i in range(len(kd_dataset)))
        print(f"{Style.GREEN}[INFO]{Style.RESET} Knowledge distillation dataset created with number of samples: {dataset_len}")
        return self
    
    def user_cluster_data(self, shrinkage_ratio):
        # Assemble the entire dataset from the devices
        dataset = np.array(self.devices[0].dataset)
        for device in self.devices[1:]:
            dataset = np.append(dataset, device.dataset, axis=0)
        dataset = datasets.Dataset.from_list(dataset.tolist())

        feature_space = np.array(dataset["img"]).reshape(len(dataset), -1)
        feature_space_2D = TSNE(n_components=2).fit_transform(feature_space)
        # Cluster datapoints to k classes using KMeans
        n_clusters = math.floor(shrinkage_ratio*NUM_CLASSES)
        dataset_clusters = KMeans(n_clusters).fit_predict(feature_space_2D)

        # Assign the a partition clusters to the devices
        idx = 0
        for device_idx, device in enumerate(self.devices):
            # Assign the cluster partition to the devices
            device.dataset_clusters = dataset_clusters[idx:idx+len(device.dataset)]
            idx+=len(device.dataset)
            self.devices[device_idx] = device
        return self
