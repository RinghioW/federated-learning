import torch
import numpy as np
import math
import torchvision
import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from optimize import optimize_transmission_matrices
from shuffle import shuffle_data
from statistics import fmean

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

class User():
    def __init__(self, id, devices, testset, n_classes=NUM_CLASSES) -> None:
        self.id = id
        self.devices = devices
        self.kd_dataset: datasets.Dataset = None
        self.model = None

        # SHUFFLE-FL

        # Transition matrix of ShuffleFL of size (floor{number of classes * shrinkage ratio}, number of devices + 1)
        # The additional column is for the kd_dataset
        # Also used by equation 7 as the optimization variable for the argmin
        # Shrinkage ratio for reducing the classes in the transition matrix
        self.shrinkage_ratio = 0.3

        # System latencies for each device
        self.adaptive_scaling_factor = 1.0

        # Staleness factor
        self.staleness_factor = 0.0

        # Average capability beta
        self.average_power = 1.
        self.average_bandwidth = 1.
        self.n_transferred_samples = 0

        self.init = False
        self.testset = testset
        self.training_losses = []
        self.training_accuracies = []
        self.test_accuracies = []


    def __repr__(self) -> str:
        return f"User(id: {self.id}, devices: {self.devices})"

    # Adapt the model to the devices
    # Implements the adaptation step from ShuffleFL Novelty
    # Constructs a function s.t. device_model = f(user_model, device_resources, device_data_distribution)
    def _adapt_model(self, model):
        # User gets the same model as the server
        self.model = model

        # Devices adapt the user model according to their capabilities
        state_dict = torch.load("checkpoints/server.pt")["model_state_dict"]


        # State dict is reduced for adapting the layers

        for device in self.devices:
            resources = device.resources()

            if resources < 30:
                params = {"quantize": True, "pruning_factor": 0.}
            elif resources < 40:
                params = {"quantize": False, "pruning_factor": 0.5}
            elif resources < 50:
                params = {"quantize": False, "pruning_factor": 0.3}
            else:
                params = {"quantize": False, "pruning_factor": 0.1}
            
            device.adapt(model, state_dict, params)
    
    # Train the user model using knowledge distillation
    def _aggregate_updates(self, epochs, learning_rate=0.0001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75):
        student = self.model()
        checkpoint = torch.load("checkpoints/server.pt")
        student.load_state_dict(checkpoint['model_state_dict'])
        student.train()
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

        # TODO : Test 2 options
        # option 1 -> each device acts as a teacher separately
        # option 2 -> average the teacher logits from all devices
        teachers = [] 
        for device in self.devices:
            teacher = device.model()
            checkpoint = torch.load(f"checkpoints/device_{device.config['id']}.pt")
            teacher.load_state_dict(checkpoint['model_state_dict'], strict=False, assign=True)
            teacher.eval()
            teachers.append(teacher)
        
        to_tensor = torchvision.transforms.ToTensor()
        train_loader = torch.utils.data.DataLoader(self.kd_dataset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch"), shuffle=True, drop_last=True, batch_size=32, num_workers=3)
        ce_loss = torch.nn.CrossEntropyLoss()
        
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
            self.training_losses.append(running_loss / len(train_loader.dataset))
            self.training_accuracies.append(running_accuracy / len(train_loader.dataset))
        
        # Save the model for checkpointing
        torch.save({'model_state_dict': student.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f"checkpoints/user_{self.id}.pt")

    # Train all the devices belonging to the user
    # Steps 11-15 in the ShuffleFL Algorithm
    def train(self, kd_epochs, on_device_epochs):
        # Adapt the model
        self._adapt_model(self.model)

        n_transferred_samples = self._shuffle()
        self.n_transferred_samples = n_transferred_samples

        # Train the devices
        for device in self.devices:
            device.train(on_device_epochs)
            print(f"Device {device.config['id']} validation accuracy: {device.validate():.3f}")
        
        # Create the knowledge distillation dataset
        self._create_kd_dataset()

        # Aggregate the updates
        self._aggregate_updates(epochs=kd_epochs)


    # Implements section 4.4 from ShuffleFL
    def _reduce_dimensionality(self):
        lda_estimator, kmeans_estimator = self._compute_centroids()
        for device in self.devices:
            device = device.cluster(lda_estimator, kmeans_estimator)

    def _shuffle(self):
        # Reduce dimensionality of the transmission matrices
        # ShuffleFL step 7, 8
        self._reduce_dimensionality()

        # User optimizes the transmission matrices
        # ShuffleFL step 9
        adaptive_scaling_factor = self.adaptive_scaling_factor
        cluster_distributions = [device.cluster_distribution() for device in self.devices]
        uplinks = [device.config["uplink_rate"] for device in self.devices]
        downlinks = [device.config["downlink_rate"] for device in self.devices]
        computes = [device.config["compute"] for device in self.devices]
        transition_matrices = optimize_transmission_matrices(adaptive_scaling_factor, cluster_distributions, uplinks, downlinks, computes)

        # Shuffle the data and update the transition matrices
        # Implements Equation 1 from ShuffleFL
        datasets = [device.dataset for device in self.devices]
        clusters = [device.clusters for device in self.devices]
        res_datasets, n_transferred_samples = shuffle_data(datasets, clusters, cluster_distributions, transition_matrices)

        # Update average capability
        # Update the devices with the new datasets
        for device, dataset in zip(self.devices, res_datasets):
            device.dataset = dataset
        
        return n_transferred_samples

    # Compute the difference in capability of the user compared to last round
    # Implements Equation 8 from ShuffleFL 
    def diff_capability(self):
        if not self.init:
            return 1.0
        # Compute current average power and bandwidth and full dataset size
        avg_power = fmean([device.config["compute"] for device in self.devices])
        avg_bandwidth = fmean([fmean([device.config["uplink_rate"], device.config["downlink_rate"]]) for device in self.devices])
        
        # Equation 8 in ShuffleFL
        staleness_factor = self._staleness_factor()
        prev_avg_power = self.average_power
        prev_avg_bandwidth = self.average_bandwidth

        diff_capability = fmean(data=[avg_power/prev_avg_power, avg_bandwidth/prev_avg_bandwidth], weights=[staleness_factor, 1-staleness_factor])
        
        # Update the average power and bandwidth
        self.average_power = avg_power
        self.average_bandwidth = avg_bandwidth

        return diff_capability
    
    # Implements Equation 9 from ShuffleFL
    def _staleness_factor(self):
        dataset_size = sum([len(device.dataset) for device in self.devices])
        data_processed = 3 * dataset_size

        # Compute the staleness factor
        return data_processed / (data_processed + self.n_transferred_samples)
    
    def _create_kd_dataset(self, percentage=0.2):
        self.kd_dataset = self._sample_devices(percentage)
    
    # TODO: Have as parameter the uplink rate of the devices
    def _compute_centroids(self):
        dataset = self._sample_devices(.1)
        features = np.array(dataset["img"]).reshape(len(dataset), -1)
        labels = np.array(dataset["label"])

        lda = LinearDiscriminantAnalysis(n_components=4).fit(features, labels)
        feature_space = lda.transform(features)
        # Cluster datapoints to k classes using KMeans
        n_clusters = math.floor(self.shrinkage_ratio*NUM_CLASSES)
        kmeans = KMeans(n_clusters=n_clusters).fit(feature_space)

        return lda, kmeans
    
    def _sample_devices(self, percentage):
        # Assemble the entire dataset from the devices
        dataset = []
        for device in self.devices:
            dataset.extend(device.sample(percentage))
        return datasets.Dataset.from_list(dataset)

    def n_samples(self):
        return len(self.kd_dataset)

    def validate(self):
        net = self.model()
        checkpoint = torch.load(f"checkpoints/user_{self.id}.pt")
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        to_tensor = torchvision.transforms.ToTensor()
        dataset = self.testset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        valloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=3)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in valloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = net(images)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                total += labels.size(0)
        self.test_accuracies.append(correct / total)
        return correct / total
