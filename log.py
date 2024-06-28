import os
import json
import time

class Logger:
    def __init__(self, name, log_dir) -> None:
        self.name = name
        self.log_dir = log_dir
        # Server
        self.s_accuracies = {}
        self.s_latencies = {}
        # User
        self.u_accuracies = {}
        self.u_losses = {}
        self.u_latencies = {}
        self.u_data_imbalances = {}
        self.u_energy_usages = {}
        self.u_memory_usages = {}
        self.u_optimizations = {}

        # Device
        self.d_computation_time = {}
        self.d_memory_usage = {}
        self.d_energy_usage = {}
        self.d_latency = {}
        self.d_data_imbalance = {}



    # Server
    def s_log_accuracy(self, epoch, accuracy):
        self.s_accuracies[epoch] = accuracy

    def s_log_latency(self, epoch, latency):
        self.s_latencies[epoch] = latency

    def s_dump(self):
        with open(os.path.join(self.log_dir, "s_accuracies.json"), "w") as f:
            json.dump(self.accuracies, f)

        with open(os.path.join(self.log_dir, "s_latencies.json"), "w") as f:
            json.dump(self.latencies, f)



    # User
    def u_log_accuracy(self, user_id, epoch, accuracy):
        if user_id not in self.accuracies:
            self.accuracies[user_id] = {}
        self.accuracies[user_id][epoch] = accuracy

    def u_log_loss(self, user_id, epoch, kd_loss, ce_loss):
        if user_id not in self.losses:
            self.losses[user_id] = {}
        self.losses[user_id][epoch] = {'kd_loss': kd_loss, 'ce_loss': ce_loss}

    def u_log_latency(self, user_id, epoch, latency):
        if user_id not in self.latencies:
            self.latencies[user_id] = {}
        self.latencies[user_id][epoch] = latency

    def u_log_data_imbalance(self, user_id, epoch, imbalance):
        if user_id not in self.data_imbalances:
            self.data_imbalances[user_id] = {}
        self.data_imbalances[user_id][epoch] = imbalance

    def u_log_energy_usage(self, user_id, epoch, energy):
        if user_id not in self.energy_usages:
            self.energy_usages[user_id] = {}
        self.energy_usages[user_id][epoch] = energy

    def u_log_memory_usage(self, user_id, epoch, memory):
        if user_id not in self.memory_usages:
            self.memory_usages[user_id] = {}
        self.memory_usages[user_id][epoch] = memory

    def u_log_optimization(self, user_id, epoch, optimization):
        if user_id not in self.u_optimizations:
            self.optimizations[user_id] = {}
        self.optimizations[user_id][epoch] = optimization

    # Device
    def d_log_computation_time(self, device_id, time):
        pass

    def d_log_memory_usage(self, device_id, memory):
        pass


    def d_log_energy_usage(self, device_id, energy):
        pass

    def d_log_latency(self, device_id, latency):
        pass

    def d_log_data_imbalance(self, device_id, imbalance):
        pass
