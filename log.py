import os
import json
import time

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir

        self.server_log = {}
        self.users_log = {}

        self.epoch = 0
        self.server_log[self.epoch] = {}
        self.users_log[self.epoch] = {}

    def update_epoch(self, n_users):
        self.epoch += 1
        self.server_log[self.epoch] = {}
        self.users_log[self.epoch] = { i: {} for i in range(n_users) }

    # Server
    # Epoch
    # |--- Accuracy
    # |--- Loss

    def s_log_accuracy(self, accuracy):
        self.server_log[self.epoch]["accuracy"] = accuracy 
    
    def s_log_loss(self, loss):
        self.server_log[self.epoch]["loss"] = loss

    def s_log_latency(self, latency):
        self.s_latencies[self.epoch]["latency"] = latency

    def s_dump(self):
        with open(os.path.join(self.log_dir, "server_log.json"), "w") as f:
            json.dump(self.server_log, f)

    # Epoch
    # |--- User
    #     |--- Device
    #         |--- Accuracy
    #         |--- Loss
    #         |--- Latency
    #         |--- Data Imbalance
    #         |--- Energy Usage
    #         |--- Memory Usage
    #         |--- Optimization

    def u_log_accuracy(self, user_id, epoch, accuracy):
        self.users_log[epoch][user_id]["accuracy"] = accuracy

    def u_log_loss(self, user_id, kd_loss, ce_loss):
        self.users_log[self.epoch][user_id]["kd_loss"] = kd_loss
        self.users_log[self.epoch][user_id]["ce_loss"] = ce_loss

    def u_log_latency(self, user_id, latency):
        self.latencies[user_id][self.epoch] = latency

    def u_log_data_imbalance(self, user_id, imbalance):
        self.users_log[self.epoch][user_id]["data_imbalance"] = imbalance

    def u_log_energy_usage(self, user_id, energy):
        self.users_log[self.epoch][user_id]["energy_usage"] = energy

    def u_log_memory_usage(self, user_id, memory):
        self.users_log[self.epoch][user_id]["memory_usage"] = memory

    def u_log_optimization(self, user_id, optimization):
        self.users_log[self.epoch][user_id]["optimization"] = optimization

    def u_dump(self):
        with open(os.path.join(self.log_dir, "users_log.json"), "w") as f:
            json.dump(self.users_log, f)
