import os
import json

class Logger:

    # Epoch: id
    # |--- "Server"
    #     |--- Accuracy: float
    #     |--- Loss: float
    #     |--- Latency: float
    # |--- User: id
    #     |--- Device: id
    #         |--- Config
    #         |--- Accuracy: float
    #         |--- Loss
    #         |--- Latency
    #         |--- Data Imbalance
    #         |--- Energy Usage
    #         |--- Memory Usage
    #     |--- "Optimization"
    #       |--- Step: int
    #           |--- Latency
    #           |--- Data Imbalance
    #           |--- Objective function

    def __init__(self, log_dir):
        self.log_dir = log_dir

        self.epoch = -1
        self.log = {self.epoch : {"server" : {}}}

    def dump(self):
        with open(os.path.join(self.log_dir, "log.json"), "w") as f:
            json.dump(self.log, f)

    def new_epoch(self, n_users):
        self.epoch += 1
        self.log[self.epoch] = {"server": {}, "users": {i: {} for i in range(n_users)}}

    # Server
    def s_log_test(self, accuracy, loss):
        self.log[self.epoch]["server"]["accuracy"] = accuracy 
        self.log[self.epoch]["server"]["loss"] = loss
    

    def s_log_latency(self, latency):
        self.log[self.epoch]["server"]["latency"] = latency


    # User
    def u_log_configs(self, user_id, configs):
        self.log[self.epoch]["users"][user_id]["devices"] = {i: {} for i in range(len(configs))}
        for device_id, config in enumerate(configs):
            self.log[self.epoch]["users"][user_id]["devices"][device_id]["config"] = config

    def u_log_test(self, user_id, accuracy):
        self.log[self.epoch]["users"][user_id]["test_accuracy"] = accuracy

    def u_log_train(self, user_id, kd_loss, ce_loss, accuracy):
        self.log[self.epoch]["users"][user_id]["train_kd_loss"] = kd_loss
        self.log[self.epoch]["users"][user_id]["train_ce_loss"] = ce_loss
        self.log[self.epoch]["users"][user_id]["train_accuracy"] = accuracy

    def u_log_devices_test(self, user_id, accuracies):
        for device_id, accuracy in enumerate(accuracies):
            self.log[self.epoch]["users"][user_id]["devices"][device_id]["accuracy"] = accuracy

    def u_log_latencies(self, user_id, latencies):
        for device_id, latency in enumerate(latencies):
            self.log[self.epoch]["users"][user_id]["devices"][device_id]["latency"] = latency

    def u_log_data_imbalances(self, user_id, imbalances):
        for device_id, imbalance in enumerate(imbalances):
            self.log[self.epoch]["users"][user_id]["devices"][device_id]["data_imbalance"] = imbalance

    def u_log_energies(self, user_id, energies):
        for device_id, energy in enumerate(energies):
            self.log[self.epoch]["users"][user_id]["devices"][device_id]["energy_usage"] = energy

    def u_log_memories(self, user_id, memories):
        for device_id, memory in enumerate(memories):
            self.log[self.epoch]["users"][user_id]["devices"][device_id]["memory_usage"] = memory

    def u_log_optimization(self, user_id, objective, latency, data_imbalance):
        if "optimization" not in self.log[self.epoch]["users"][user_id]:
            self.log[self.epoch]["users"][user_id]["optimization"] = {"objective": [], "latency": [], "data_imbalance": []}
        self.log[self.epoch]["users"][user_id]["optimization"]["objective"].append(objective)
        self.log[self.epoch]["users"][user_id]["optimization"]["latency"].append(latency)
        self.log[self.epoch]["users"][user_id]["optimization"]["data_imbalance"].append(data_imbalance)




