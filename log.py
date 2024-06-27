import csv
import os
import time

class Logger:

    def __init__(self):
        self.devices = []

    # Create a log file
    log_file = open("log.csv", "w")
    log_writer = csv.writer(log_file)

    def dump(self):
        with open("log.csv", "w") as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(["device", "computation_time", "memory_usage", "energy_usage", "latency", "data_imbalance"])

    def plot(self):
        pass
    
    def log(device):
        pass

    def log_computation_time(device):
        pass

    def log_memory_usage(device):
        pass


    def log_energy_usage(device):
        pass

    def log_latency(device):
        pass

    def log_data_imbalance(device):
        pass
