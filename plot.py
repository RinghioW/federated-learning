import matplotlib.pyplot as plt
import numpy as np

full = [
    0.01,
0.0101,
0.2426,
0.3986,
0.4803,
0.4987,
0.5054,
0.5148,
0.5174,
0.5252,
0.5203,
0.5254,
0.5299,
0.527,
0.5245,
0.5285,
]

fair = [
    0.01,
0.0102,
0.0523,
0.0876,
0.0977,
0.113,
0.1173,
0.1391,
0.1515,
0.1654,
0.1762,
0.1841,
0.1992,
0.2033,
0.2143,
0.2062,
]

adaptive = [
    0.01,
0.0164,
0.1201,
0.1832,
0.226,
0.2639,
0.2904,
0.3151,
0.3321,
0.3562,
0.3737,
0.3829,
0.4035,
0.4027,
0.4333,
0.4366,
]

balanced = [
    0.01,
0.0126,
0.087,
0.1302,
0.1565,
0.1829,
0.2065,
0.2237,
0.2535,
0.2506,
0.2607,
0.2994,
0.2948,
0.2963,
0.3135,
0.3272,
]

balance_proportional = [
    0.01,
0.0198,
0.2259,
0.283,
0.3219,
0.3552,
0.3876,
0.4068,
0.4201,
0.4256,
0.4441,
0.4505,
0.4621,
0.4665,
0.4683,
0.4687,]

size_proportional = [
    0.01,
0.0148,
0.1105,
0.1777,
0.2335,
0.2692,
0.3033,
0.3239,
0.3437,
0.3589,
0.3876,
0.3995,
0.4023,
0.4266,
0.4292,
0.4393,]

upload_proportional = [
    0.01,
0.0103,
0.1357,
0.1959,
0.2533,
0.292,
0.3217,
0.3585,
0.3797,
0.3897,
0.4111,
0.4265,
0.4413,
0.4399,
0.4422,
0.4518,
]

plt.plot(full, label="Full")
plt.plot(fair, label="Fair")
plt.plot(adaptive, label="Adaptive")
plt.plot(balanced, label="Balanced")
plt.plot(balance_proportional, label="Balance Proportional")
plt.plot(size_proportional, label="Size Proportional")
plt.plot(upload_proportional, label="Upload Proportional")
plt.legend()
plt.savefig("server_accuracy.svg")