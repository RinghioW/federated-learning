import numpy as np
import matplotlib.pyplot as plt

full = [75.50924149583848,
102.97211096251003,
347.5718231109962,
109.19630006628847,
80.64146759895404,]
fair = [10.919630006628847,
10.919630006628847,
10.919630006628847,
10.919630006628847,
10.919630006628847,]
size_proportional = [30.203696598335394,
41.18884438500402,
139.02872924439848,
43.67852002651539,
32.25658703958162,]
upload_proportional = [0.6364086465588309,
0.6364086465588309,
0.6364086465588309,
0.6364086465588309,
0.6364086465588309,
]
balance_proportional = [10.915042082072464,
11.370168799503123,
23.985849330428568,
9.924801530018204,
10.91839292161442,]
adaptive = [10.919630006628847]
balanced = [4.913833502982981]
shuffle = 6.252846937015767
# Plot as a bar chart
plt.figure()
plt.bar(["Full", "Fair", "Size\nProp", "Upload\nProp", "Balance\nProp", "Adapt", "Balanced"], [full, fair, size_proportional, upload_proportional, balance_proportional, adaptive, balanced])
plt.ylabel("Latency (s)")
plt.title("Latency comparison")
# Avoid overlapping labels
# Avoid cutting off the labels
plt.tight_layout()
# Center the labels
plt.savefig("latency_comparison.svg")