import matplotlib.pyplot as plt
import numpy as np


# Span, height and loads are inputs
#----
span = 16
height = (1.20)


# Number of trusses, 1 pr m of truss system for now
num_trusses = span
#
num_lower_nodes = num_trusses/2
num_upper_nodes = num_lower_nodes + 1

length_upper_seg = span / num_lower_nodes
length_ls = length_upper_seg

first_low = [length_ls/2, -height]
last_low = [span-length_ls/2, -height]


lower_nodes = []
lower_nodes.append(first_low)
upper_nodes = []
upper_nodes = [[0, 0]]


incr = length_upper_seg
delta_lower = incr/2
delta_upper = 0
for i in range(1, int(num_lower_nodes)):
    delta_lower += incr
    delta_upper += incr
    upper_nodes.append([round(delta_upper, 2), 0])
    lower_nodes.append([round(delta_lower,2), -height])

upper_nodes.append([span,0])
print(upper_nodes)
print(lower_nodes)

all_nodes =[]
print(len(upper_nodes))
for i in range(1, len(upper_nodes)+1):

    if i >= len(upper_nodes):
        all_nodes.append(upper_nodes[i-1])
    else:
        all_nodes.append(upper_nodes[i-1])
        all_nodes.append(lower_nodes[i-1])

print(all_nodes)
plt.scatter(*zip(*upper_nodes))
plt.plot(*zip(*upper_nodes))
plt.scatter(*zip(*lower_nodes))
plt.plot(*zip(*lower_nodes))
plt.plot(*zip(*all_nodes))
plt.show()