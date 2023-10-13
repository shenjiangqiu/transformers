# %%
import matplotlib.pyplot as plt
import numpy as np

import torch


def plot_tree(data, filename):
    width = len(data)
    height = len(data[0])
    # find all node that is in the final path
    all_node_in_path = torch.zeros(width, height + 1)
    all_node_in_path[:, height] = 1
    current_frontier = [i for i in range(width)]
    current_iter = height - 1
    while current_iter >= 0:
        selected_node = data[current_frontier, current_iter]
        all_node_in_path[selected_node, current_iter] = 1

        current_frontier = selected_node.tolist()
        current_iter -= 1

    plt.figure(figsize=(width, height))
    for i, node in enumerate(data):
        for j, parent_node in enumerate(node):
            if parent_node != -1:
                if all_node_in_path[parent_node][j] == 1 and all_node_in_path[i][j + 1] == 1:
                    plt.plot([parent_node, i], [j, j + 1], "ro-")
                else:
                    plt.plot([parent_node, i], [j, j + 1], "ko-")

    plt.show()
    # plt.savefig("test.png")
    plt.savefig(filename)


def plot_source_tree(data, filename):
    width = len(data)
    height = len(data[0])
    plt.figure(figsize=(width, height))
    for i, node in enumerate(data):
        node = node[: torch.where(node == -1)[0][0]]
        plt.plot(node, [i for i in range(len(node))], "ko-")
    plt.show()
    # plt.savefig("test.png")
    plt.savefig(filename)


# %%
