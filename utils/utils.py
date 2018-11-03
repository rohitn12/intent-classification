import logging
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

models_folder = "models"
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(filename="logs/common_log.log", filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())


def read_data(filename):
    lines, intent = [], []
    with open(filename, encoding="ISO-8859-1") as f:
        for line in f:
            line = line.strip()
            intent.append(line.split(" ")[0])
            lines.append(line.split(" ", 1)[1])
    return lines, intent


def divide_data_by_main_class(data_list, full_labels_list):
    class_wise_data = defaultdict(list)
    class_wise_labels = defaultdict(list)
    for i in range(len(data_list)):
        curr_data = data_list[i]
        curr_full_label = full_labels_list[i]
        class_wise_data[curr_full_label.split(":")[0]].append(curr_data)
        class_wise_labels[curr_full_label.split(":")[0]].append(curr_full_label.split(":")[1])
    return class_wise_data, class_wise_labels


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                  pos=None, parent=None):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
       pos: a dict saying where all nodes go if they have been assigned
       parent: parent of this branch.'''
    if pos == None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = list(G.neighbors(root))
    if parent != None:  # this should be removed for directed graphs.
        neighbors.remove(parent)  # if directed, then parent not in neighbors.
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos,
                                parent=root)
    return pos


def hierarchy_plot(primary_classes):
    G = nx.Graph()
    question_data_point = "Question input\n(Data Point to run the prediction on)"
    base_classifier = "Primary Classifier (SVM)\n(predicts primary class)\n"
    nb_classifier_string = "Subclass\nClassifier\n(NB)"

    G.add_edges_from([(question_data_point, base_classifier)])
    for each_main_class in primary_classes:
        G.add_edges_from([(base_classifier, nb_classifier_string + "\n" + each_main_class)])

    nx.draw(G, pos=hierarchy_pos(G, question_data_point), with_labels=True, node_color='w')

    plt.savefig("classification_hierarchy.png", format="PNG")


def load_available_model():
    check_for_models = os.listdir(models_folder)
    all_models = {}
    for each_model in check_for_models:
        all_models[each_model.split("_classifier")[0]] = pickle.load(
            open(os.path.join(models_folder, each_model), 'rb'))
    return all_models
