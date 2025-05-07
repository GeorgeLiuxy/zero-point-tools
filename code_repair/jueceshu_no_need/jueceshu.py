import numpy as np
import csv
import time

kunming_weather = "D:\实验\毕设\数据集\kunming_weather_continus_data.txt"


def create_dataset(is_test=False):
    dataset = []
    # 打开CSV文件
    with open(kunming_weather, 'r') as file:
        reader = csv.reader(file)
        # 初始化一个空数组来存储数据
        dataset = []
        # 遍历每一行数据
        for row in reader:
            # 将每一行数据转换为数组，并添加到data数组中
            dataset.append(row)

    np.random.shuffle(dataset)  # 将数据按行充分打乱
    '''
    labels = ["MeanAirPressure",
              "DailyMinimumAirPressure",
              "MeanTempe",
              "DailyMinTempe",
              "MeanRelativeHumidity",
              "CumulativePrecipitation",
              "LargeEvaporation",
              "AverageWindSpd",
              "MaximumWindSpd",
              "SunshineDuration",
              "MeanSurfaceTempe",
              "DailyMinimumSurfaceTempe",
              "WhetherItSnows"]
    '''
    features = ["MAP",
                "DMAP",
                "MT",
                "DMT",
                "MRH",
                "CP",
                "LE",
                "AWS",
                "MWS",
                "SD",
                "MST",
                "DMST"]
    train_count = int(len(dataset) * 0.8)
    train_dataset = dataset[1: train_count]
    if (is_test):
        test_dataset = dataset[train_count:]
        test_source = []
        for data in test_dataset:
            row_data = {}
            for feature_index in range(len(features)):
                row_data[features[feature_index]] = data[feature_index]
            row_data["result"] = data[-1]
            test_source.append(row_data)
        return test_source

    return train_dataset, features


import operator


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0  # 创建节点，值置为0
        class_count[vote] += 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]


from math import log


def calcEnt(dataset):
    num_examples = len(dataset)  # 计算数据集中的样本数量
    label_counts = {}  # 创建一个字典，用来存储每个标签的数量

    # 遍历数据集中的每个样本
    for feature in dataset:
        current_label = feature[-1]  # 获取样本的标签，假设标签是样本的最后一个特征

        # 如果当前标签不在字典中，就在字典中添加这个标签，并设置数量为0
        if (current_label not in label_counts.keys()):
            label_counts[current_label] = 0

        # 将当前标签的数量加1
        label_counts[current_label] += 1

    ret_ent = 0  # 初始化熵为0

    # 遍历每个标签
    for key in label_counts:
        prop = float(label_counts[key] / num_examples)  # 计算当前标签的概率

        # 根据熵的公式，计算并累加每个标签的熵
        ret_ent += - (prop * log(prop, 2))

    return ret_ent  # 返回数据集的熵


def remove_feature_from_dataset(dataset, axis, val):
    ret_dataset = []
    for feature in dataset:
        if (feature[axis] == val):
            reduced_feature = feature[:axis]  # 取当前列前面的列
            # np.concatenate((reduced_feature, feature[axis+1:]), axis=0)
            reduced_feature.extend(feature[axis + 1:])  # 连接当前列后面的列
            ret_dataset.append(reduced_feature)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1  # 特征数量（减去标签列）
    base_entropy = calcEnt(dataset)  # 数据集的初始熵
    best_info_gain = 0  # 初始信息增益比为0
    best_feature = -1
    best_split_point = None
    for i in range(num_features):
        feature_list = [example[i] for example in dataset]  # 取当前特征对应的所有的样本值
        # feature_list = np.array([float(example[i]) for example in dataset]) #取当前特征对应的所有的样本值
        unique_vals = set(feature_list)  # 取当前特征对应的所有可能样本
        # unique_vals = np.quantile(feature_list, [0.25, 0.5, 0.75])
        for split_point in unique_vals:
            new_entropy = 0
            sub_dataset1 = [example for example in dataset if example[i] <= split_point]
            sub_dataset2 = [example for example in dataset if example[i] > split_point]
            prob1 = len(sub_dataset1) / float(len(dataset))
            new_entropy += prob1 * calcEnt(sub_dataset1)
            prob2 = len(sub_dataset2) / float(len(dataset))
            new_entropy += prob2 * calcEnt(sub_dataset2)
            info_gain = base_entropy - new_entropy  # 信息增益
            if (info_gain > best_info_gain):  # base_entropy - new_entropy > base_entropy
                best_info_gain = info_gain
                best_feature = i
                best_split_point = split_point
    return best_feature, best_split_point


def create_tree(dataset, feature_labels):
    # 提取数据集中所有样本的类别标签
    class_list = [example[-1] for example in dataset]
    # 如果所有样本的类别标签都相同，那么返回这个类别标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果数据集中只剩下类别标签，那么返回出现次数最多的类别标签
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    # 选择最佳的特征和对应的分割点
    best_feature, best_split_point = choose_best_feature_to_split(dataset)
    # 获取最佳特征的标签
    best_feature_label = feature_labels[best_feature]
    # 创建一个字典来表示决策树
    my_tree = {best_feature_label: {}}
    # 从标签列表中删除已经使用的特征标签
    feature_labels.pop(best_feature)
    # 根据最佳特征的分割点将数据集分为两部分，并从数据集中删除已经使用的特征
    sub_dataset1 = [example[:best_feature] + example[best_feature + 1:] for example in dataset if
                    example[best_feature] <= best_split_point]
    sub_dataset2 = [example[:best_feature] + example[best_feature + 1:] for example in dataset if
                    example[best_feature] > best_split_point]
    # 递归地对每一部分数据集调用 create_tree 函数
    my_tree[best_feature_label][f'<={best_split_point}'] = create_tree(sub_dataset1, feature_labels[:])
    my_tree[best_feature_label][f'>{best_split_point}'] = create_tree(sub_dataset2, feature_labels[:])
    # 返回决策树
    return my_tree


from graphviz import Digraph


def visualize_decision_tree(tree, name="DecisionTree"):
    def add_edges(graph, subtree, parent_node):
        for key, value in subtree.items():
            if isinstance(value, dict):
                sub_node = f"{parent_node}_{str(key)}"  # 将key转换为字符串
                graph.edge(parent_node, sub_node, label=str(key))  # 将label转换为字符串
                add_edges(graph, value, sub_node)
            else:
                graph.edge(parent_node, str(value), label=str(key))  # 将value和label转换为字符串

    graph = Digraph(name=name, format="png")
    graph.attr('node', shape='Mrecord')
    add_edges(graph, tree, "root")
    return graph


def predict(input_tree, test_vec, feature_labels):
    # 获取决策树的第一个节点
    first_str = list(input_tree.keys())[0]
    # 获取第一个节点的子树
    second_dict = input_tree[first_str]
    # 获取第一个节点对应的特征在特征标签列表中的索引
    # feature_index = feature_labels.index(first_str)
    feature_index = first_str
    # 遍历第一个节点的所有子节点
    for key in second_dict.keys():
        # 如果子节点的标签表示的是一个“<=”的条件
        if key[0] == '<':
            # 获取条件中的值
            value = float(key[2:])
            # 如果测试向量中对应的特征值满足这个条件
            if float(test_vec[feature_index]) <= value:
                # 如果子节点是一个字典，那么递归地调用 predict 函数
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = predict(second_dict[key], test_vec, feature_labels)
                # 否则，子节点就是一个类别标签，直接返回这个标签
                else:
                    class_label = second_dict[key]
        # 如果子节点的标签表示的是一个“>”的条件
        if key[0] == '>':
            # 获取条件中的值
            value = float(key[1:])
            # 如果测试向量中对应的特征值满足这个条件
            if float(test_vec[feature_index]) > value:
                # 如果子节点是一个字典，那么递归地调用 predict 函数
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = predict(second_dict[key], test_vec, feature_labels)
                # 否则，子节点就是一个类别标签，直接返回这个标签
                else:
                    class_label = second_dict[key]
    # 返回预测的类别标签
    return class_label


import json

if (__name__ == "__main__"):
    dataset, label = create_dataset()
    test_source = create_dataset(True)
    feature_labels = label.copy()
    train_start_time = time.time()
    my_tree = create_tree(dataset, label)
    train_end_time = time.time()
    elapsed_time = train_end_time - train_start_time
    print("train takes {} seconds".format(elapsed_time))
    test_tree = my_tree
    predict_true_count = 0
    predict_count = 0
    fail_info_array = []
    test_start_time = time.time()
    for test_item in test_source:
        predict_count += 1
        predict_result = predict(test_tree, test_item, feature_labels)
        fail_info = ''
        if (predict_result == test_item["result"]):
            predict_true_count += 1
            if (predict_true_count % 100 == 0):
                print("Test {} / {} pcs of data is true".format(predict_true_count, predict_count))
        else:
            for key, value in test_item.items():
                fail_info += "{}-{} , ".format(key, value)
            fail_info_array.append(
                "Predict result is {} actual is {} ".format(predict_result, test_item["result"]) + fail_info)
    test_end_time = time.time()
    elapsed_test_time = test_end_time - test_start_time
    print("test takes {} seconds".format(elapsed_test_time))
    for info in fail_info_array:
        print(info)
    print("Accuracy is {}".format(predict_true_count / len(test_source)))
    with open('data.json', 'w') as file:
        json.dump(my_tree, file)

    # 可视化决策树
    graph = visualize_decision_tree(my_tree)
    graph.view()  # 这将在默认的图片查看器中显示决策树图像
