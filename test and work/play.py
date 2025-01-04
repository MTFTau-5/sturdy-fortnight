import networkx as nx
import pickle


def build_graphs_from_pkl(pkl_file_path):
    graphs = []
    with open(pkl_file_path, 'rb') as f:
        all_data = pickle.load(f)

        # 假设之前定义的设备、标签、信噪比的映射字典如下（如果实际有变动可相应修改）
        devicem = {'fan': 0, 'pump': 1, 'slider': 2, 'valve': 3}
        labelm = {'abnormal': 1, 'normal': 0}
        snrm = {'-6dB': -1, '0dB': 0, '6dB': 1}

        # 检查all_data结构是否符合预期，避免后续不必要的循环报错
        if not isinstance(all_data, list):
            print("all_data 的数据结构不符合预期，应该是一个列表，请检查data.pkl文件内容")
            return graphs

        for snr_idx, snr_data in enumerate(all_data):
            # 处理获取snr_name时可能出现的索引问题
            snr_list = [k for k, v in snrm.items() if v == snr_idx]
            if snr_list:
                snr_name = snr_list[0]
            else:
                print(f"snr_idx {snr_idx} 在 snrm 字典中找不到对应项，跳过此次循环")
                continue

            # 检查snr_data结构是否符合预期，应为可迭代对象（如列表）包含音频特征等信息
            if not hasattr(snr_data, '__iter__'):
                print(f"snr_data（对应索引 {snr_idx} ）的数据结构不符合预期，应为可迭代对象，请检查data.pkl文件内容")
                continue

            for mfcc_data, device_idx, label_idx in snr_data:
                graph = nx.Graph()
                # 处理获取device_name时可能出现的索引问题
                device_list = [k for k, v in devicem.items() if v == device_idx]
                if device_list:
                    device_name = device_list[0]
                    graph.add_node(device_name, type='device')
                else:
                    print(f"device_idx {device_idx} 在 devicem 字典中找不到对应项，跳过此次数据处理")
                    continue

                # 处理获取label_name时可能出现的索引问题
                label_list = [k for k, v in labelm.items() if v == label_idx]
                if label_list:
                    label_name = label_list[0]
                    graph.add_node(label_name, type='label')
                else:
                    print(f"label_idx {label_idx} 在 labelm 字典中找不到对应项，跳过此次数据处理")
                    continue

                # 使用映射后的标签构建节点，格式如 "设备名_标签名"
                label_node_name = f"{device_name}_{label_name}"
                graph.add_node(label_node_name, type='audio_info')

                # 添加设备和映射后标签节点之间的边（如果不存在）
                graph.add_edge(device_name, label_node_name)
                # 添加信噪比和映射后标签节点之间的边（如果不存在）
                graph.add_edge(snr_name, label_node_name)

                graphs.append(graph)

    return graphs


if __name__ == '__main__':
    pkl_file_path = '/home/mtftau-5/workplace/dataset/data.pkl'  # 这里填写实际的pkl文件路径，如果不在当前目录需写全路径
    graphs = build_graphs_from_pkl(pkl_file_path)
    # 可以在这里对生成的多个图进行进一步处理，比如保存每个图到不同文件等
