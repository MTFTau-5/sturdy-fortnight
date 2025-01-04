import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset, DataLoader
import torch
import pickle

class AudioDataset(Dataset):
    def __init__(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            all_data = pickle.load(f)
        # 将数据合并到一个列表中
        self.merged_data = []
        for snr_data in all_data:
            self.merged_data.extend(snr_data)

    def __len__(self):
        return len(self.merged_data)

    def __getitem__(self, idx):
        mfcc_feature = self.merged_data[idx][0]
        device_num = self.merged_data[idx][1]
        label = self.merged_data[idx][2]
        # 将 MFCC 特征转换为张量
        mfcc_feature_tensor = torch.from_numpy(mfcc_feature).float()
        return mfcc_feature_tensor, device_num, label


if __name__ == "__main__":
    pkl_file_path = '/home/mtftau-5/workplace/dataset/data.pkl'
    audio_dataset = AudioDataset(pkl_file_path)

    # 生成样本数据
    # 原代码这里有错误，正确的调用方式是这样
    X, y = make_blobs(n_samples=50000,centers=4, n_features=2, random_state=0)

    # 初始化 K-Means 模型
    kmeans = KMeans(n_clusters=4, random_state=0)

    # 训练 K-Means 模型
    kmeans.fit(X)

    # 预测每个样本的簇标签
    labels = kmeans.predict(X)

    # 可视化结果
    plt.figure(figsize=(8, 6))
    # 原代码这里有错误，应该根据预测的簇标签来设置颜色
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='red', s=100)
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
