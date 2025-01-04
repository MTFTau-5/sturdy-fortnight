import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import math
from sklearn.model_selection import train_test_split 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim*2 // num_heads
        assert self.head_dim * num_heads//2 == self.embed_dim

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x).view(batch_size, 
                                seq_length, self.num_heads, 
                                3 * self.head_dim//2).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) /math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        return self.out_proj(output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + self.dropout2(src2))
        
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


# 定义包含Transformer的音频分类模型
class AudioTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=6, dim_feedforward=1024):
        super(AudioTransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, 208)
        encoder_layer = TransformerEncoderLayer(208, num_heads, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(676, num_classes)
        self.fc = nn.Linear(num_classes,2)

# 在类的forward方法内
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 1, 2).contiguous()
        x = nn.MaxPool1d(kernel_size=4) (x)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.fc(x)
        
        return x


class AudioDataset(Dataset):
    def __init__(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            all_data = pickle.load(f)

        self.merged_data = []
        for snr_data in all_data:
            self.merged_data.extend(snr_data)

    def __len__(self):
        return len(self.merged_data)

    def __getitem__(self, idx):
        mfcc_feature = self.merged_data[idx][0]
        device_num = self.merged_data[idx][1]
        label = self.merged_data[idx][2]

        # 将MFCC特征转换为张量，这里假设mfcc_feature是合适的多维数组结构，根据实际调整维度顺序等
        mfcc_feature_tensor = torch.from_numpy(mfcc_feature).float()

        return mfcc_feature_tensor, device_num, label


if __name__ == "__main__":
    pkl_file_path = '/home/mtftau-5/workplace/dataset/data.pkl'

    audio_dataset = AudioDataset(pkl_file_path)
    train_data, test_data = train_test_split(audio_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    input_dim = 153
    num_classes = 10


    model = AudioTransformerModel(input_dim, num_classes)


    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0008)
    model = model.to(device)
    num_epochs = 35
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_features, batch_device_nums, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1} training loss: {running_loss / len(train_loader)}')

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_device_nums, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    print(f"Final test loss: {test_loss / len(test_loader)}")
    print(f"Final test accuracy: {100 * correct / total}%")