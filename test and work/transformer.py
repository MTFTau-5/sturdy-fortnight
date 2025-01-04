import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import math
from sklearn.model_selection import train_test_split 


device = 'cuda:0' if torch.cuda.is_available() else 'cup'

class AudioDataset(Dataset):
    def __init__(self,pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            all_data = pickle.load(f)
        
        self.merged_data = []
        for snr_data in all_data:
            self.merged_data.extend(snr_data)
    def __len__(self):
        return len(self.merged_data)
    
    def __getitem__(self, index):
        mfcc_feature = self.merged_data[index][0]
        device_num = self.merged_data[index][1]
        label = self.merged_data[index][2]
        mfcc_feature_tensor = torch.from_numpy(mfcc_feature).float()
        return mfcc_feature_tensor, device_num, label             

class MultHeadAttention(nn.Module):
    def __init__(self, embed_dim , num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim / num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim) 

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size,
                          seq_len, 
                          self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q,k,v = qkv.chunk(3, dim=-1)
        attn_probs = F.softmax(torch.matmul(q,k.transpose(-2,-1)) /math.sqrt(self.head_dim), dim=-1)
        output = torch.matmul(attn_probs, v).transpose(1,2).contiguous().view(batch_size, seq_len, embed_dim) 
        
        return self.out_proj(output)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward):
        super().__init__(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultHeadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.doopout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.nomar = nn.LayerNorm(embed_dim)
        
    def forward (self,src):
        src2 = self.self_attn(src)
        src = self.nomar(src+self.doopout(src2))
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = self.nomar(src+self.doopout(src2))
        
        return src
    
class TransformerEncoder(nn.Module):
    def __init__ (self ,encoder_layer, num_layers):
        super().__init__(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
    def forward(self,src):
        for layer in self.layers:
            src = layer(src)
        return src



        
        
        
        
         
        
        