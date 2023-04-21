import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, 2) / math.log(10000) * d_model
        emb = torch.exp(-emb)
        pos = torch.arange(T).float
        emb = pos[:, None] * emb[None, :]
        assert emb.shape == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim = -1)
        assert emb.shape == [T, d_model  // 2, 2]
        emb = emb.view(T, d_model)
    
        self.timembedding = nn.Sequential(
            # 构建一个emb形状的tensor，然后用nn.Embedding.from_pretrained()来初始化参数，t输入作为索引
            nn.Embedding.from_pretrained(emb), 
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim)
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb
    
class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding =  nn.Sequential(
            nn.Embedding(num_embeddings=num_labels+1, embedding_dim=d_model, padding_idx=0), # 一个简单的查找表（lookup table），存储固定字典和大小的词嵌入。
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim)
        )
    
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride = 2, padding = 1) # output size: (W − F + 2P )/S + 1
        self.initialize()

    def initliaze(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forwward(self, x, temb):
        x = self.main(x)
        return x
    
class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride = 1, padding = 1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor = 2, mode = 'nearest')
        x = self.main(x)
        return x
    
class AttenBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_normal_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** -0.5) # Q * K / sqrt(C)
        assert w.shape == [B, H * W, H * W]
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert h.shape == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
            
        if attn:
            self.attn = AttenBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_normal_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(x)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

    

    