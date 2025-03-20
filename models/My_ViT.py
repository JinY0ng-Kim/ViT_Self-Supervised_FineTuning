import torch
from torch import nn
from torch import Tensor
from torchinfo import summary

class Embedding(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 patch_size: int, 
                 emb_size: int, 
                 img_size: int):
        
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 +1, emb_size))


    def forward(self, x: Tensor) -> Tensor:
        # [2, 3, 224, 224]
        x = self.proj(x) # [2, 768, 14, 14]

        x = x.permute(0, 2, 3, 1) # [2, 14, 14, 768]
        x = x.reshape(x.size(0), -1, x.size(-1)) # [2, 196, 768]

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # cls_tokens[2, 1, 768]

        x = torch.cat([cls_tokens, x], dim=1) # [2, 197, 768]

        x += self.positions # [2, 197, 768] : [2, 197, 168] + [197, 768]Broadcasting
        
        return x

class MLP_Block(nn.Module):
    def __init__(self, 
                 emb_dim: int, 
                 MLP_Expansion: int, 
                 MLP_dropout: float):
        
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, MLP_Expansion * emb_dim),
            nn.GELU(),
            nn.Dropout(MLP_dropout),
            nn.Linear(MLP_Expansion * emb_dim, emb_dim)
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return x
        
class TransformerEncoder_Block(nn.Module):
    def __init__(self,
                emb_dim: int,
                n_heads: int,
                dropout: float,
                MLP_Expansion: int,
                MLP_dropout: float
                ): 
        super().__init__()
        
        self.norm1 = nn.LayerNorm(emb_dim)
        self.att = nn.MultiheadAttention(embed_dim = emb_dim, num_heads = n_heads, dropout = dropout)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP_Block(emb_dim, MLP_Expansion, MLP_dropout)

    def forward(self, x):
        # [2, 197, 768]
        norm1_out = self.norm1(x)
        att_out, _ = self.att(norm1_out, norm1_out, norm1_out)
        x = x + att_out # Residual 

        norm2_out = self.norm2(x)
        mlp_out = self.mlp(norm2_out)
        x = x + mlp_out # Residual
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, 
                 depth: int, 
                 emb_dim: int, 
                 n_heads: int, 
                 dropout: float, 
                 MLP_Expansion: int, 
                 MLP_dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            *[TransformerEncoder_Block(emb_dim = emb_dim, 
                                       n_heads = n_heads, 
                                       dropout = dropout, 
                                       MLP_Expansion = MLP_Expansion, 
                                       MLP_dropout = MLP_dropout) for _ in range(depth)]
            )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class ClassHead(nn.Module):
    def __init__(self, emb_dim: int, n_classes: int):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x):
        # [2, 197, 768]
        x = x.mean(dim=1) # [2, 768]
        x = self.layer(x) # [2, 1000]

        # cls_token = x[:, 0, :]  # Using CLS Token
        # x = self.mlp(cls_token)
        return x

class My_ViT(nn.Module):
    def __init__(self,
               in_channels: int = 3,
               patch_size: int = 16,
               emb_dim: int = 768,
               n_heads: int = 8,
               img_size: int = 224,
               depth: int = 12,
               MLP_Expansion: int = 4,
               MLP_dropout: float = 0,
               dropout: float = 0.1,
               n_classes: int = 1000):
        super().__init__()

        self.vit = nn.Sequential(
            Embedding(in_channels, patch_size, emb_dim, img_size),
            TransformerEncoder(depth = depth, 
                               emb_dim = emb_dim, 
                               n_heads = n_heads, 
                               dropout = dropout,
                               MLP_Expansion = MLP_Expansion,
                               MLP_dropout = MLP_dropout),
            ClassHead(emb_dim, n_classes)
            )

    def forward(self, x):
        return self.vit(x)


if __name__ == "__main__":
    summary(My_ViT(), (2, 3, 224, 224), device='cpu', depth=5)