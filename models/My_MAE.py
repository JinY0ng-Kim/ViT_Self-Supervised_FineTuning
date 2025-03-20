import torch
from torch import nn
from torch import Tensor
from torchinfo import summary

class PatchEmbedding_Flatten(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 patch_size: int, 
                 emb_size: int):
        
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)


    def forward(self, x: Tensor) -> Tensor:
        # [2, 3, 224, 224]
        x = self.proj(x) # [2, 768, 14, 14]

        x = x.permute(0, 2, 3, 1) # [2, 14, 14, 768]
        x = x.reshape(x.size(0), -1, x.size(-1)) # [2, 196, 768]

        return x

class Random_masking(nn.Module):
    def __init__(self, 
                 masking_ratio: float):
        super().__init__()

        self.Ratio = masking_ratio
    
    def forward(self, x):
        
        batch, lenght, dim = x.shape

        num_keep = int(lenght * (1 - self.Ratio))
        
        noise = torch.rand(batch, lenght, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        x_encoder = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        mask = torch.ones([batch, lenght], device=x.device)
        mask[:, :num_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_encoder, mask, ids_restore
    
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

class Encoder_to_Decoder(nn.Module):
    def __init__(self,
                 Encoder_emb_dim,
                 Decoder_emb_dim):
        super().__init__()

        self.to_decoder_embedding = nn.Linear(Encoder_emb_dim, Decoder_emb_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, Decoder_emb_dim))

    def forward(self, x, ids_restore):
        x = self.to_decoder_embedding(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1) # [2, 197, 512] cls
        
        return x

class My_MAE(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 img_size: int = 224,

                 Encoder_emb_dim: int = 1024,
                 Encoder_n_heads: int = 16,
                 Encoder_depth: int = 24,

                 Decoder_emb_dim: int = 512,
                 Decoder_depth: int = 8,
                 Decoder_n_heads: int = 16,
                 
                 
                 MLP_Expansion: int = 4,
                 MLP_dropout: float = 0,
                 dropout: float = 0.1,
                 n_classes: int = 1000,
                 masking_ratio: float = 0.75):
        super().__init__()
        self.masking_ratio = masking_ratio

        self.PatchEmbedding = PatchEmbedding_Flatten(in_channels, patch_size, Encoder_emb_dim)
        self.random_masking = Random_masking(self.masking_ratio)

        self.Encoder_Positions = nn.Parameter(torch.zeros((img_size // patch_size) **2 +1, Encoder_emb_dim), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, Encoder_emb_dim))

        self.Encoder = nn.Sequential(
            TransformerEncoder(depth = Encoder_depth, 
                               emb_dim = Encoder_emb_dim, 
                               n_heads = Encoder_n_heads, 
                               dropout = dropout,
                               MLP_Expansion = MLP_Expansion,
                               MLP_dropout = MLP_dropout),
            nn.LayerNorm(Encoder_emb_dim)
        )

        self.encoder_to_decoder = Encoder_to_Decoder(Encoder_emb_dim, Decoder_emb_dim)

        self.Decoder_Positions = nn.Parameter(torch.randn((img_size // patch_size) **2 +1, Decoder_emb_dim), requires_grad=False)
        self.Decoder = nn.Sequential(
            TransformerEncoder(depth = Decoder_depth, 
                               emb_dim = Decoder_emb_dim, 
                               n_heads = Decoder_n_heads, 
                               dropout = dropout,
                               MLP_Expansion = MLP_Expansion,
                               MLP_dropout = MLP_dropout),                   
            nn.LayerNorm(Decoder_emb_dim)
        )

        self.Last_Linear = nn.Linear(Decoder_emb_dim, patch_size**2 * in_channels)

    def forward(self, x):
        # [2, 3, 224, 224]
        x = self.PatchEmbedding(x) # [2, 196, 1024]
        x = x + self.Encoder_Positions[1:, :]

        x, mask, ids_restore = self.random_masking(x) # [2, 49, 1024] 75%

        cls_token = self.cls_token + self.Encoder_Positions[:1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1) # [2, 50, 1024] cls

        x = self.Encoder(x)

        x = self.encoder_to_decoder(x, ids_restore) # [2, 197, 512]

        x = x + self.Decoder_Positions
        x = self.Decoder(x) # [2, 197, 512]

        x = self.Last_Linear(x) # [2, 197, 768], patch_size^2 * 3
        x = x[:, 1:, :] # [2, 196, 768], remove cls

        return x, mask
    
if __name__ == "__main__":
    summary(My_MAE(), (2, 3, 224, 224), device='cpu', depth=4)