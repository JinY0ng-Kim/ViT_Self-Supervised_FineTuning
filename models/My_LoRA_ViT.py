import torch
from torch import nn
from torch import Tensor
from torchinfo import summary
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
   

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
        
        self.linear_1 = nn.Linear(emb_dim, MLP_Expansion * emb_dim)
        self.ac_dr = nn.Sequential(
            nn.GELU(),
            nn.Dropout(MLP_dropout)
        )
        self.linear_2 = nn.Linear(MLP_Expansion * emb_dim, emb_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.ac_dr(x)
        x = self.linear_2(x)
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
    #Hyperparameter
    OUTPUT_DIR = 'PATH'
    IN_CHANNEL = 3 
    NUM_CLASS = 1000
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4 
    NUM_EPOCH = 300
    WEIGHT_DECAY = 1e-2
    WORLD_SIZE = torch.cuda.device_count()

    config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["linear_1", "linear_2"],
    lora_dropout=0.1,
    bias="none"
    )

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    train_dataset = "DataLoader()"
    test_dataset = "DataLoader()"

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = My_ViT()
    model.load_state_dict("PATH")
    model = get_peft_model(model, config)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    summary(model, (2, 3, 224, 224), device='cpu', depth=7)
    model.print_trainable_parameters()

    train_losses = []
    val_losses = []
    iou_scores = []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCH):
        model.train()
        train_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{NUM_EPOCH}", unit='batch') as pbar:
            for rgb, _, label in train_loader:
                rgb, label = rgb.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(rgb)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}, Training loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for rgb, _, label in val_loader:
                rgb, label = rgb.to(device), label.to(device)
                outputs = model(rgb)
                loss = criterion(outputs, label)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}, Validation loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'best_NoRA_model_{epoch+1}.pth'))

    model = model.merge_and_unload()
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'best_model.pth'))
    summary(model, (2, 3, 224, 224), device='cpu', depth=5)

    now_epochs = range(0, epoch + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(now_epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(now_epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'))
    plt.show()
    
