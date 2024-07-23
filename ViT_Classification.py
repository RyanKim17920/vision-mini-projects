import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam 

class FFN(nn.Module):
    def __init__(self, dim, ffn_dim, dropout = 0.1):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Embeddings(nn.Module):
    def __init__(self, patch_size, img_size, dim = 128, channels = 3):
        super(Embeddings, self).__init__()
        assert img_size % patch_size == 0, "image size should be able to be divided by patch size"
        patches = (img_size // patch_size) ** 2
        patch_dim = patch_size ** 2 * channels

        self.to_dim = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_emb = nn.Parameter(torch.randn(1, patches, dim))

    def forward(self, x):
        x = self.to_dim(x)
        x += self.pos_emb
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_heads=64, dropout=0.1):
        super(Attention, self).__init__()

        inner_dim = heads * dim_heads

        self.heads = heads
        self.toqkv = nn.Linear(dim, inner_dim * 3)

        self.flash = dict(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.toqkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        with torch.backends.cuda.sdp_kernel(**self.flash):
            out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class ViT(nn.Module):
    def __init__(self, patch_size, img_size, depth, classes, heads=8, dim=128, ff_dim=128, channels=3, dim_heads=64, dropout=0.1):
        super(ViT, self).__init__()
        self.embedding = Embeddings(patch_size, img_size, dim, channels)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_heads, dropout),
                FFN(dim, ff_dim, dropout)
            ]))
        self.ln = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.embedding(x)
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x 
        x = self.ln(x)
        x = x.mean(dim=1)
        return self.to_logits(x)

class ViTModule(pl.LightningModule):
    def __init__(self, patch_size, img_size, depth, num_classes, heads=8, dim=128, ff_dim=128, channels=3, dim_heads=64, dropout=0.1):
        super(ViTModule, self).__init__()
        self.model = ViT(patch_size, img_size, depth, num_classes, heads, dim, ff_dim, channels, dim_heads, dropout)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()
        self.log('train_acc', accuracy, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()
        self.log('val_acc', accuracy, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        epoch = self.current_epoch
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        train_acc = self.trainer.callback_metrics.get('train_acc_epoch')
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_acc = self.trainer.callback_metrics.get('val_acc')

        if train_loss and train_acc and val_loss and val_acc:
            print(f"Epoch {epoch}:")
            print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=1e-3)
        return optimizer


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

batch_size = 128

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize ViT model
patch_size = 4
img_size = 32
depth = 6
num_classes = 100

vit_model = ViTModule(patch_size, img_size, depth, num_classes)

# Initialize a trainer
trainer = pl.Trainer(max_epochs=250, callbacks = [EarlyStopping(monitor="val_acc", min_delta=0.00, patience=5, verbose=False, mode="max")], 
                     precision="16", accumulate_grad_batches=64,
                     strategy="deepspeed_stage_2_offload",
                    enable_progress_bar=False)

# Train the model
trainer.fit(vit_model, train_loader, test_loader)



