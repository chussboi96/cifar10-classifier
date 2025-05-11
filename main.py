import os
import torch
import torch.nn as nn
from PIL import Image
import gradio as gr
from torchvision import transforms, models


# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, n_classes=10,
                 embed_dim=192, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class HybridCNNMLP(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 cnn_feature_dim=64, mlp_hidden=256, n_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.cnn_feature_dim = cnn_feature_dim
        # CNN encoder for each patch
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, cnn_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # (B, cnn_feature_dim, 1, 1)
        )
        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(self.n_patches * cnn_feature_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, n_classes)
        )

    def forward(self, x):
        B = x.size(0)
        # split into patches
        patches = x.unfold(2, self.patch_size, self.patch_size) \
                   .unfold(3, self.patch_size, self.patch_size)
        # (B, C, n_h, n_w, p, p)
        patches = patches.permute(0,2,3,1,4,5).contiguous()
        # reshape to (B*n_patches, C, p, p)
        patches = patches.view(-1, x.size(1), self.patch_size, self.patch_size)
        # encode each patch
        cnn_out = self.cnn(patches)        # (B*n_patches, cnn_feature_dim, 1,1)
        cnn_out = cnn_out.view(-1, self.cnn_feature_dim)  # (B*n_patches, feat)
        # group back to (B, n_patches*feat)
        feats = cnn_out.view(B, self.n_patches * self.cnn_feature_dim)
        # classify
        out = self.mlp(feats)
        return out


# --- CIFAR-10 class names ---
class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]


# 1) Load ViT
vit = VisionTransformer().to(device)
vit.load_state_dict(torch.load(r"C:\Users\LOQ\vit_cifar10_cosine.pth", map_location=device))
vit.eval()

# 2) Load Hybrid CNN-MLP
hybrid = HybridCNNMLP().to(device)
hybrid.load_state_dict(torch.load(r"C:\Users\LOQ\hybrid_cnn_mlp.pth", map_location=device))
hybrid.eval()

# 3) Load ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
resnet.maxpool = nn.Identity()
for param in resnet.parameters(): 
    param.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, 10)

# Load weights
resnet.load_state_dict(torch.load(
    r"C:\Users\LOQ\resnet18_cifar10.pth", 
    map_location=device
))

resnet = resnet.to(device)
resnet.eval()

# --- Transforms ---
# ViT & Hybrid use CIFAR normalization
cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std  = [0.2023, 0.1994, 0.2010]
transform_vit = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std)
])
transform_hybrid = transform_vit
# ResNet uses ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
transform_resnet = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# --- Prediction function ---
def classify(image, true_label_str):
    # map the true label string back to its integer index
    true_label = class_names.index(true_label_str)

    x_vit    = transform_vit(image).unsqueeze(0).to(device)
    x_hybrid = transform_hybrid(image).unsqueeze(0).to(device)
    x_res    = transform_resnet(image).unsqueeze(0).to(device)

    with torch.no_grad():
        p_vit    = vit(x_vit).argmax(1).item()
        p_hybrid = hybrid(x_hybrid).argmax(1).item()
        p_res    = resnet(x_res).argmax(1).item()

    def fmt(name, pred):
        correct = (pred == true_label)
        color = "green" if correct else "red"
        return f"**{name}:** <span style='color:{color}'>{class_names[pred]}</span>  \n"

    # build a Markdown string
    md  = fmt("ViT",            p_vit)
    md += fmt("Hybrid CNN-MLP",  p_hybrid)
    md += fmt("ResNet18",        p_res)
    return md

# --- Gradio UI (using Markdown output) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # CIFAR-10 Classifier Comparison  
    Upload a CIFAR-10 test image, select its true label, and see predictions.  
    Correct = green, Incorrect = red.
    """)
    with gr.Row():
        img_input   = gr.Image(type="pil", label="Upload Image")
        label_input = gr.Dropdown(choices=class_names, label="True Label")
    output = gr.Markdown(label="Predictions")
    gr.Button("Classify").click(fn=classify, inputs=[img_input, label_input], outputs=output)

demo.launch()