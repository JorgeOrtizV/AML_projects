import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms.functional import resize
import itertools
from tqdm import tqdm

class UNet2(nn.Module):
    def __init__(self, c_in=1, c_out=1):
        super(UNet2, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder layers
        self.encoder1 = conv_block(c_in, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        # Self-Attention after deeper encoder layers
        self.sa3 = SelfAttention(256, 8)  # Adds attention at encoder3
        self.sa4 = SelfAttention(512, 4)  # Adds attention at encoder4

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck with deeper processing
        self.bottleneck1 = conv_block(512, 1024)
        self.bottleneck2 = conv_block(1024, 1024)
        self.bottleneck3 = conv_block(1024, 1024)

        # Decoder layers
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, c_out, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e3 = self.sa3(e3)  # Apply self-attention
        e4 = self.encoder4(self.pool(e3))
        e4 = self.sa4(e4)  # Apply self-attention

        # Bottleneck
        b = self.bottleneck1(self.pool(e4))
        b = self.bottleneck2(b)
        b = self.bottleneck3(b)

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)

        # Final output
        out = self.final_conv(d1)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, mid_channels)
        self.act = nn.GELU() ## Try Relu, leakyReLU
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.residual = residual
        
    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.norm1(x2)
        x2 = self.act(x2)
        x2 = self.conv2(x2)
        x2 = self.norm2(x2)
        if self.residual:
            return self.act(x+x2)
        else:
            return x2
        
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.linear = nn.Linear(channels, channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute(0,2,1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        x = self.ln(attention_value)
        x = self.linear(x)
        x = self.act(x)
        x = self.linear(x)
        attention_value = x + attention_value
        
        return attention_value.permute(0, 2, 1).view(b, c, h, w)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxPool = nn.MaxPool2d(2)
        self.doubleConv1 = DoubleConv(in_channels, in_channels, residual=True)
        self.doubleConv2 = DoubleConv(in_channels, out_channels)
        
        self.act = nn.SiLU()
        self.linear = nn.Linear(emb_dim, out_channels)
        
    def forward(self, x):
        x = self.maxPool(x)
        x = self.doubleConv1(x)
        x = self.doubleConv2(x)
        
        return x  
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        
        #self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.doubleConv1 = DoubleConv(in_channels, in_channels, residual=True)
        self.doubleConv2 = DoubleConv(in_channels, out_channels, in_channels//2)
        self.act = nn.SiLU()
        self.linear = nn.Linear(emb_dim, out_channels)
        
    def forward(self, x, skip_x):
        #print(x.size())
        x = self.up(x)
        #print(x.size())
        if x.shape[-2:] != skip_x.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip_x.shape[-2:], mode='bilinear', align_corners=True)
            #print(x.size())
        x = torch.cat([skip_x, x], dim=1)
        x = self.doubleConv1(x)
        x = self.doubleConv2(x)
        return x

    


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)
        self.bottleneck2 = conv_block(1024, 1024)
        self.bottleneck3 = conv_block(1024, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        b2 = self.bottleneck2(b)
        b3 = self.bottleneck3(b2)

        # Decoder
        d4 = self.upconv4(b3)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)

        # Final output
        out = self.final_conv(d1)
        return out

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss
        return loss.mean()
    
class MitralDataset(Dataset):
    def __init__(self, data, target_size=(512, 512)):
        self.data = data
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]['frame']
        label = self.data[idx]['label']
        box = self.data[idx]['box']

        frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        box_tensor = torch.tensor(box, dtype=torch.float32).unsqueeze(0)

        frame_tensor = resize(frame_tensor, self.target_size)
        label_tensor = resize(label_tensor, self.target_size)
        box_tensor = resize(box_tensor, self.target_size)


        return frame_tensor, label_tensor, box_tensor
    

def calculate_iou(predictions, labels):
    predictions = predictions.bool()
    labels = labels.bool()

    intersection = (predictions & labels).sum(dim=(1, 2, 3))
    union = (predictions | labels).sum(dim=(1, 2, 3))

    iou = intersection / (union + 1e-8)
    return iou.mean().item()


if __name__ == "__main__":

    with open('data/train_expert.pkl', 'rb') as f:
        train_expert = pickle.load(f)
    with open('data/train_amateur.pkl', 'rb') as f:
        train_amateur = pickle.load(f)
    with open('data/amateur.pkl', 'rb') as f:
        amateur = pickle.load(f)
    with open('data/val_expert.pkl', 'rb') as f:
        val_expert = pickle.load(f)
    with open('data/val_amateur.pkl', 'rb') as f:
        val_amateur = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amateur_dataset = MitralDataset(amateur)
    professional_train_dataset = MitralDataset(train_expert)
    professional_val_dataset = MitralDataset(val_expert)

    train_dataset = amateur_dataset + professional_train_dataset
    val_dataset = professional_val_dataset

    # define weights
    amateur_weight = 1.0
    professional_weight = 5.0
    weights = (
        [amateur_weight] * len(amateur_dataset) +
        [professional_weight] * len(professional_train_dataset)
    )

    weighted_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_dataset, sampler=weighted_sampler, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)    
    
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    model.to(device)
    
    for epoch in tqdm(range(20)):
        model.train()
        epoch_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels= images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{20}], Loss: {epoch_loss / len(train_loader):.4f}")

        # Evaluate IoU on validation set
        model.eval()
        total_iou = 0.0
        with torch.no_grad():
            for val_images, val_labels,_ in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                output = model(val_images)
                val_predictions = (torch.sigmoid(output) > 0.5).float()
                total_iou += calculate_iou(val_predictions, val_labels)
        avg_iou = total_iou / len(val_loader)
        print(f"Epoch [{epoch + 1}/{20}], Validation IoU: {avg_iou:.4f}")

        
    torch.save(model.state_dict(), './models/unet2_1')
        
    