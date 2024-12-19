import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize
import itertools


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)

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

    # Meshgrid
    lr_values = [1e-5, 1e-4, 1e-3, 1e-2]
    batch_size_values = [4, 8, 16, 32]
    alpha_values = [0.25, 0.5]
    gamma_values = [2.0, 3.0]
    epochs_values = [10,20,25,30]

    best_iou = 0.0
    best_params = {}

    with open('data/train_expert.pkl', 'rb') as f:
        train_expert = pickle.load(f)
    with open('data/train_amateur.pkl', 'rb') as f:
        train_amateur = pickle.load(f)
    with open('data/val_expert.pkl', 'rb') as f:
        val_expert = pickle.load(f)
    with open('data/val_amateur.pkl', 'rb') as f:
        val_amateur = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_expert_dataset = MitralDataset(train_expert)
    val_expert_dataset = MitralDataset(val_expert)

    for lr, batch_size, alpha, gamma, epochs in itertools.product(
        lr_values, batch_size_values, alpha_values, gamma_values, epochs_values
    ):

        train_loader = DataLoader(train_expert_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_expert_dataset, batch_size=batch_size, shuffle=False)
        
        model = UNet()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)

        model.to(device)

        current_params = {
                'lr' : lr,
                'batch_size' : batch_size,
                'gamma' : gamma,
                'alpha': alpha,
                'epochs' : epochs
            }
        print("Training using : {}".format(current_params))
        
        for epoch in range(epochs):
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

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

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
            print(f"Epoch [{epoch + 1}/{epochs}], Validation IoU: {avg_iou:.4f}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            best_params = {
                'lr' : lr,
                'batch_size' : batch_size,
                'gamma' : gamma,
                'alpha': alpha,
                'epochs' : epochs
            }
            torch.save(model.state_dict(), './models/best.pth')
            print(f"New best IoU: {best_iou:.4f} with parameters {best_params}")
        
        torch.save(model.state_dict(), './models/latest.pth')
        print(f"Latest IoU: {avg_iou:.4f}")
    
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Best Parameters: {best_params}")
    