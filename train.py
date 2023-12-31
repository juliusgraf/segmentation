import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from jacobian import jacobian_spectral_norm

# Hyperparameters
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
ETA_MIN = 0.001
T_MAX = 78
JACOBIAN_LOSS_WEIGHT = 1e-5
EPS_JACOBIAN_LOSS = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 32 # 1280 originally
IMAGE_WIDTH = 48 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, lr_scheduler):#train_fn(loader, model, optimizer, loss_fn, scaler, lr_scheduler)
    loop = tqdm(loader)
    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)
        
        # Forward
        #with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        jacobian_norm = jacobian_spectral_norm(data, predictions, interpolation=False, training=True)
        jacobian_loss = torch.maximum(jacobian_norm, torch.ones_like(jacobian_norm)-EPS_JACOBIAN_LOSS).mean()
        
        loss += JACOBIAN_LOSS_WEIGHT * jacobian_loss
        loss=loss.mean() #ajouter pour enlever le scaler
        # Backward
        optimizer.zero_grad()
        loss.backward()# scaler.scale(loss).backward()
        optimizer.step() # scaler.step(optimizer)
        # scaler.update()
        lr_scheduler.step()
        # Update tqdm loop
        loop.set_postfix(loss = loss.item())
        

def main():
    train_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit = 35, p = 1.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )
    
    val_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )
    
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # Cross entropy loss if out_channels = 3
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_MAX,eta_min=ETA_MIN)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    #scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, lr_scheduler)#train_fn(train_loader, model, optimizer, loss_fn, scaler, lr_scheduler)

        # Add scheduler to the model
        # Save model
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)
        
        # Check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        
        # Print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
        
if __name__ == "__main__":
    main()