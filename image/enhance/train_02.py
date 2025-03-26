import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import yaml
import csv
from pathlib import Path
from tqdm import tqdm
import fused_ssim
import argparse
from datetime import datetime
import glob
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# U-Net model definition
class UNet(nn.Module):
    _conv311_args = {
        'kernel_size': 3,
        'padding': 1,
        'padding_mode': "reflect"
    }

    def __init__(self, channels: int=3, num_hiddens: list[int]=[16, 32, 48, 64]):
        super(UNet, self).__init__()

        nums = [channels] + num_hiddens
        self.nums = nums

        # Encoder
        enc_convs = []
        for n1, n2 in zip(nums[:-1], nums[1:]):
            enc_convs.append(self._make_conv_block(n1, n2))
        self.enc_convs = nn.ParameterList(enc_convs)
        
        # Decoder
        dec_convs = [nn.Conv2d(nums[1], channels, kernel_size=1, bias=False)]
        # for n2, n1 in zip(reversed(nums)[:-1], reversed(nums)[1:]):
        for n1, n2 in zip(nums[1:-1], nums[2:]):
            dec_convs.append(self._make_conv_block(n2+n1, n1))
        self.dec_convs = nn.ParameterList(dec_convs)
        
        # Upsampling
        upsamples = []
        for n in nums[2:]:
            upsamples.append(nn.Sequential(
                nn.Conv2d(n, 4*n, **self._conv311_args),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True),
            ))
        self.upsamples = nn.ParameterList(upsamples)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **self._conv311_args),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, **self._conv311_args),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    @staticmethod
    def affine_transform(x0, y0):
        shape = x0.shape
        x = x0.reshape(*shape[:2], -1).transpose(-1, -2)
        y = y0.reshape(*shape[:2], -1).transpose(-1, -2)
        xTy = x.transpose(-1, -2) @ y
        xTx = x.transpose(-1, -2) @ x
        try:
            eye = torch.linalg.matrix_norm(xTx, keepdim=True) * torch.eye(3).unsqueeze(0).to(xTx)
            xTx = xTx + 1e-6 * eye
            A = torch.linalg.inv(xTx) @ xTy
        except torch._C._LinAlgError:
            print("Warning: torch._C._LinAlgError")
            return y0.detach()
        else:
            return (x @ A).transpose(-1, -2).reshape(shape)

    def forward(self, x_input):
        means = torch.mean(x_input, (1, 2, 3), keepdim=True)
        stds = torch.std(x_input, (1, 2, 3), keepdim=True) + 1e-8
        x0 = (x_input-means)/stds

        e0 = self.enc_convs[0](x0)
        encs = [e0]
        for i in range(1, len(self.nums)-1):
            pool = F.max_pool2d(encs[i-1], 2)
            enc = self.enc_convs[i](pool)
            encs.append(enc)

        for i in range(len(self.nums)-2, 0, -1):
            up = self.upsamples[i-1](encs[i])
            concat = torch.cat([up, encs[i-1]], dim=1)
            dec = self.dec_convs[i](concat)
        d = self.dec_convs[0](dec)

        # out = d + x0
        out = d
        # out = self.affine_transform(d, x0)
        # out = 0.2 * d + 0.8 * self.affine_transform(d, x0)
        return out*stds+means

# Image dataset class
class ImageEnhancementDataset(Dataset):
    def __init__(self, image_paths, cached_images, config, is_train=True, seed=None):
        self.image_paths = image_paths
        self.cached_images = cached_images
        self.config = config
        self.train_tile_size = config['train_tile_size']
        self.is_train = is_train
        self.seed = seed
        
        self.to_tensor = transforms.ToTensor()
        
        self.augmentations = A.Compose([
            A.MotionBlur(blur_limit=15, allow_shifted=False, p=0.5),
            A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.5), p=0.4),
            A.ShotNoise(scale_range=(0.0, 0.04), p=0.6),
            A.RingingOvershoot(blur_limit=(7, 15), cutoff=(0.25*np.pi, 1.0*np.pi), p=0.5),
            A.ImageCompression(quality_range=(40, 95), compression_type='jpeg', p=0.5),
            A.ImageCompression(quality_range=(40, 95), compression_type='webp', p=0.2),
        ], p=1.0)
        
        if not is_train and seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if img_path in self.cached_images:
            img = self.cached_images[img_path]
        else:
            img = load_image(img_path, self.config['image_dim'], self.train_tile_size)
        
        h, w, _ = img.shape
        
        if self.is_train:
            # Take random square sub-tile
            max_x = w - self.train_tile_size
            max_y = h - self.train_tile_size
            x = random.randint(0, max(0, max_x))
            y = random.randint(0, max(0, max_y))
            img = img[y:y+self.train_tile_size, x:x+self.train_tile_size]

        aug_img = self.augmentations(image=img)['image']

        if False:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(img)
            ax2.imshow(aug_img)
            plt.show()

        return (
            self.to_tensor(aug_img).float(),
            self.to_tensor(img).float()
        )

# Downsampling function without interpolation
def downsample_image(img, scale_factor):
    """
    Downsample an image by averaging s*s square tiles without interpolation.
    Discard right and bottom paddings if necessary.
    """
    h, w, c = img.shape
    dtype = img.dtype
    new_h = h // scale_factor
    new_w = w // scale_factor
    
    # Crop to multiple of scale_factor
    img = img[:new_h * scale_factor, :new_w * scale_factor]
    
    # Reshape and average
    img = np.mean(np.mean(
        img.reshape(new_h, scale_factor, new_w, scale_factor, c),
        axis=1), axis=2)
    return img.astype(dtype)

def load_image(img_path, image_dim, train_tile_size):
    # load cached if possible
    cache_dir = os.path.join(os.path.dirname(__file__), "image_cache_1")
    # cache_dir = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        img_id = str((img_path, image_dim, train_tile_size)).encode('utf-8')
        filename = hashlib.md5(img_id).hexdigest()+'.png'
        filename = os.path.join(cache_dir, filename)
        if os.path.isfile(filename):
            img_path = filename

    img = cv2.imread(img_path)
    if img is None:
        open(filename, 'w').close()
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    if min(h, w) < train_tile_size:
        open(filename, 'w').close()
        return None

    # downsample
    dim = min(h, w)
    if dim >= image_dim:
        scale_factor = max(1, dim // image_dim)
        if scale_factor > 1:
            img = downsample_image(img, scale_factor)

    assert min(img.shape[0], img.shape[1]) >= train_tile_size

    if cache_dir is not None:
        if not os.path.isfile(filename):
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img

# Image processing and caching
def process_images(image_directory, image_dim, train_tile_size, cache=True):
    """
    Process images from the directory, downscale them, and return valid paths.
    If cache is True, also cache the processed images.
    """
    image_paths = []
    
    # Find all image files recursively
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(image_directory, '**', ext), recursive=True))

    def process_image(img_path, image_dim, train_tile_size, cache):
        try:
            img = load_image(img_path, image_dim, train_tile_size)
            if img is not None:
                if cache:
                    return img_path, img
                return img_path, None
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
        return None, None

    valid_paths = []
    cached_images = {}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, img_path, image_dim, train_tile_size, cache) 
                for img_path in image_paths]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            img_path, img = future.result()
            if img_path:
                valid_paths.append(img_path)
                if cache and img is not None:
                    cached_images[img_path] = img

    return valid_paths, cached_images

# Custom sampler for deterministic training
class DeterministicSampler:
    def __init__(self, data_source, seed=None):
        self.data_source = data_source
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def __iter__(self):
        indices = list(range(len(self.data_source)))
        self.rng.shuffle(indices)
        return iter(indices)
        
    def __len__(self):
        return len(self.data_source)

# Function to create a unique work directory
def create_work_directory(work_directory):
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
        return work_directory
    
    counter = 2
    while True:
        new_dir = f"{work_directory}{counter}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        counter += 1

# Learning rate scheduler
def get_lr_scheduler(optimizer, warmup, max_epochs, lr, lrf):
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        else:
            # Linear decay from lr to lrf
            return lrf + (1.0 - lrf) * (1.0 - min(1.0, (epoch - warmup) / (max_epochs - warmup)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Loss function
def compute_loss(pred, target, ssim_weight):
    l1_loss = F.l1_loss(pred, target)
    ssim_loss = 1.0 - fused_ssim.fused_ssim(pred.contiguous(), target)
    return (1 - ssim_weight) * l1_loss + ssim_weight * ssim_loss, l1_loss, ssim_loss

# Validation function
def validate(model, val_loader, device, ssim_weight):
    model.eval()
    total_loss = 0
    total_l1_loss = 0
    total_ssim_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Validation")
        num_batches = 0
        for degraded, target in progress_bar:
            degraded = degraded.to(device)
            target = target.to(device)

            tile = 2**(len(model.nums)-1)
            h, w = degraded.shape[2:]
            h, w = tile*(h//tile), tile*(w//tile)
            degraded = degraded[:, :, :h, :w]
            target = target[:, :, :h, :w]
            
            output = model(degraded)
            loss, l1_loss, ssim_loss = compute_loss(output, target, ssim_weight)
            
            total_loss += loss.item()
            total_l1_loss += l1_loss.item()
            total_ssim_loss += ssim_loss.item()

            num_batches += 1
            progress_bar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
            })
    
    return total_loss / len(val_loader), total_l1_loss / len(val_loader), total_ssim_loss / len(val_loader)

# Main training function
def train(config):
    set_seed(42)
    
    # Determine work directory
    if config['resume']:
        work_dir = config['work_directory']
    else:
        work_dir = create_work_directory(config['work_directory'])
        config['work_directory'] = work_dir
    print('work_dir:', work_dir)
    
    # Save config
    with open(os.path.join(work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Setup device
    if config['gpu_indices'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['gpu_indices'][0]}")
    else:
        device = torch.device("cpu")

    # Process images
    print("Processing images...")
    valid_paths, cached_images = process_images(
        config['image_directory'], 
        config['image_dim'], 
        config['train_tile_size'],
        config['cache']
    )
    
    # Split data
    val_size = int(config['val'] * len(valid_paths))
    train_size = len(valid_paths) - val_size
    
    # Use fixed random state for reproducibility
    rng = np.random.RandomState(42)
    indices = np.arange(len(valid_paths))
    rng.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_paths = [valid_paths[i] for i in train_indices]
    val_paths = [valid_paths[i] for i in val_indices]
    
    print(f"Training on {len(train_paths)} images, validating on {len(val_paths)} images")
    
    # Create datasets
    train_dataset = ImageEnhancementDataset(
        train_paths,
        cached_images,
        config,
        is_train=True
    )
    val_dataset = ImageEnhancementDataset(
        val_paths,
        cached_images,
        config,
        is_train=False,
        seed=42
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        sampler=DeterministicSampler(train_dataset, seed=42),
        num_workers=config['num_workers'] if config['num_workers'] > 0 else os.cpu_count(),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'] if config['num_workers'] > 0 else os.cpu_count(),
        pin_memory=True
    )
    
    # Create model
    model = UNet()
    
    # Multi-GPU training
    if len(config['gpu_indices']) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=config['gpu_indices'])
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = get_lr_scheduler(optimizer, config['warmup'], config['max_epochs'], config['lr'], config['lrf'])
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=work_dir)
    
    # CSV logger
    csv_path = os.path.join(work_dir, 'training_log.csv')
    csv_header = ['epoch', 'lr', 'train_loss', 'train_l1_loss', 'train_ssim_loss', 
                  'val_loss', 'val_l1_loss', 'val_ssim_loss']
    
    with open(csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)
    
    # Initialize variables
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Load checkpoint if resuming
    if config['resume']:
        checkpoint_path = os.path.join(work_dir, 'last.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            patience_counter = checkpoint['patience_counter']
            
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config['max_epochs']):
        model.train()
        train_loss = 0
        train_l1_loss = 0
        train_ssim_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']}")
        num_batches = 0
        for degraded, target in progress_bar:
            degraded = degraded.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(degraded)
            
            loss, l1, ssim = compute_loss(output, target, config['ssim_weight'])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_l1_loss += l1.item()
            train_ssim_loss += ssim.item()
            
            # Update progress bar
            num_batches += 1
            progress_bar.set_postfix({
                'loss': f"{train_loss/num_batches:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Step the scheduler
        scheduler.step()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_l1_loss = train_l1_loss / len(train_loader)
        avg_train_ssim_loss = train_ssim_loss / len(train_loader)
        
        # Validation
        val_loss, val_l1_loss, val_ssim_loss = validate(model, val_loader, device, config['ssim_weight'])
        
        # Log metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train/l1', avg_train_l1_loss, epoch)
        writer.add_scalar('Loss/val/l1', val_l1_loss, epoch)
        writer.add_scalar('Loss/train/ssim', avg_train_ssim_loss, epoch)
        writer.add_scalar('Loss/val/ssim', val_ssim_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch + 1,
                optimizer.param_groups[0]['lr'],
                avg_train_loss,
                avg_train_l1_loss,
                avg_train_ssim_loss,
                val_loss,
                val_l1_loss,
                val_ssim_loss
            ])
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter
        }
        
        torch.save(checkpoint, os.path.join(work_dir, 'last.pt'))
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(work_dir, 'best.pt'))
            patience_counter = 0
            print(f"Validation loss: {val_loss:.4f} - New best model saved!")
        else:
            patience_counter += 1
            print(f"Validation loss: {val_loss:.4f} - No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"Early stopping after {patience_counter} epochs without improvement")
            break
    
    writer.close()
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    if False:
        model = UNet()
        x = torch.randn((8, 3, 640, 480))
        y = model(x)
        print(y.shape)
        exit(0)

    parser = argparse.ArgumentParser(description="Image Quality Enhancement Training")
    
    # Add arguments
    parser.add_argument('--work_directory', type=str, default='./runs/train', help='Directory to save models and logs')
    parser.add_argument('--image_directory', type=str, required=True, help='Directory containing high-quality images')
    parser.add_argument('--num_workers', type=int, default=-1, help='Number of worker threads for data loading')
    parser.add_argument('--gpu_indices', type=int, nargs='+', default=[0], help='GPU indices to use')
    parser.add_argument('--image_dim', type=int, default=512, help='Target image dimension')
    parser.add_argument('--train_tile_size', type=int, default=256, help='Size of training tiles')
    parser.add_argument('--val', type=float, default=0.05, help='Fraction of data for validation')
    parser.add_argument('--cache', action='store_true', help='Cache images in RAM')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.1, help='Final learning rate fraction')
    parser.add_argument('--ssim_weight', type=float, default=0.5, help='Weight of SSIM in loss function')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    config = vars(args)
    
    train(config)

# python3 train_02.py --image_directory ~/adr/coco/df2k
