# -*- coding: utf-8 -*-
"""
Production Training Script - Tomato Disease Detection
EfficientNetV2-S with Progressive Fine-tuning, FocalLoss, Balanced Sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import timm
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = Path("/home/smart-farming")
DATA_DIR = BASE_DIR / "tomato_extract/tomato_disease_knowledge.json-20260204T122943Z-3-001/PlantDiseased-20260204T122941Z-3-002/PlantDiseased"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
OUTPUT_DIR = BASE_DIR / "models" / "production"
PLOTS_DIR = BASE_DIR / "outputs" / "plots_production"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "tomato_bacterial_spot", "tomato_early_blight", "tomato_healthy",
    "tomato_late_blight", "tomato_leaf_mold", "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot", "tomato_spider_mites",
    "tomato_target_spot", "tomato_yellow_leaf_curl_virus"
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Training config
MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15
STAGE3_EPOCHS = 20
STAGE1_LR = 1e-3
STAGE2_LR = 5e-5
STAGE3_LR = 1e-5
BATCH_SIZE = 32
NUM_WORKERS = 4
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
MIN_CONFIDENCE = 0.70

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ============================================================================
# DATASET
# ============================================================================
class TomatoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for idx, cls in enumerate(CLASS_NAMES):
            cls_dir = Path(root_dir) / cls
            if not cls_dir.exists():
                continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() in exts:
                    self.samples.append((str(f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

    def get_class_counts(self):
        counts = [0] * NUM_CLASSES
        for _, l in self.samples:
            counts[l] += 1
        return counts


def get_train_transforms():
    return A.Compose([
        A.Resize(int(IMAGE_SIZE * 1.15), int(IMAGE_SIZE * 1.15)),
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
                           border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.Perspective(scale=0.05, p=0.35),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=30, p=0.6),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2,
                       shadow_dimension=5, p=0.4),
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.Blur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.35),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.35),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.35),
        A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


# ============================================================================
# MODEL
# ============================================================================
class TomatoModelProduction(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        print(f"[Model] {model_name} | features={self.feature_dim} | classes={num_classes}")

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True
        self._info("Backbone FROZEN")

    def unfreeze_top(self, n=4):
        for p in self.backbone.parameters():
            p.requires_grad = False
        # EfficientNetV2 blocks are in backbone.blocks
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            total = len(blocks)
            for i in range(max(0, total - n), total):
                for p in blocks[i].parameters():
                    p.requires_grad = True
        # Also unfreeze final norm
        if hasattr(self.backbone, 'bn2'):
            for p in self.backbone.bn2.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, 'conv_head'):
            for p in self.backbone.conv_head.parameters():
                p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True
        self._info(f"Top {n} blocks UNFROZEN")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        self._info("ALL layers UNFROZEN")

    def _info(self, msg):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [{msg}] Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")


# ============================================================================
# TRAINER
# ============================================================================
class ProductionTrainer:
    def __init__(self, model, train_loader, val_loader, class_counts, device=DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scaler = GradScaler()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_acc = 0.0
        self.best_loss = float('inf')

        # Focal loss with class weights
        weights = 1.0 / (np.array(class_counts, dtype=np.float32) + 1)
        weights = weights / weights.sum() * NUM_CLASSES
        self.class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.criterion = FocalLoss(alpha=self.class_weights, gamma=FOCAL_GAMMA)
        print(f"[Loss] FocalLoss gamma={FOCAL_GAMMA}")
        print(f"[Weights] {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in weights]))}")

    def train_stage(self, stage, epochs, lr, weight_decay=0.01, warmup=200):
        stage_names = {1: "HEAD ONLY", 2: "TOP BLOCKS", 3: "FULL MODEL"}
        print(f"\n{'='*80}")
        print(f"STAGE {stage}: {stage_names[stage]} | epochs={epochs} lr={lr}")
        print(f"{'='*80}")

        if stage == 1:
            self.model.freeze_backbone()
        elif stage == 2:
            self.model.unfreeze_top(n=4)
        else:
            self.model.unfreeze_all()

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
        total_steps = epochs * len(self.train_loader)
        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=min(warmup, total_steps//2))
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup), eta_min=1e-7)
        scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[min(warmup, total_steps//2)])

        patience, patience_counter = 10, 0
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = self._train_one(optimizer, scheduler)
            v_loss, v_acc = self._validate()
            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['train_acc'].append(t_acc)
            self.history['val_acc'].append(v_acc)

            is_best = v_acc > self.best_acc
            if is_best:
                self.best_acc = v_acc
                self.best_loss = v_loss
                patience_counter = 0
                self._save(f"best_production.pth", epoch, stage, v_acc, v_loss)
            else:
                patience_counter += 1

            marker = " *** BEST ***" if is_best else ""
            print(f"  S{stage} E{epoch}/{epochs} | "
                  f"Train: {t_loss:.4f} / {t_acc:.1f}% | "
                  f"Val: {v_loss:.4f} / {v_acc:.1f}%{marker}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Save stage checkpoint
        self._save(f"stage{stage}_final.pth", epochs, stage, self.best_acc, self.best_loss)

    def _train_one(self, optimizer, scheduler):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            with autocast():
                out = self.model(images)
                loss = self.criterion(out, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()
            total_loss += loss.item() * images.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += images.size(0)
        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast():
                out = self.model(images)
                loss = self.criterion(out, labels)
            total_loss += loss.item() * images.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += images.size(0)
        return total_loss / total, 100.0 * correct / total

    def _save(self, name, epoch, stage, acc, loss):
        path = OUTPUT_DIR / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': MODEL_NAME,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'image_size': IMAGE_SIZE,
            'mean': MEAN, 'std': STD,
            'epoch': epoch, 'stage': stage,
            'val_acc': acc, 'val_loss': loss,
            'min_confidence': MIN_CONFIDENCE,
        }, path)

    @torch.no_grad()
    def final_eval(self):
        print(f"\n{'='*80}\nFINAL EVALUATION\n{'='*80}")
        self.model.eval()
        all_labels, all_preds = [], []
        for images, labels in self.val_loader:
            images = images.to(self.device)
            out = self.model(images)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

        all_labels, all_preds = np.array(all_labels), np.array(all_preds)
        acc = 100.0 * (all_labels == all_preds).mean()
        print(f"\nOverall Accuracy: {acc:.2f}%\n")

        short_names = [c.replace('tomato_', '') for c in CLASS_NAMES]
        report = classification_report(all_labels, all_preds, target_names=short_names, digits=4)
        print(report)

        # Save report
        with open(OUTPUT_DIR / "classification_report.txt", 'w') as f:
            f.write(f"Overall Accuracy: {acc:.2f}%\n\n{report}")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
               xticklabels=short_names, yticklabels=short_names,
               ylabel='True', xlabel='Predicted', title=f'Confusion Matrix (Acc: {acc:.1f}%)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=8)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "confusion_matrix_production.png", dpi=150)
        plt.close()

        # Training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Val')
        ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss Curves')
        ax1.legend()
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'], label='Val')
        ax2.set(xlabel='Epoch', ylabel='Accuracy %', title='Accuracy Curves')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "training_curves_production.png", dpi=150)
        plt.close()

        print(f"\nPlots saved to {PLOTS_DIR}")
        print(f"Model saved to {OUTPUT_DIR / 'best_production.pth'}")
        return acc


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("PRODUCTION TRAINING - Tomato Disease Detection")
    print(f"Model: {MODEL_NAME} | Device: {DEVICE}")
    print("=" * 80)

    # Datasets
    train_ds = TomatoDataset(TRAIN_DIR, get_train_transforms())
    val_ds = TomatoDataset(VAL_DIR, get_val_transforms())
    class_counts = train_ds.get_class_counts()
    print(f"\nTraining: {len(train_ds)} images | Validation: {len(val_ds)} images")
    for name, count in zip(CLASS_NAMES, class_counts):
        print(f"  {name}: {count}")

    # Balanced sampler
    weights_per_class = 1.0 / (np.array(class_counts, dtype=np.float32) + 1)
    sample_weights = [weights_per_class[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Model & Trainer
    model = TomatoModelProduction(pretrained=True)
    trainer = ProductionTrainer(model, train_loader, val_loader, class_counts)

    start = time.time()
    trainer.train_stage(1, STAGE1_EPOCHS, STAGE1_LR)
    trainer.train_stage(2, STAGE2_EPOCHS, STAGE2_LR)
    trainer.train_stage(3, STAGE3_EPOCHS, STAGE3_LR)
    elapsed = time.time() - start
    print(f"\nTotal training time: {elapsed/60:.1f} minutes")

    # Load best model and evaluate
    best_ckpt = torch.load(OUTPUT_DIR / "best_production.pth", map_location=DEVICE)
    model.load_state_dict(best_ckpt['model_state_dict'])
    trainer.model = model.to(DEVICE)
    acc = trainer.final_eval()

    # Save model metadata
    meta = {
        'model_name': MODEL_NAME, 'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES, 'image_size': IMAGE_SIZE,
        'mean': MEAN, 'std': STD, 'min_confidence': MIN_CONFIDENCE,
        'val_accuracy': float(acc), 'training_time_min': elapsed/60,
    }
    with open(OUTPUT_DIR / "model_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*80}")
    print(f"DONE! Best accuracy: {acc:.2f}%")
    print(f"Model: {OUTPUT_DIR / 'best_production.pth'}")
    print(f"Metadata: {OUTPUT_DIR / 'model_metadata.json'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
