# -*- coding: utf-8 -*-
"""Export trained model to ONNX format for production deployment"""

import torch
import json
from pathlib import Path
import timm
import torch.nn as nn

OUTPUT_DIR = Path("/home/smart-farming/models/production")

class TomatoModelProduction(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.feature_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


def export():
    ckpt_path = OUTPUT_DIR / "best_production.pth"
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found. Train the model first.")
        return

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_name = ckpt['model_name']
    num_classes = ckpt['num_classes']
    image_size = ckpt['image_size']

    model = TomatoModelProduction(model_name, num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    onnx_path = OUTPUT_DIR / "tomato_disease_v2.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=['image'], output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17, do_constant_folding=True
    )

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX exported: {onnx_path} ({size_mb:.1f} MB)")

    # Save metadata alongside
    meta = {
        'model_name': model_name, 'num_classes': num_classes,
        'class_names': ckpt['class_names'], 'image_size': image_size,
        'mean': ckpt['mean'], 'std': ckpt['std'],
        'min_confidence': ckpt['min_confidence'],
        'val_accuracy': ckpt.get('val_acc', 0),
        'onnx_file': onnx_path.name,
    }
    with open(OUTPUT_DIR / "onnx_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print("Metadata saved.")


if __name__ == "__main__":
    export()
