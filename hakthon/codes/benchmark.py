import time
from src.data_pipeline import create_data_loaders
from src.model_architecture import create_model
from src.trainer import ProgressiveTrainer
import config

train_loader, val_loader, class_counts = create_data_loaders(batch_size=32)
model = create_model(config.MODEL_NAME, device='cuda')
trainer = ProgressiveTrainer(model, train_loader, val_loader, config.CLASS_NAMES, class_counts, device='cuda')

start = time.time()
trainer.train_stage1(epochs=1, lr=1e-3, weight_decay=0.01, warmup_steps=10)
end = time.time()
print(f"Time for 1 epoch of {len(train_loader.dataset)} images: {end-start} seconds")
