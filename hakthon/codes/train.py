"""
Main Training Script for Tomato Disease Detection
Orchestrates progressive 3-stage fine-tuning
"""

import torch
import argparse
from pathlib import Path

import config
from src.dataset_analyzer import DatasetAnalyzer
from src.data_pipeline import create_data_loaders
from src.model_architecture import create_model
from src.trainer import ProgressiveTrainer
from src.utils import set_seed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Tomato Disease Detection Model')
    
    # Training stages
    parser.add_argument('--stage1-epochs', type=int, default=config.STAGE1_EPOCHS,
                       help='Number of epochs for stage 1')
    parser.add_argument('--stage2-epochs', type=int, default=config.STAGE2_EPOCHS,
                       help='Number of epochs for stage 2')
    parser.add_argument('--stage3-epochs', type=int, default=config.STAGE3_EPOCHS,
                       help='Number of epochs for stage 3')
    
    # Skip stages
    parser.add_argument('--skip-stage1', action='store_true',
                       help='Skip stage 1 training')
    parser.add_argument('--skip-stage2', action='store_true',
                       help='Skip stage 2 training')
    parser.add_argument('--skip-stage3', action='store_true',
                       help='Skip stage 3 training')
    
    # Model
    parser.add_argument('--model-name', type=str, default=config.MODEL_NAME,
                       help='ViT model name from timm')
    
    # Data
    parser.add_argument('--batch-size-stage1', type=int, default=config.BATCH_SIZE_STAGE1,
                       help='Batch size for stage 1')
    parser.add_argument('--batch-size-stage2', type=int, default=config.BATCH_SIZE_STAGE2,
                       help='Batch size for stage 2')
    parser.add_argument('--batch-size-stage3', type=int, default=config.BATCH_SIZE_STAGE3,
                       help='Batch size for stage 3')
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS,
                       help='Number of data loading workers')
    parser.add_argument('--use-balanced-sampling', action='store_true',
                       help='Use balanced sampling for class imbalance')
    
    # Device
    parser.add_argument('--device', type=str, default=config.DEVICE,
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                       help='Random seed')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (reduced epochs)')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Debug mode
    if args.debug:
        print("\n" + "=" * 80)
        print("DEBUG MODE: Running with reduced epochs")
        print("=" * 80)
        args.stage1_epochs = 2
        args.stage2_epochs = 2
        args.stage3_epochs = 2
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TOMATO DISEASE DETECTION - PROGRESSIVE FINE-TUNING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {args.device}")
    print(f"  Mixed Precision: {not args.no_amp}")
    print(f"  Random Seed: {args.seed}")
    print(f"\nTraining Stages:")
    print(f"  Stage 1 (Classifier Head): {args.stage1_epochs} epochs")
    print(f"  Stage 2 (Top Blocks): {args.stage2_epochs} epochs")
    print(f"  Stage 3 (Full Model): {args.stage3_epochs} epochs")
    print(f"\nBatch Sizes:")
    print(f"  Stage 1: {args.batch_size_stage1}")
    print(f"  Stage 2: {args.batch_size_stage2}")
    print(f"  Stage 3: {args.batch_size_stage3}")
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("\nWarning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Step 1: Analyze dataset
    print("\n" + "=" * 80)
    print("STEP 1: DATASET ANALYSIS")
    print("=" * 80)
    
    analyzer = DatasetAnalyzer(
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        class_names=config.CLASS_NAMES
    )
    
    report = analyzer.analyze()
    class_counts = [report['statistics']['train_per_class'][c] for c in config.CLASS_NAMES]
    
    # Step 2: Create model
    print("\n" + "=" * 80)
    print("STEP 2: MODEL INITIALIZATION")
    print("=" * 80)
    
    model = create_model(
        model_name=args.model_name,
        device=device
    )
    
    # Step 3: Progressive fine-tuning
    print("\n" + "=" * 80)
    print("STEP 3: PROGRESSIVE FINE-TUNING")
    print("=" * 80)
    
    # Stage 1: Classifier head training
    if not args.skip_stage1:
        print("\n" + ">" * 80)
        print("STAGE 1: CLASSIFIER HEAD TRAINING")
        print(">" * 80)
        
        # Create data loaders for stage 1
        train_loader, val_loader, _ = create_data_loaders(
            batch_size=args.batch_size_stage1,
            num_workers=args.num_workers,
            use_balanced_sampling=args.use_balanced_sampling or config.USE_BALANCED_SAMPLING
        )
        
        # Create trainer
        trainer = ProgressiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=config.CLASS_NAMES,
            class_counts=class_counts,
            device=device,
            use_amp=not args.no_amp
        )
        
        # Train stage 1
        trainer.train_stage1(
            epochs=args.stage1_epochs,
            lr=config.STAGE1_LR,
            weight_decay=config.STAGE1_WEIGHT_DECAY,
            warmup_steps=config.STAGE1_WARMUP_STEPS
        )
        
        # Save training curves
        trainer.save_training_curves("stage1_training_curves.png")
    
    # Stage 2: Top blocks fine-tuning
    if not args.skip_stage2:
        print("\n" + ">" * 80)
        print("STAGE 2: TOP BLOCKS FINE-TUNING")
        print(">" * 80)
        
        # Create data loaders for stage 2
        train_loader, val_loader, _ = create_data_loaders(
            batch_size=args.batch_size_stage2,
            num_workers=args.num_workers,
            use_balanced_sampling=args.use_balanced_sampling or config.USE_BALANCED_SAMPLING
        )
        
        # Create trainer (reuse model from stage 1)
        if args.skip_stage1:
            trainer = ProgressiveTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=config.CLASS_NAMES,
                class_counts=class_counts,
                device=device,
                use_amp=not args.no_amp
            )
        else:
            # Update data loaders
            trainer.train_loader = train_loader
            trainer.val_loader = val_loader
        
        # Train stage 2
        trainer.train_stage2(
            epochs=args.stage2_epochs,
            lr=config.STAGE2_LR,
            weight_decay=config.STAGE2_WEIGHT_DECAY,
            warmup_steps=config.STAGE2_WARMUP_STEPS,
            num_blocks=config.NUM_TOP_BLOCKS_STAGE2
        )
        
        # Save training curves
        trainer.save_training_curves("stage2_training_curves.png")
    
    # Stage 3: Full model fine-tuning
    if not args.skip_stage3:
        print("\n" + ">" * 80)
        print("STAGE 3: FULL MODEL FINE-TUNING")
        print(">" * 80)
        
        # Create data loaders for stage 3
        train_loader, val_loader, _ = create_data_loaders(
            batch_size=args.batch_size_stage3,
            num_workers=args.num_workers,
            use_balanced_sampling=args.use_balanced_sampling or config.USE_BALANCED_SAMPLING
        )
        
        # Create trainer (reuse model from stage 2)
        if args.skip_stage1 and args.skip_stage2:
            trainer = ProgressiveTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=config.CLASS_NAMES,
                class_counts=class_counts,
                device=device,
                use_amp=not args.no_amp
            )
        else:
            # Update data loaders
            trainer.train_loader = train_loader
            trainer.val_loader = val_loader
        
        # Train stage 3
        trainer.train_stage3(
            epochs=args.stage3_epochs,
            lr=config.STAGE3_LR,
            weight_decay=config.STAGE3_WEIGHT_DECAY,
            warmup_steps=config.STAGE3_WARMUP_STEPS
        )
        
        # Save training curves
        trainer.save_training_curves("stage3_training_curves.png")
    
    # Step 4: Final evaluation
    print("\n" + "=" * 80)
    print("STEP 4: FINAL EVALUATION")
    print("=" * 80)
    
    metrics = trainer.evaluate(save_plots=True)
    
    # Save final training curves (all stages combined)
    trainer.save_training_curves("final_training_curves.png")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
    print(f"\nFinal Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Macro F1-Score: {metrics['macro_avg']['f1-score']:.4f}")
    print(f"  Weighted F1-Score: {metrics['weighted_avg']['f1-score']:.4f}")
    
    print(f"\nModel checkpoints saved to: {config.MODELS_DIR}")
    print(f"Training plots saved to: {config.PLOTS_DIR}")
    print(f"Logs saved to: {config.LOGS_DIR}")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Review training curves and confusion matrix")
    print("  2. Test inference with: python src/inference.py")
    print("  3. Start web interface with: python app.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
