"""
Evaluation Script for Tomato Disease Detection Model
Comprehensive evaluation on validation set with detailed metrics
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json

import config
from src.model_architecture import create_model
from src.data_pipeline import create_data_loaders
from src.utils import (
    load_checkpoint, compute_metrics, print_metrics,
    plot_confusion_matrix, visualize_predictions
)
from tqdm import tqdm


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Tomato Disease Detection Model')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save prediction visualizations')
    
    return parser.parse_args()


def evaluate_model(model, val_loader, device, save_predictions=False):
    """
    Evaluate model on validation set
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        device: Device to use
        save_predictions: Whether to save prediction visualizations
    
    Returns:
        metrics: Dict of evaluation metrics
        all_labels: True labels
        all_predictions: Predicted labels
        all_probs: Prediction probabilities
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probs = []
    all_images = []
    
    print("\nEvaluating model...")
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if save_predictions and len(all_images) < 16:
                all_images.extend(images.cpu())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions, config.CLASS_NAMES)
    
    # Print metrics
    print_metrics(metrics, title="Validation Set Evaluation")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels,
        all_predictions,
        config.CLASS_NAMES,
        config.PLOTS_DIR / "evaluation_confusion_matrix.png"
    )
    
    # Save prediction visualizations
    if save_predictions and all_images:
        visualize_predictions(
            torch.stack(all_images[:16]),
            all_labels[:16],
            all_predictions[:16],
            config.CLASS_NAMES,
            config.PLOTS_DIR / "evaluation_predictions.png"
        )
    
    return metrics, all_labels, all_predictions, all_probs


def compute_additional_metrics(all_labels, all_predictions, all_probs):
    """
    Compute additional evaluation metrics
    
    Args:
        all_labels: True labels
        all_predictions: Predicted labels
        all_probs: Prediction probabilities
    
    Returns:
        additional_metrics: Dict of additional metrics
    """
    additional_metrics = {}
    
    # Top-3 accuracy
    top3_predictions = np.argsort(all_probs, axis=1)[:, -3:]
    top3_correct = np.array([label in top3 for label, top3 in zip(all_labels, top3_predictions)])
    additional_metrics['top3_accuracy'] = top3_correct.mean()
    
    # Top-5 accuracy
    top5_predictions = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = np.array([label in top5 for label, top5 in zip(all_labels, top5_predictions)])
    additional_metrics['top5_accuracy'] = top5_correct.mean()
    
    # Average confidence for correct predictions
    correct_mask = all_labels == all_predictions
    if correct_mask.any():
        correct_probs = all_probs[correct_mask, all_predictions[correct_mask]]
        additional_metrics['avg_confidence_correct'] = correct_probs.mean()
    
    # Average confidence for incorrect predictions
    incorrect_mask = ~correct_mask
    if incorrect_mask.any():
        incorrect_probs = all_probs[incorrect_mask, all_predictions[incorrect_mask]]
        additional_metrics['avg_confidence_incorrect'] = incorrect_probs.mean()
    
    # Expected Calibration Error (ECE)
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = all_probs[np.arange(len(all_labels)), all_predictions]
    accuracies = (all_labels == all_predictions).astype(float)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    additional_metrics['expected_calibration_error'] = ece
    
    return additional_metrics


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print("=" * 80)
    print("TOMATO DISEASE DETECTION - MODEL EVALUATION")
    print("=" * 80)
    print(f"\nModel: {args.model_path}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    
    # Load model
    print("\nLoading model...")
    model = create_model(device=device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    _, val_loader, _ = create_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_sampling=False
    )
    
    # Evaluate
    metrics, all_labels, all_predictions, all_probs = evaluate_model(
        model,
        val_loader,
        device,
        save_predictions=args.save_predictions
    )
    
    # Compute additional metrics
    print("\n" + "=" * 80)
    print("ADDITIONAL METRICS")
    print("=" * 80)
    
    additional_metrics = compute_additional_metrics(all_labels, all_predictions, all_probs)
    
    print(f"\nTop-3 Accuracy: {additional_metrics['top3_accuracy']:.4f} ({additional_metrics['top3_accuracy'] * 100:.2f}%)")
    print(f"Top-5 Accuracy: {additional_metrics['top5_accuracy']:.4f} ({additional_metrics['top5_accuracy'] * 100:.2f}%)")
    
    if 'avg_confidence_correct' in additional_metrics:
        print(f"\nAverage Confidence (Correct): {additional_metrics['avg_confidence_correct']:.4f}")
    if 'avg_confidence_incorrect' in additional_metrics:
        print(f"Average Confidence (Incorrect): {additional_metrics['avg_confidence_incorrect']:.4f}")
    
    print(f"\nExpected Calibration Error: {additional_metrics['expected_calibration_error']:.4f}")
    
    # Save evaluation report
    evaluation_report = {
        'model_path': args.model_path,
        'metrics': metrics,
        'additional_metrics': {k: float(v) for k, v in additional_metrics.items()}
    }
    
    report_path = config.OUTPUTS_DIR / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print(f"\nEvaluation report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
