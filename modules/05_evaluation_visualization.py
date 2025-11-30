"""# Performance metrics"""

"""
Step 10: Enhanced Model Diagnostics and Visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

print("📊 Creating diagnostic visualizations...")
print("=" * 70)


# ========================================
# 5.1: Load Results
# ========================================

def load_training_results(CONFIG):
    """
    Load training results from saved pickle file.
    
    Args:
        CONFIG: Configuration dictionary
        
    Returns:
        Dictionary containing history, predictions, targets, and metrics
    """
    results_path = CONFIG['output_dir'] / 'enhanced_training_results.pkl'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found at {results_path}. Please run training first.")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


# ========================================
# 5.2: Training Curves Visualization
# ========================================

def plot_training_curves(results, CONFIG):
    """
    Plot training and validation curves including loss components and physics parameters.
    
    Args:
        results: Dictionary containing training history
        CONFIG: Configuration dictionary
    """
    hist = results['history']
    epochs = np.arange(1, len(hist['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, hist['train_loss'], linewidth=2, label='Train', alpha=0.8)
    ax.plot(epochs, hist['val_loss'], linewidth=2, label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training vs Validation Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss Components
    ax = axes[0, 1]
    ax.plot(epochs, hist['train_pred'], linewidth=2, label='Prediction (Lu)', alpha=0.8)
    ax.plot(epochs, hist['train_pde'], linewidth=2, label='PDE Residual (Lf)', alpha=0.8)
    ax.plot(epochs, hist['train_grad'], linewidth=2, label='Gradient (LfX)', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs, hist['learning_rates'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Physics Parameters Evolution
    ax = axes[1, 1]
    ax.plot(epochs, hist['alphas'], linewidth=2, label='α (effective)', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('α')
    ax.set_title('Degradation Rate Evolution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG['output_dir'] / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: training_curves.png")
    plt.show()


# ========================================
# 5.3: Predictions vs Actual Visualization
# ========================================

def plot_predictions_vs_actual(results, CONFIG):
    """
    Plot scatter plot of predictions vs actual values.
    
    Args:
        results: Dictionary containing test predictions and targets
        CONFIG: Configuration dictionary
    """
    predictions = results['test_predictions']
    targets_test = results['test_targets']
    metrics = results['metrics']
    
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_test, predictions, s=20, alpha=0.6, edgecolors='none')

    # Perfect prediction line
    min_val = min(targets_test.min(), predictions.min())
    max_val = max(targets_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual SOH', fontsize=12)
    plt.ylabel('Predicted SOH', fontsize=12)
    plt.title(f'Predictions vs Actual (R²={metrics["r2_score"]:.4f})',
              fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['output_dir'] / 'predictions_vs_actual.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: predictions_vs_actual.png")
    plt.show()


# ========================================
# 5.4: Error Distribution Visualization
# ========================================

def plot_error_distribution(results, CONFIG):
    """
    Plot error distribution histogram and error vs actual SOH.
    
    Args:
        results: Dictionary containing test predictions and targets
        CONFIG: Configuration dictionary
    """
    predictions = results['test_predictions']
    targets_test = results['test_targets']
    errors = predictions - targets_test

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error (SOH)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Error vs Actual SOH
    ax = axes[1]
    ax.scatter(targets_test, errors, s=20, alpha=0.6, edgecolors='none')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Actual SOH')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Error vs Actual SOH', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG['output_dir'] / 'error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: error_analysis.png")
    plt.show()


# ========================================
# 5.5: Physics Parameters Distribution
# ========================================

def plot_physics_parameters(results, CONFIG):
    """
    Plot distribution of learned physics parameters.
    
    Args:
        results: Dictionary containing physics parameters
        CONFIG: Configuration dictionary
    """
    alphas_test = results['test_alphas']
    betas_test = results['test_betas']
    Eas_test = results['test_Eas']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Alpha distribution
    ax = axes[0]
    ax.hist(alphas_test, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax.set_xlabel('α (effective)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Degradation Rate α\nMean: {np.mean(alphas_test):.6f}', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Beta distribution
    ax = axes[1]
    ax.hist(betas_test, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('β')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Shape Parameter β\nMean: {np.mean(betas_test):.4f}', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Ea distribution
    ax = axes[2]
    ax.hist(Eas_test / 1000, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Ea (kJ/mol)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Activation Energy Ea\nMean: {np.mean(Eas_test)/1000:.2f} kJ/mol', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG['output_dir'] / 'physics_parameters.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: physics_parameters.png")
    plt.show()


# ========================================
# 5.6: Summary Report
# ========================================

def print_summary_report(results, CONFIG):
    """
    Print comprehensive summary report of model performance.
    
    Args:
        results: Dictionary containing all results
        CONFIG: Configuration dictionary
    """
    hist = results['history']
    metrics = results['metrics']
    alphas_test = results['test_alphas']
    betas_test = results['test_betas']
    Eas_test = results['test_Eas']
    epochs = len(hist['train_loss'])

    print("\n" + "=" * 70)
    print("📋 ENHANCED MODEL SUMMARY REPORT")
    print("=" * 70)

    print("\n🎯 Training:")
    print(f"   Best Epoch:       {results['best_epoch']}")
    print(f"   Best Val Loss:    {results['best_val_loss']:.6f}")
    print(f"   Total Epochs:     {epochs}")

    print("\n📊 Test Performance:")
    print(f"   RMSE:             {metrics['rmse']:.6f} ({metrics['rmse']*100:.2f}% SOH)")
    print(f"   MAE:              {metrics['mae']:.6f} ({metrics['mae']*100:.2f}% SOH)")
    print(f"   MAPE:             {metrics['mape']:.2f}%")
    print(f"   Max Error:        {metrics['max_error']:.6f} ({metrics['max_error']*100:.2f}% SOH)")
    print(f"   R² Score:         {metrics['r2_score']:.4f}")

    print("\n⚛️ Physics Parameters (Test Set):")
    print(f"   α (effective):    {np.mean(alphas_test):.6f} ± {np.std(alphas_test):.6f}")
    print(f"   β:                {np.mean(betas_test):.4f} ± {np.std(betas_test):.4f}")
    print(f"   Ea:               {np.mean(Eas_test)/1000:.2f} ± {np.std(Eas_test)/1000:.2f} kJ/mol")

    print("\n📁 Saved Files:")
    print(f"   • enhanced_best_model.pth")
    print(f"   • enhanced_training_results.pkl")
    print(f"   • training_curves.png")
    print(f"   • predictions_vs_actual.png")
    print(f"   • error_analysis.png")
    print(f"   • physics_parameters.png")

    print("\n" + "=" * 70)


# ========================================
# 5.7: Complete Visualization Pipeline
# ========================================

def run_visualization_pipeline(CONFIG):
    """
    Run complete visualization and diagnostics pipeline.
    
    Args:
        CONFIG: Configuration dictionary
    """
    # Load results
    print("\n📂 Loading training results...")
    results = load_training_results(CONFIG)
    print("✅ Results loaded\n")
    
    # Generate all visualizations
    print("📊 Generating visualizations...\n")
    
    plot_training_curves(results, CONFIG)
    plot_predictions_vs_actual(results, CONFIG)
    plot_error_distribution(results, CONFIG)
    plot_physics_parameters(results, CONFIG)
    
    # Print summary report
    print_summary_report(results, CONFIG)
    
    print("=" * 70)
    print("✅ STEP 10 COMPLETE: Diagnostics and Visualization")
    print("=" * 70)
    
    return results


# ========================================
# Main Execution (when run directly)
# ========================================
if __name__ == "__main__":
    try:
        results = run_visualization_pipeline(CONFIG)
        print("\n✅ Visualization pipeline complete!")
        
    except NameError:
        print("⚠️  CONFIG not found. Please run 01_environment_setup.py first")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("Please run 04_training_pipeline.py first to generate results")
