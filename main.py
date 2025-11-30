"""
PINN-based Hybrid LSTM Architecture - Main Orchestrator
========================================================

This file coordinates the execution of all modules for the complete workflow.
"""

import sys
from pathlib import Path

# Add modules directory to path
modules_dir = Path(__file__).parent / 'modules'
sys.path.insert(0, str(modules_dir))

print("=" * 80)
print("PINN-BASED HYBRID LSTM ARCHITECTURE FOR BATTERY SOH PREDICTION")
print("=" * 80)
print("\nThis workflow consists of 5 main steps:")
print("  1. Environment Setup")
print("  2. Model Architecture Definitions")
print("  3. Data Preprocessing")
print("  4. Training Pipeline (with Bayesian Optimization)")
print("  5. Evaluation and Visualization")
print("\n" + "=" * 80)

# ========================================
# STEP 1: Environment Setup
# ========================================
print("\n🚀 STEP 1: Environment Setup")
print("-" * 80)
exec(open('modules/01_environment_setup.py').read())

# ========================================
# STEP 2: Model Architecture Definitions
# ========================================
print("\n🏗️  STEP 2: Model Architecture Definitions")
print("-" * 80)
exec(open('modules/02_model_architectures.py').read())

# ========================================
# STEP 3: Data Preprocessing
# ========================================
print("\n📊 STEP 3: Data Preprocessing")
print("-" * 80)
from modules.data_preprocessing import run_preprocessing_pipeline

# Run preprocessing
train_loader, val_loader, test_loader, train_ids, val_ids, test_ids = run_preprocessing_pipeline(
    CONFIG, 
    force_preprocess=False  # Set to True to ignore cache
)

# Store datasets for training
train_dataset_enh = train_loader.dataset
val_dataset_enh = val_loader.dataset
test_dataset_enh = test_loader.dataset

print("\n✅ Data preprocessing complete!")

# ========================================
# STEP 4: Training Pipeline
# ========================================
print("\n🎓 STEP 4: Training Pipeline with Bayesian Optimization")
print("-" * 80)
from modules.training_pipeline import run_training_pipeline

# Run training with Bayesian optimization
optimal_model, history, results, study = run_training_pipeline(
    train_dataset_enh, 
    val_dataset_enh, 
    test_dataset_enh,
    CONFIG,
    n_trials=30  # Adjust based on compute budget
)

print("\n✅ Training pipeline complete!")

# ========================================
# STEP 5: Evaluation and Visualization
# ========================================
print("\n📈 STEP 5: Evaluation and Visualization")
print("-" * 80)
from modules.evaluation_visualization import run_visualization_pipeline

# Generate visualizations and summary report
results = run_visualization_pipeline(CONFIG)

print("\n✅ Visualization pipeline complete!")

# ========================================
# WORKFLOW COMPLETE
# ========================================
print("\n" + "=" * 80)
print("✅ COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
print("=" * 80)
print("\n📁 Output files saved to:", CONFIG['output_dir'])
print("\nNext steps:")
print("  - Review the generated visualizations")
print("  - Check model performance metrics")
print("  - Load the trained model from 'enhanced_best_model.pth' for inference")
print("\n" + "=" * 80)
