"""# Training the LSTM model"""

"""
LSTM-PINN Battery SOH Prediction
Step 9 (ENHANCED with Bayesian Optimization): Training the Enhanced Model

This cell:
1. Uses Optuna for Bayesian hyperparameter optimization
2. Uses EnhancedLSTMPINNModel from Step 8
3. Uses enhanced dataloaders from Step 2
4. Finds optimal hyperparameters, then trains final model
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
import time
import pickle
from pathlib import Path
from torch.utils.data import DataLoader

# ========================================
# 4.0: Check Optuna Installation
# ========================================

print("📦 Checking Optuna installation...")
print("=" * 70)
try:
    import optuna
    print(f"✅ Optuna already installed (v{optuna.__version__})")
except ImportError:
    print("Installing Optuna...")
    !pip install -q optuna
    import optuna
    print(f"✅ Optuna installed (v{optuna.__version__})")
print()


# ========================================
# 4.1: Define Training Function (No Early Stopping)
# ========================================

print("🚀 Defining training function for Enhanced Model...")
print("=" * 70)

def train_enhanced_model(model, train_loader, val_loader,
                         epochs=200, learning_rate=0.001,
                         lambda_pde=0.1, lambda_grad=0.01,
                         device='cuda', verbose=True):
    """
    Train the Enhanced LSTM-PINN model.

    Args:
        model: EnhancedLSTMPINNModel
        train_loader: DataLoader with EnhancedSOHDataset
        val_loader: DataLoader with EnhancedSOHDataset
        verbose: If False, reduces print output (for Optuna trials)
    """
    # Import required classes
    from modules.model_architectures import EnhancedPhysicsLoss
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-6
    )

    criterion = EnhancedPhysicsLoss(
        lambda_pred=1.0,
        lambda_pde=lambda_pde,
        lambda_grad=lambda_grad
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_pred': [], 'train_pde': [], 'train_grad': [],
        'val_pred': [], 'learning_rates': [],
        'alphas': [], 'betas': [], 'Eas': []
    }

    best_val_loss = float('inf')
    best_epoch = 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"🎯 TRAINING ENHANCED MODEL")
        print(f"{'='*70}")
        print(f"  λ_PDE: {lambda_pde:.4f}")
        print(f"  λ_grad: {lambda_grad:.4f}")
        print(f"  Learning Rate: {learning_rate:.6f}")
        print(f"  Max Epochs: {epochs}")
        print(f"  Early Stopping: DISABLED")
        print(f"{'='*70}\n")

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training Phase
        model.train()
        train_losses = {'total': [], 'pred': [], 'pde': [], 'grad': []}

        if verbose:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        else:
            train_pbar = train_loader

        for batch in train_pbar:
            soh_seq, temp_seq, re_seq, rct_seq, targets, time_steps = batch

            soh_seq = soh_seq.to(device)
            temp_seq = temp_seq.to(device)
            re_seq = re_seq.to(device)
            rct_seq = rct_seq.to(device)
            targets = targets.to(device)
            time_steps = time_steps.to(device)

            optimizer.zero_grad()
            outputs = model(soh_seq, temp_seq, re_seq, rct_seq, time_steps)
            losses = criterion(outputs, targets, time_steps, model)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses['total'].append(losses['total'].item())
            train_losses['pred'].append(losses['prediction'].item())
            train_losses['pde'].append(losses['pde_residual'].item())
            train_losses['grad'].append(losses['gradient'].item())

            if verbose and hasattr(train_pbar, 'set_postfix'):
                train_pbar.set_postfix({'loss': f"{losses['total'].item():.6f}"})

        # Validation Phase
        model.eval()
        val_losses = {'total': [], 'pred': []}
        epoch_alphas, epoch_betas, epoch_Eas = [], [], []

        with torch.no_grad():
            if verbose:
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            else:
                val_pbar = val_loader

            for batch in val_pbar:
                soh_seq, temp_seq, re_seq, rct_seq, targets, time_steps = batch

                soh_seq = soh_seq.to(device)
                temp_seq = temp_seq.to(device)
                re_seq = re_seq.to(device)
                rct_seq = rct_seq.to(device)
                targets = targets.to(device)
                time_steps = time_steps.to(device)

                outputs = model(soh_seq, temp_seq, re_seq, rct_seq, time_steps)
                losses = criterion(outputs, targets, time_steps, model)

                val_losses['total'].append(losses['total'].item())
                val_losses['pred'].append(losses['prediction'].item())

                epoch_alphas.append(outputs['alpha'].cpu().numpy())
                epoch_betas.append(outputs['beta'].cpu().numpy())
                epoch_Eas.append(outputs['Ea'].cpu().numpy())

                if verbose and hasattr(val_pbar, 'set_postfix'):
                    val_pbar.set_postfix({'loss': f"{losses['total'].item():.6f}"})

        # Statistics
        avg_train_loss = np.mean(train_losses['total'])
        avg_val_loss = np.mean(val_losses['total'])
        avg_train_pred = np.mean(train_losses['pred'])
        avg_train_pde = np.mean(train_losses['pde'])
        avg_train_grad = np.mean(train_losses['grad'])
        avg_val_pred = np.mean(val_losses['pred'])
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_pred'].append(avg_train_pred)
        history['train_pde'].append(avg_train_pde)
        history['train_grad'].append(avg_train_grad)
        history['val_pred'].append(avg_val_pred)
        history['learning_rates'].append(current_lr)
        history['alphas'].append(np.mean(np.concatenate(epoch_alphas)))
        history['betas'].append(np.mean(np.concatenate(epoch_betas)))
        history['Eas'].append(np.mean(np.concatenate(epoch_Eas)))

        scheduler.step(avg_val_loss)

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            checkpoint_msg = "✅ New best"
        else:
            checkpoint_msg = f"Best: Epoch {best_epoch}"

        # Print Progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"{'='*70}")
            print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            print(f"  Components: Pred={avg_train_pred:.6f}, PDE={avg_train_pde:.6f}, Grad={avg_train_grad:.6f}")
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
            print(f"  {checkpoint_msg} (Val Loss: {best_val_loss:.6f})")

    training_time = time.time() - training_start

    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"  Total Time: {training_time/60:.2f} minutes")
        print(f"  Best Epoch: {best_epoch}/{epochs}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
        print()

    return history, best_val_loss, best_epoch

print("✅ Training function defined")
print()


# ========================================
# 4.2: Define Bayesian Optimization Objective
# ========================================

print("🎯 Defining Bayesian Optimization objective...")
print("=" * 70)

def objective_enhanced(trial, train_dataset_enh, val_dataset_enh, CONFIG, device):
    """
    Optuna objective function for hyperparameter search.
    """
    # Import required classes
    from modules.model_architectures import EnhancedLSTMPINNModel
    
    # Sample hyperparameters
    lambda_pde = trial.suggest_float('lambda_pde', 0.05, 0.5, log=True)
    lambda_grad = trial.suggest_float('lambda_grad', 0.0, 0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    lstm_hidden = trial.suggest_categorical('lstm_hidden', [64, 128, 256])
    physics_hidden = trial.suggest_categorical('physics_hidden', [128, 256])
    batch_size = trial.suggest_categorical('batch_size', [32, 64])

    print(f"\n{'='*70}")
    print(f"Trial {trial.number + 1}")
    print(f"{'='*70}")
    print(f"  λ_PDE: {lambda_pde:.6f}")
    print(f"  λ_grad: {lambda_grad:.6f}")
    print(f"  LR: {learning_rate:.6f}")
    print(f"  LSTM: {lstm_hidden}, Physics: {physics_hidden}")
    print(f"  Batch: {batch_size}")

    # Create trial-specific dataloaders
    trial_train_loader = DataLoader(
        train_dataset_enh, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )
    trial_val_loader = DataLoader(
        val_dataset_enh, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )

    # Initialize model
    trial_model = EnhancedLSTMPINNModel(
        lstm_hidden=lstm_hidden,
        lstm_layers=CONFIG['lstm_layers'],
        physics_hidden=physics_hidden,
        physics_layers=CONFIG['physics_layers']
    ).to(device)

    # Train for limited epochs (for speed)
    TRIAL_EPOCHS = 50  # Reduced for faster search

    _, best_val_loss, _ = train_enhanced_model(
        model=trial_model,
        train_loader=trial_train_loader,
        val_loader=trial_val_loader,
        epochs=TRIAL_EPOCHS,
        learning_rate=learning_rate,
        lambda_pde=lambda_pde,
        lambda_grad=lambda_grad,
        device=device,
        verbose=False  # Quiet mode for trials
    )

    print(f"  Final val loss: {best_val_loss:.6f}")

    return best_val_loss

print("✅ Objective function defined")
print()


# ========================================
# 4.3: Run Full Training Pipeline with Bayesian Optimization
# ========================================

def run_training_pipeline(train_dataset_enh, val_dataset_enh, test_dataset_enh, CONFIG, n_trials=30):
    """
    Run complete training pipeline with Bayesian optimization.
    
    Args:
        train_dataset_enh: Training dataset
        val_dataset_enh: Validation dataset
        test_dataset_enh: Test dataset
        CONFIG: Configuration dictionary
        n_trials: Number of Bayesian optimization trials
        
    Returns:
        Tuple of (optimal_model, history, results, study)
    """
    # Import required classes
    from modules.model_architectures import EnhancedLSTMPINNModel
    
    device = CONFIG['device']
    
    print(f"\n⚙️ Starting Bayesian Optimization...")
    print(f"{'='*70}")
    print(f"  Trials: {n_trials}")
    print(f"  Epochs per trial: 50")
    print(f"  Model: EnhancedLSTMPINNModel")
    print(f"  Dataset: EnhancedSOHDataset")
    print(f"\n{'='*70}\n")

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    optimization_start = time.time()

    try:
        study.optimize(
            lambda trial: objective_enhanced(trial, train_dataset_enh, val_dataset_enh, CONFIG, device),
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted")

    optimization_time = time.time() - optimization_start

    print(f"\n{'='*70}")
    print(f"✅ BAYESIAN OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Time: {optimization_time/60:.1f} minutes")
    print(f"  Trials completed: {len(study.trials)}")
    print()
    print(f"🏆 Best Trial:")
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print(f"     {key}: {value}")
    print(f"  Best val loss: {best_trial.value:.6f}")
    print()


    # ========================================
    # Train Final Model with Optimal Hyperparameters
    # ========================================

    print("🎯 Training FINAL model with optimal hyperparameters...")
    print("=" * 70)

    # Create optimal model
    optimal_model = EnhancedLSTMPINNModel(
        lstm_hidden=best_trial.params['lstm_hidden'],
        lstm_layers=CONFIG['lstm_layers'],
        physics_hidden=best_trial.params['physics_hidden'],
        physics_layers=CONFIG['physics_layers']
    ).to(device)

    # Create optimal dataloaders
    optimal_batch_size = best_trial.params['batch_size']

    optimal_train_loader = DataLoader(
        train_dataset_enh, batch_size=optimal_batch_size, shuffle=True,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )
    optimal_val_loader = DataLoader(
        val_dataset_enh, batch_size=optimal_batch_size, shuffle=False,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )
    optimal_test_loader = DataLoader(
        test_dataset_enh, batch_size=optimal_batch_size, shuffle=False,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )

    print(f"Optimal Batch Size: {optimal_batch_size}")
    print(f"Train batches: {len(optimal_train_loader)}")
    print(f"Val batches: {len(optimal_val_loader)}")
    print(f"Test batches: {len(optimal_test_loader)}")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in optimal_model.parameters())
    trainable_params = sum(p.numel() for p in optimal_model.parameters() if p.requires_grad)

    print(f"✅ Optimal model initialized")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()

    # Train for FULL epochs (200)
    history_optimal, best_val_loss_optimal, best_epoch_optimal = train_enhanced_model(
        model=optimal_model,
        train_loader=optimal_train_loader,
        val_loader=optimal_val_loader,
        epochs=CONFIG['epochs'],  # Full 200 epochs
        learning_rate=best_trial.params['learning_rate'],
        lambda_pde=best_trial.params['lambda_pde'],
        lambda_grad=best_trial.params['lambda_grad'],
        device=device,
        verbose=True
    )

    # Save optimal model
    torch.save({
        'model_state_dict': optimal_model.state_dict(),
        'hyperparameters': best_trial.params,
        'best_epoch': best_epoch_optimal,
        'best_val_loss': best_val_loss_optimal
    }, CONFIG['output_dir'] / 'enhanced_best_model.pth')


    # ========================================
    # Evaluate on Test Set
    # ========================================

    print("📊 Evaluating optimal model on test set...")
    print("=" * 70)

    optimal_model.eval()

    all_predictions = []
    all_targets = []
    all_alpha = []
    all_beta = []
    all_Ea = []

    with torch.no_grad():
        for batch in tqdm(optimal_test_loader, desc="Testing"):
            soh_seq, temp_seq, re_seq, rct_seq, targets, time_steps = batch

            soh_seq = soh_seq.to(device)
            temp_seq = temp_seq.to(device)
            re_seq = re_seq.to(device)
            rct_seq = rct_seq.to(device)
            targets = targets.to(device)
            time_steps = time_steps.to(device)

            outputs = optimal_model(soh_seq, temp_seq, re_seq, rct_seq, time_steps)

            all_predictions.append(outputs['soh_pred'].squeeze().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_alpha.append(outputs['alpha'].cpu().numpy())
            all_beta.append(outputs['beta'].cpu().numpy())
            all_Ea.append(outputs['Ea'].cpu().numpy())

    predictions = np.concatenate(all_predictions)
    targets_test = np.concatenate(all_targets)
    alphas_test = np.concatenate(all_alpha)
    betas_test = np.concatenate(all_beta)
    Eas_test = np.concatenate(all_Ea)

    # Compute metrics
    rmse = np.sqrt(np.mean((predictions - targets_test) ** 2))
    mae = np.mean(np.abs(predictions - targets_test))
    mape = np.mean(np.abs((predictions - targets_test) / (targets_test + 1e-8))) * 100
    max_error = np.max(np.abs(predictions - targets_test))
    r2 = 1 - (np.sum((targets_test - predictions) ** 2) /
              np.sum((targets_test - np.mean(targets_test)) ** 2))

    print(f"\n✅ Evaluation complete")
    print(f"\n📊 OPTIMAL ENHANCED MODEL TEST PERFORMANCE:")
    print(f"{'='*70}")
    print(f"   RMSE:       {rmse:.6f}  ({rmse*100:.2f}% SOH)")
    print(f"   MAE:        {mae:.6f}  ({mae*100:.2f}% SOH)")
    print(f"   MAPE:       {mape:.2f}%")
    print(f"   Max Error:  {max_error:.6f}  ({max_error*100:.2f}% SOH)")
    print(f"   R² Score:   {r2:.4f}")
    print()
    print(f"   Physics Parameters (Test Set Averages):")
    print(f"      α (effective): {np.mean(alphas_test):.6f} ± {np.std(alphas_test):.6f}")
    print(f"      β:             {np.mean(betas_test):.4f} ± {np.std(betas_test):.4f}")
    print(f"      Ea:            {np.mean(Eas_test)/1000:.2f} ± {np.std(Eas_test)/1000:.2f} kJ/mol")
    print()


    # ========================================
    # Save Results
    # ========================================

    print("💾 Saving comprehensive results...")
    print("=" * 70)

    # Save optimization results
    optimization_results = {
        'study': study,
        'best_trial': best_trial,
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
        'optimization_time': optimization_time
    }
    with open(CONFIG['output_dir'] / 'bayesian_optimization.pkl', 'wb') as f:
        pickle.dump(optimization_results, f)

    # Save training and evaluation results
    results = {
        'history': history_optimal,
        'best_epoch': best_epoch_optimal,
        'best_val_loss': best_val_loss_optimal,
        'test_predictions': predictions,
        'test_targets': targets_test,
        'test_alphas': alphas_test,
        'test_betas': betas_test,
        'test_Eas': Eas_test,
        'hyperparameters': best_trial.params,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'max_error': max_error,
            'r2_score': r2
        }
    }

    with open(CONFIG['output_dir'] / 'enhanced_training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ Results saved to {CONFIG['output_dir']}")
    print(f"   • bayesian_optimization.pkl")
    print(f"   • enhanced_training_results.pkl")
    print(f"   • enhanced_best_model.pth")
    print()

    print("=" * 70)
    print("✅ STEP 9 COMPLETE: Enhanced Model with Bayesian Optimization")
    print("=" * 70)
    
    return optimal_model, history_optimal, results, study


# ========================================
# Main Execution (when run directly)
# ========================================
if __name__ == "__main__":
    try:
        # Note: This assumes train_dataset_enh, val_dataset_enh, test_dataset_enh are already created
        # from running 03_data_preprocessing.py
        
        optimal_model, history, results, study = run_training_pipeline(
            train_dataset_enh, 
            val_dataset_enh, 
            test_dataset_enh,
            CONFIG,
            n_trials=30
        )
        print("\n✅ Training pipeline complete!")
        
    except NameError as e:
        print(f"⚠️  Required variables not found: {e}")
        print("Please run 01_environment_setup.py and 03_data_preprocessing.py first")
