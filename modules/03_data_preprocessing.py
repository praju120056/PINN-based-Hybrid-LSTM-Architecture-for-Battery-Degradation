"""# Pre-processing step"""

"""
LSTM-PINN Battery SOH Prediction - ENHANCED
Step 2 Enhanced: Data Preprocessing with Temperature & Impedance

This replaces Step 2 to extract additional physics features:
- Temperature (for Arrhenius model)
- Re (electrolyte resistance)
- Rct (charge transfer resistance)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
from typing import Dict
import matplotlib.pyplot as plt

# ========================================
# 2.1: Enhanced Data Preprocessing
# ========================================

print("🔧 Enhanced Data Preprocessing with Temperature & Impedance...")
print("=" * 70)

class EnhancedBatteryDataPreprocessor:
    """
    Enhanced preprocessor that extracts:
    - SOH sequences (as before)
    - Temperature per cycle
    - Impedance (Re, Rct) per cycle
    """

    def __init__(self, data_dir: Path, rated_capacity: float = 2.0):
        self.data_dir = data_dir
        self.rated_capacity = rated_capacity
        self.metadata_path = data_dir / 'metadata.csv'
        self.temp_scaler = StandardScaler()
        self.re_scaler = StandardScaler()
        self.rct_scaler = StandardScaler()
        self.is_fitted = False

    def extract_enhanced_sequences(self, min_cycles: int = 50):
        """
        Extract SOH + Temperature + Impedance sequences.
        INCLUDES: Data Smoothing & NaN Handling (The Final Fixes)
        """
        from scipy.signal import savgol_filter # Import filter

        print("\n📋 Loading metadata with temperature & impedance...")
        metadata = pd.read_csv(self.metadata_path)

        # Filter discharge cycles
        discharge_df = metadata[metadata['type'] == 'discharge'].copy()
        print(f"   ✓ Found {len(discharge_df)} discharge cycles")

        battery_data = {}

        for battery_id in tqdm(discharge_df['battery_id'].unique(), desc="Processing batteries"):
            battery_cycles = discharge_df[discharge_df['battery_id'] == battery_id].sort_values('start_time')

            if len(battery_cycles) < min_cycles:
                continue

            # --- FIX 1: Handle Missing Impedance (Fill NaNs) ---
            battery_cycles['Re'] = pd.to_numeric(battery_cycles['Re'], errors='coerce').ffill().bfill()
            battery_cycles['Rct'] = pd.to_numeric(battery_cycles['Rct'], errors='coerce').ffill().bfill()

            # Extract capacity
            capacities = pd.to_numeric(battery_cycles['Capacity'], errors='coerce').values
            valid_mask = ~np.isnan(capacities) & (capacities > 0)

            if np.sum(valid_mask) < min_cycles:
                continue

            # Extract features
            capacities_valid = capacities[valid_mask]
            temperatures = battery_cycles['ambient_temperature'].values[valid_mask]
            re_values = battery_cycles['Re'].values[valid_mask]
            rct_values = battery_cycles['Rct'].values[valid_mask]

            # Calculate SOH
            soh = capacities_valid / self.rated_capacity

            # --- FIX 2: TARGET SMOOTHING (Critical for Physics Loss) ---
            # Window length 15, polyorder 2 removes "sensor noise" but keeps the trend
            if len(soh) > 15:
                soh = savgol_filter(soh, window_length=15, polyorder=2)

            soh = np.clip(soh, 0.5, 1.2)

            # --- FIX 3: Safety Defaults for Physics Inputs ---
            default_re = 0.05
            default_rct = 0.10
            re_values = np.nan_to_num(re_values, nan=default_re)
            rct_values = np.nan_to_num(rct_values, nan=default_rct)

            # Store data
            battery_data[battery_id] = {
                'soh': soh,
                'temperature': temperatures,
                're': re_values,
                'rct': rct_values,
                'cycle_count': len(soh),
                'temp_mean': np.mean(temperatures),
                'temp_range': [np.min(temperatures), np.max(temperatures)],
                're_growth': re_values[-1] / (re_values[0] + 1e-6),
                'rct_growth': rct_values[-1] / (rct_values[0] + 1e-6)
            }

        print(f"\n   ✓ Processed {len(battery_data)} batteries with smoothed data")
        return battery_data
    
    def fit_scalers(self, battery_data: Dict):
        """
        Fits the scalers on the provided battery data (use TRAIN set only!).
        """
        print(f"\n📏 Fitting scalers on {len(battery_data)} batteries...")

        all_temp = []
        all_re = []
        all_rct = []

        # Collect all data points
        for data in battery_data.values():
            all_temp.extend(data['temperature'])
            all_re.extend(data['re'])
            all_rct.extend(data['rct'])

        # Fit the scalers
        self.temp_scaler.fit(np.array(all_temp).reshape(-1, 1))
        self.re_scaler.fit(np.array(all_re).reshape(-1, 1))
        self.rct_scaler.fit(np.array(all_rct).reshape(-1, 1))

        self.is_fitted = True
        print(f"   ✓ Scalers fitted. Temp Mean: {self.temp_scaler.mean_[0]:.2f}, Var: {self.temp_scaler.var_[0]:.2f}")

    def create_enhanced_sequences(self, battery_data: Dict, sequence_length: int = 20):
        if not self.is_fitted:
            print("⚠️ WARNING: Scalers not fitted! Data will be unnormalized.")
            print("   Call .fit_scalers(train_data) before creating sequences.")

        print(f"\n🔀 Creating enhanced sequences (length={sequence_length})...")

        all_soh_seq = []
        all_temp_seq = []
        all_re_seq = []
        all_rct_seq = []
        all_targets = []
        all_battery_ids = []

        for battery_id, data in battery_data.items():
            soh = data['soh']

            # RAW Features
            temp_raw = data['temperature']
            re_raw = data['re']
            rct_raw = data['rct']

            # NORMALIZE Features (if fitted)
            if self.is_fitted:
                temp = self.temp_scaler.transform(temp_raw.reshape(-1, 1)).flatten()
                re = self.re_scaler.transform(re_raw.reshape(-1, 1)).flatten()
                rct = self.rct_scaler.transform(rct_raw.reshape(-1, 1)).flatten()
            else:
                temp, re, rct = temp_raw, re_raw, rct_raw

            # Create sequences using NORMALIZED data
            for i in range(len(soh) - sequence_length):
                all_soh_seq.append(soh[i:i + sequence_length])
                all_temp_seq.append(temp[i:i + sequence_length])
                all_re_seq.append(re[i:i + sequence_length])
                all_rct_seq.append(rct[i:i + sequence_length])
                all_targets.append(soh[i + sequence_length])
                all_battery_ids.append(battery_id)

        # Convert to arrays (same as before)
        soh_sequences = np.array(all_soh_seq, dtype=np.float32)
        temp_sequences = np.array(all_temp_seq, dtype=np.float32)
        re_sequences = np.array(all_re_seq, dtype=np.float32)
        rct_sequences = np.array(all_rct_seq, dtype=np.float32)
        targets = np.array(all_targets, dtype=np.float32)
        battery_ids = np.array(all_battery_ids)

        print(f"   ✓ Created {len(soh_sequences)} sequences (Normalized: {self.is_fitted})")
        return soh_sequences, temp_sequences, re_sequences, rct_sequences, targets, battery_ids

print("✅ EnhancedBatteryDataPreprocessor Updated")

# ========================================
# 2.2: Enhanced Dataset Class
# ========================================

print("📦 Defining enhanced dataset class...")
print("=" * 70)

class EnhancedSOHDataset(Dataset):
    """
    Enhanced dataset with temperature and impedance features.
    """

    def __init__(self, soh_seq, temp_seq, re_seq, rct_seq, targets):
        # SOH sequences (main input)
        self.soh_sequences = torch.FloatTensor(soh_seq).unsqueeze(-1)  # (N, seq, 1)

        # Auxiliary physics features
        self.temp_sequences = torch.FloatTensor(temp_seq)  # (N, seq)
        self.re_sequences = torch.FloatTensor(re_seq)      # (N, seq)
        self.rct_sequences = torch.FloatTensor(rct_seq)    # (N, seq)

        # Targets
        self.targets = torch.FloatTensor(targets)

        # Time steps
        self.time_steps = torch.arange(len(soh_seq), dtype=torch.float32)

    def __len__(self):
        return len(self.soh_sequences)

    def __getitem__(self, idx):
        return (
            self.soh_sequences[idx],
            self.temp_sequences[idx],
            self.re_sequences[idx],
            self.rct_sequences[idx],
            self.targets[idx],
            self.time_steps[idx]
        )

print("✓ Enhanced dataset class defined")
print()


# ========================================
# 2.3: Data Preprocessing Pipeline
# ========================================

def run_preprocessing_pipeline(CONFIG, force_preprocess=True):
    """
    Run the complete enhanced preprocessing pipeline.
    
    Args:
        CONFIG: Configuration dictionary from environment setup
        force_preprocess: Whether to ignore cache and reprocess data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_ids, val_ids, test_ids)
    """
    # Verify Step 1 completed
    try:
        assert 'data_dir' in CONFIG, "CONFIG['data_dir'] not found. Run Step 1 first."
        assert CONFIG['data_dir'].exists(), "Data directory not found."
        print("✅ Prerequisites verified\n")
    except AssertionError as e:
        print(f"❌ Error: {e}")
        print("Please run 01_environment_setup.py first!")
        raise

    # Use the enhanced preprocessor
    preprocessor_enhanced = EnhancedBatteryDataPreprocessor(
        data_dir=CONFIG['data_dir'],
        rated_capacity=2.0
    )

    # ========================================
    # 1. Extract Enhanced Data (Raw)
    # ========================================

    print("🔄 Extracting enhanced battery data...")
    print("=" * 70)

    # Check for cache
    enhanced_cache_path = CONFIG['data_dir'] / 'preprocessed_battery_data_enhanced.pkl'

    if enhanced_cache_path.exists() and not force_preprocess:
        print(f"\n✓ Found enhanced cache at {enhanced_cache_path}")
        print("  Loading from cache...")
        with open(enhanced_cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        battery_data_enhanced = cached_data['battery_data']
        print(f"  Loaded {len(battery_data_enhanced)} batteries")
    else:
        print("\n⚙️  Running enhanced preprocessing...")
        battery_data_enhanced = preprocessor_enhanced.extract_enhanced_sequences(
            min_cycles=CONFIG['min_cycles']
        )
        # Save cache
        cache_data = {
            'battery_data': battery_data_enhanced,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open(enhanced_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"\n✓ Cached for future runs: {enhanced_cache_path}")

    print()

    # ========================================
    # 2. Split Data by Battery (BEFORE Sequencing)
    # ========================================

    print("🎯 Splitting data by battery...")
    print("=" * 70)

    # Get unique battery IDs
    all_battery_ids = list(battery_data_enhanced.keys())
    n_batteries = len(all_battery_ids)

    # Shuffle IDs
    np.random.seed(42)
    np.random.shuffle(all_battery_ids)

    # Define Split Sizes
    train_end = int(0.70 * n_batteries)
    val_end = train_end + int(0.15 * n_batteries)

    train_ids = all_battery_ids[:train_end]
    val_ids = all_battery_ids[train_end:val_end]
    test_ids = all_battery_ids[val_end:]

    # Create Dictionaries for each split
    train_data_dict = {bid: battery_data_enhanced[bid] for bid in train_ids}
    val_data_dict = {bid: battery_data_enhanced[bid] for bid in val_ids}
    test_data_dict = {bid: battery_data_enhanced[bid] for bid in test_ids}

    print(f"✓ Split counts: Train {len(train_data_dict)}, Val {len(val_data_dict)}, Test {len(test_data_dict)}")
    print()

    # ========================================
    # 3. Fit Scalers & Create Sequences
    # ========================================

    print("📏 Fitting Scalers & Creating Sequences...")
    print("=" * 70)

    # CRITICAL: Fit scalers ONLY on Training Data
    preprocessor_enhanced.fit_scalers(train_data_dict)

    # Create sequences using the fitted scalers
    print("\n-- Processing Training Set --")
    train_soh, train_temp, train_re, train_rct, train_targets, battery_ids_train = preprocessor_enhanced.create_enhanced_sequences(
        train_data_dict, sequence_length=CONFIG['sequence_length']
    )

    print("\n-- Processing Validation Set --")
    val_soh, val_temp, val_re, val_rct, val_targets, battery_ids_val = preprocessor_enhanced.create_enhanced_sequences(
        val_data_dict, sequence_length=CONFIG['sequence_length']
    )

    print("\n-- Processing Test Set --")
    test_soh, test_temp, test_re, test_rct, test_targets, battery_ids_test = preprocessor_enhanced.create_enhanced_sequences(
        test_data_dict, sequence_length=CONFIG['sequence_length']
    )

    print()

    # ========================================
    # 4. Create Enhanced Datasets
    # ========================================

    print("📦 Creating enhanced datasets...")
    print("=" * 70)

    train_dataset_enh = EnhancedSOHDataset(train_soh, train_temp, train_re, train_rct, train_targets)
    val_dataset_enh = EnhancedSOHDataset(val_soh, val_temp, val_re, val_rct, val_targets)
    test_dataset_enh = EnhancedSOHDataset(test_soh, test_temp, test_re, test_rct, test_targets)

    print(f"✓ Enhanced datasets created")
    print(f"   Features per sample: SOH, Temp (Norm), Re (Norm), Rct (Norm)")
    print()

    # ========================================
    # 5. Create Enhanced DataLoaders
    # ========================================

    print("🔄 Creating enhanced dataloaders...")
    print("=" * 70)

    device = CONFIG['device']
    
    train_loader_enh = DataLoader(
        train_dataset_enh, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )

    val_loader_enh = DataLoader(
        val_dataset_enh, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )

    test_loader_enh = DataLoader(
        test_dataset_enh, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )

    print(f"✓ Enhanced dataloaders ready")
    print(f"   Train batches: {len(train_loader_enh)}")
    print(f"   Val batches: {len(val_loader_enh)}")
    print(f"   Test batches: {len(test_loader_enh)}")
    print()

    # ========================================
    # 6. Visualization (Updated with Vibrant Colors)
    # ========================================
    # Visualizing normalized features to confirm scaling
    print("📊 Visualizing features (first training batch)...")
    sample_soh, sample_temp, sample_re, sample_rct, _, _ = next(iter(train_loader_enh))

    # Define distinct, vibrant colors
    colors = {
        'soh': '#FF1744',      # Bright Red
        'temp': '#00E676',     # Bright Green
        're': '#2979FF',       # Bright Blue
        'rct': '#FF6D00'       # Bright Orange
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # SOH plot
    axes[0,0].plot(sample_soh[0].numpy(), color=colors['soh'], linewidth=2)
    axes[0,0].set_title("SOH (Raw)", fontweight='bold', fontsize=11)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylabel('SOH', fontweight='bold')

    # Temperature plot
    axes[0,1].plot(sample_temp[0].numpy(), color=colors['temp'], linewidth=2)
    axes[0,1].set_title("Temperature (Normalized)", fontweight='bold', fontsize=11)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylabel('Norm. Temp', fontweight='bold')

    # Re plot
    axes[1,0].plot(sample_re[0].numpy(), color=colors['re'], linewidth=2)
    axes[1,0].set_title("Re (Normalized)", fontweight='bold', fontsize=11)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylabel('Norm. Re', fontweight='bold')
    axes[1,0].set_xlabel('Cycle', fontweight='bold')

    # Rct plot
    axes[1,1].plot(sample_rct[0].numpy(), color=colors['rct'], linewidth=2)
    axes[1,1].set_title("Rct (Normalized)", fontweight='bold', fontsize=11)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylabel('Norm. Rct', fontweight='bold')
    axes[1,1].set_xlabel('Cycle', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("✅ Captured battery IDs")
    print("Train IDs:", np.unique(battery_ids_train)[:5])
    print("Val IDs:", np.unique(battery_ids_val)[:5])
    print("Test IDs:", np.unique(battery_ids_test)[:5])

    print("\n✅ STEP 2 REPLACEMENT COMPLETE")
    
    return train_loader_enh, val_loader_enh, test_loader_enh, battery_ids_train, battery_ids_val, battery_ids_test


# ========================================
# Main Execution (when run directly)
# ========================================
if __name__ == "__main__":
    try:
        # Run preprocessing
        train_loader, val_loader, test_loader, train_ids, val_ids, test_ids = run_preprocessing_pipeline(
            CONFIG, 
            force_preprocess=True
        )
        print("\n✅ Data preprocessing complete!")
        
    except NameError:
        print("⚠️  CONFIG not found. Please run 01_environment_setup.py first")
