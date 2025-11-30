"""# Architecture class definition"""

"""
Step 8 Prerequisites:Class Definitions
"""

import torch
import torch.nn as nn

print("🔧 Defining CorrectedLSTMExtractor...")
print("=" * 70)

class CorrectedLSTMExtractor(nn.Module):
    """
    LSTM feature extractor that returns the last hidden state.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(CorrectedLSTMExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            last_hidden: (batch, hidden_size)
            hidden_states: tuple of (hn, cn)
        """
        lstm_out, (hn, cn) = self.lstm(x)
        # Use last hidden state from the last LSTM layer
        last_hidden = hn[-1]  # Shape: (batch, hidden_size)
        return last_hidden, (hn, cn)

print("✅ CorrectedLSTMExtractor defined")
print()

"""
Step 8: Enhanced Physics Loss Function (Corrected)
"""

print("⚖️ Defining EnhancedPhysicsLoss...")
print("=" * 70)

class EnhancedPhysicsLoss(nn.Module):
    """
    Multi-component loss for Enhanced LSTM-PINN:
    1. Prediction Loss (Lu): Huber Loss (Robust to outliers as it is a combo of mse and mae)
    2. PDE Residual Loss (Lf): SmoothL1Loss (Robust for physics gradients)
    3. Gradient Loss (LfX): Monotonicity constraint
    """

    def __init__(self, lambda_pred=1.0, lambda_pde=0.1, lambda_grad=0.01):
        super(EnhancedPhysicsLoss, self).__init__()
        self.lambda_pred = lambda_pred
        self.lambda_pde = lambda_pde
        self.lambda_grad = lambda_grad

        # FIX 1: Use Huber Loss for prediction (Critical for outlier handling)
        # delta=0.02 means errors < 2% are squared (precise), > 2% are linear (robust)
        self.prediction_loss_fn = nn.HuberLoss(delta=0.02)

        # FIX 2: Use SmoothL1Loss for physics gradients
        self.pde_loss_fn = nn.SmoothL1Loss()

    def forward(self, outputs, targets, time_steps, model):
        """
        Args:
            outputs: dict from EnhancedLSTMPINNModel.forward()
            targets: true next SOH values
            time_steps: time indices
            model: the EnhancedLSTMPINNModel instance
        """
        # 1. Prediction Loss (Lu)
        soh_pred = outputs['soh_pred'].squeeze()

        # CRITICAL FIX: Use the Huber Loss defined in __init__
        # (The previous code tried to use self.mse_loss which did not exist)
        loss_pred = self.prediction_loss_fn(soh_pred, targets)

        # Current SOH (last point in input sequence)
        soh_current = outputs['soh_current'].squeeze()

        # 2. PDE Residual Loss (Lf)
        # Data-driven gradient: dSOH/dt ≈ (SOH_pred - SOH_current) / 1 cycle
        dSOH_dt_data = (soh_pred - soh_current) / 1.0

        # Physics-driven gradient from DeepHTPM
        dSOH_dt_phys = model.physics_head.compute_physics_derivative(
            soh_current,
            outputs['alpha'],
            outputs['beta'],
            time_steps
        ).squeeze()

        # Use robust loss for PDE consistency
        loss_pde = self.pde_loss_fn(dSOH_dt_data, dSOH_dt_phys)

        # 3. Gradient/Monotonicity Loss (LfX)
        # Penalize if predicted SOH > current SOH (should decrease)
        # This stops the model from predicting that the battery "healed" itself
        loss_grad = torch.relu(soh_pred - soh_current).mean()

        # Total Loss
        total_loss = (
            self.lambda_pred * loss_pred +
            self.lambda_pde * loss_pde +
            self.lambda_grad * loss_grad
        )

        return {
            'total': total_loss,
            'prediction': loss_pred,
            'pde_residual': loss_pde,
            'gradient': loss_grad
        }

print("✅ EnhancedPhysicsLoss defined")
print()

"""
LSTM-PINN Battery SOH Prediction - ENHANCED PHYSICS
Step 8 Enhanced: Temperature-Aware + Impedance-Based Physics

NEW Physics Constraints:
1. Arrhenius temperature dependence: α(T) = α₀ × exp(Ea/R × (1/T_ref - 1/T))
2. Impedance-based degradation: Degradation ∝ (Re + Rct) growth
3. Combined multi-physics model

Changes from original Step 8:
- Enhanced data preprocessing (extract temp & impedance)
- Modified dataset to include auxiliary physics features
- Enhanced DeepHTPM with temperature and impedance inputs
- Physics loss accounts for temperature effects
"""

# ========================================
# 8E.3: Enhanced DeepHTPM with Multi-Physics
# ========================================

print("⚛️  Defining Enhanced DeepHTPM with Temperature & Impedance Physics...")
print("=" * 70)

"""
FIXED EnhancedDeepHTPM with Numerical Stability
Replace the existing class in Step 8 with this version
"""

class EnhancedDeepHTPM(nn.Module):
    """
    Enhanced DeepHTPM with multi-physics constraints and numerical stability.

    Fixes:
    1. Safe temperature conversion (always Celsius → Kelvin)
    2. Clamped exponentials to prevent overflow
    3. Protected division by zero
    4. Input validation
    """

    def __init__(self, lstm_hidden_size: int = 64,
                 physics_hidden_size: int = 128,
                 num_physics_layers: int = 3):
        super(EnhancedDeepHTPM, self).__init__()

        # Physics constants
        self.R = 8.314  # Gas constant (J/mol·K)
        self.T_ref = 298.15  # Reference temperature (25°C in Kelvin)

        # Network to predict BASE physics parameters
        in_features = lstm_hidden_size + 1 + 3  # +1 time, +3 physics features

        layers = []
        for i in range(num_physics_layers - 1):
            layers.extend([
                nn.Linear(in_features if i == 0 else physics_hidden_size,
                         physics_hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1)
            ])

        # Output: [α_base, β, Ea]
        layers.append(nn.Linear(physics_hidden_size, 3))
        self.physics_net = nn.Sequential(*layers)

        # Impedance influence network
        self.impedance_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def compute_temperature_factor(self, temperature, Ea):
        """
        Arrhenius temperature dependence with numerical stability.

        Args:
            temperature: (batch,) - Temperature in Celsius (e.g., 4, 24)
            Ea: (batch, 1) - Activation energy (J/mol)

        Returns:
            temp_factor: (batch, 1) - Temperature scaling factor
        """
        # Temperature from metadata is in Celsius (e.g., 4, 24)
        # Always convert to Kelvin
        T_kelvin = temperature + 273.15

        # Handle dimensions
        if T_kelvin.dim() == 1:
            T_kelvin = T_kelvin.unsqueeze(1)

        # Clamp Ea to reasonable range (5-80 kJ/mol)
        Ea_clamped = torch.clamp(Ea, 5000, 80000)

        # Compute Arrhenius exponent: Ea/R × (1/T_ref - 1/T)
        exponent = Ea_clamped / self.R * (1.0 / self.T_ref - 1.0 / T_kelvin)

        # CRITICAL: Clamp exponent to prevent overflow/underflow
        # exp(10) ≈ 22000, exp(-10) ≈ 0.000045
        exponent = torch.clamp(exponent, -10.0, 10.0)

        # Compute temperature factor
        temp_factor = torch.exp(exponent)

        # Additional safety clamp
        temp_factor = torch.clamp(temp_factor, 0.1, 10.0)

        return temp_factor

    def compute_impedance_factor(self, re, rct, re_initial, rct_initial):
        """
        Impedance-based degradation correlation with zero-division protection.

        Args:
            re, rct: Current impedance values
            re_initial, rct_initial: Initial impedance values

        Returns:
            impedance_factor: (batch, 1) - Impedance scaling factor
        """
        # Protect against division by zero
        eps = 1e-6
        re_growth = re / (re_initial + eps)
        rct_growth = rct / (rct_initial + eps)

        # Clamp to reasonable ranges (0.5x to 5x growth)
        re_growth = torch.clamp(re_growth, 0.5, 5.0)
        rct_growth = torch.clamp(rct_growth, 0.5, 5.0)

        # Combine impedances
        impedance_input = torch.stack([re_growth, rct_growth], dim=1)

        # Learn impedance influence
        impedance_factor = self.impedance_net(impedance_input)

        # Scale to [1.0, 2.0]
        impedance_factor = 1.0 + impedance_factor

        return impedance_factor

    def forward(self, lstm_features, time, current_temp,
                current_re, current_rct, initial_re, initial_rct):
        """
        Forward pass with input validation and numerical stability.
        """
        # ========================================
        # Input Validation
        # ========================================
        if torch.isnan(lstm_features).any():
            raise ValueError("NaN detected in LSTM features")
        if torch.isnan(current_temp).any():
            raise ValueError("NaN detected in temperature")
        if torch.isnan(current_re).any():
            raise ValueError("NaN detected in Re")
        if torch.isnan(current_rct).any():
            raise ValueError("NaN detected in Rct")

        # Check for invalid initial values
        if not (initial_re > 0).all():
            print(f"Warning: Zero/negative initial Re detected. Min: {initial_re.min()}")
            initial_re = torch.clamp(initial_re, min=1e-6)
        if not (initial_rct > 0).all():
            print(f"Warning: Zero/negative initial Rct detected. Min: {initial_rct.min()}")
            initial_rct = torch.clamp(initial_rct, min=1e-6)

        # ========================================
        # Feature Normalization
        # ========================================
        # Normalize time
        time_norm = time.unsqueeze(1) if time.dim() == 1 else time
        time_norm = time_norm / 1000.0

        # Normalize temperature (Celsius: typical range -10 to 50)
        temp_norm = current_temp / 50.0
        if temp_norm.dim() == 1:
            temp_norm = temp_norm.unsqueeze(1)

        # Normalize impedance growth
        re_norm = current_re / (initial_re + 1e-6)
        rct_norm = current_rct / (initial_rct + 1e-6)
        if re_norm.dim() == 1:
            re_norm = re_norm.unsqueeze(1)
            rct_norm = rct_norm.unsqueeze(1)

        # Clamp normalized values
        re_norm = torch.clamp(re_norm, 0.1, 10.0)
        rct_norm = torch.clamp(rct_norm, 0.1, 10.0)

        # ========================================
        # Physics Network
        # ========================================
        # Concatenate all features
        physics_input = torch.cat([
            lstm_features, time_norm, temp_norm, re_norm, rct_norm
        ], dim=1)

        # Predict base parameters
        physics_params = self.physics_net(physics_input)

        # Extract parameters with conservative constraints
        # α_base: base degradation rate [0.0001, 0.003]
        alpha_base = torch.sigmoid(physics_params[:, 0:1]) * 0.002 + 0.0001

        # β: shape parameter [0.5, 1.5]
        beta = torch.sigmoid(physics_params[:, 1:2]) * 1.0 + 0.5

        # Ea: activation energy [20k, 60k] J/mol
        Ea = torch.sigmoid(physics_params[:, 2:3]) * 40000 + 20000

        # ========================================
        # Multi-Physics Scaling
        # ========================================
        # Temperature factor (Arrhenius)
        temp_factor = self.compute_temperature_factor(current_temp, Ea)

        # Impedance factor
        impedance_factor = self.compute_impedance_factor(
            current_re, current_rct, initial_re, initial_rct
        )

        # Combined effective degradation rate
        # α_effective = α_base × temp_factor × impedance_factor
        alpha_effective = alpha_base * temp_factor * impedance_factor

        # Safety check on output
        if torch.isnan(alpha_effective).any():
            raise ValueError("NaN in alpha_effective! Check intermediate calculations.")

        return {
            'alpha': alpha_effective,
            'alpha_base': alpha_base,
            'beta': beta,
            'Ea': Ea,
            'temp_factor': temp_factor,
            'impedance_factor': impedance_factor
        }

    def compute_physics_derivative(self, soh, alpha, beta, time):
        """
        Physics-based derivative: dSOH/dt = -α × β × (1 - α×t)^(β-1)

        Now alpha includes temperature and impedance effects.
        """
        # Normalize time
        time_norm = time.unsqueeze(1) if time.dim() == 1 else time
        time_norm = time_norm / 1000.0

        # Compute degradation term with clamping
        term = 1.0 - alpha * time_norm
        term = torch.clamp(term, min=0.01, max=1.0)

        # Compute derivative
        dSOH_dt = -alpha * beta * torch.pow(term, beta - 1)

        # Clamp to reasonable degradation rates
        dSOH_dt = torch.clamp(dSOH_dt, -0.1, 0.0)

        return dSOH_dt


print("✅ Fixed EnhancedDeepHTPM class defined")
print("\n🔧 Key fixes applied:")
print("   1. Safe Celsius → Kelvin conversion")
print("   2. Clamped exponentials (-10 to +10)")
print("   3. Protected division (epsilon = 1e-6)")
print("   4. Input validation with warnings")
print("   5. Conservative parameter ranges")
print("✓ Enhanced DeepHTPM defined")
print("   Physics: Temperature (Arrhenius) + Impedance correlation")
print()


# ========================================
# 8E.4: Enhanced LSTM-PINN Model
# ========================================

print("🔗 Defining Enhanced LSTM-PINN Model...")
print("=" * 70)

class EnhancedLSTMPINNModel(nn.Module):
    """
    Enhanced LSTM-PINN with multi-physics constraints.

    Architecture:
    - LSTM: Extracts temporal features from SOH
    - SOH Head: Predicts next SOH (data-driven)
    - Enhanced Physics Head: Learns α(T, Re, Rct), β, Ea
    """

    def __init__(self, lstm_hidden: int = 64, lstm_layers: int = 2,
                 physics_hidden: int = 128, physics_layers: int = 3):
        super(EnhancedLSTMPINNModel, self).__init__()

        # LSTM for SOH sequences
        self.lstm = CorrectedLSTMExtractor(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers
        )

        # SOH prediction head (data-driven)
        self.soh_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden, 1),
            nn.Sigmoid()
        )

        # Enhanced physics head
        self.physics_head = EnhancedDeepHTPM(
            lstm_hidden_size=lstm_hidden,
            physics_hidden_size=physics_hidden,
            num_physics_layers=physics_layers
        )

    def forward(self, soh_sequence, temp_sequence, re_sequence, rct_sequence, time):
        """
        Forward pass with enhanced features.

        Args:
            soh_sequence: (batch, seq_len, 1)
            temp_sequence: (batch, seq_len)
            re_sequence: (batch, seq_len)
            rct_sequence: (batch, seq_len)
            time: (batch,)
        """
        # Extract LSTM features from SOH
        lstm_features, _ = self.lstm(soh_sequence)

        # Predict SOH (data-driven head)
        soh_pred = self.soh_head(lstm_features)

        # Get current physics features (last in sequence)
        current_temp = temp_sequence[:, -1]
        current_re = re_sequence[:, -1]
        current_rct = rct_sequence[:, -1]

        # Get initial physics features (first in sequence)
        initial_re = re_sequence[:, 0]
        initial_rct = rct_sequence[:, 0]

        # Get current SOH
        soh_current = soh_sequence[:, -1, :]

        # Enhanced physics prediction
        physics_out = self.physics_head(
            lstm_features, time,
            current_temp, current_re, current_rct,
            initial_re, initial_rct
        )

        return {
            'soh_pred': soh_pred,
            'soh_current': soh_current,
            'lstm_features': lstm_features,
            **physics_out  # Includes alpha, beta, Ea, temp_factor, impedance_factor
        }

print("✓ Enhanced LSTM-PINN Model defined")
print("   Two-head architecture with multi-physics constraints")
print()


# ========================================
# 8E.5: Test with Sample Data
# ========================================

print("🧪 Testing enhanced model with sample data...")
print("=" * 70)

# Test only if CONFIG is available
try:
    # Use device from CONFIG
    device = CONFIG['device']
    # Create dummy data for testing
    batch_size = 4
    seq_len = 20

    dummy_soh = torch.randn(batch_size, seq_len, 1) * 0.1 + 0.85
    dummy_temp = torch.randn(batch_size, seq_len) * 5 + 25  # °C
    dummy_re = torch.randn(batch_size, seq_len) * 0.01 + 0.05  # Ohms
    dummy_rct = torch.randn(batch_size, seq_len) * 0.02 + 0.10  # Ohms
    dummy_time = torch.arange(batch_size, dtype=torch.float32)

    # Initialize model
    enhanced_model = EnhancedLSTMPINNModel(
        lstm_hidden=64,
        lstm_layers=2,
        physics_hidden=128,
        physics_layers=3
    ).to(device)

    # Test forward pass
    enhanced_model.eval()
    with torch.no_grad():
        dummy_soh = dummy_soh.to(device)
        dummy_temp = dummy_temp.to(device)
        dummy_re = dummy_re.to(device)
        dummy_rct = dummy_rct.to(device)
        dummy_time = dummy_time.to(device)

        outputs = enhanced_model(dummy_soh, dummy_temp, dummy_re, dummy_rct, dummy_time)

    print(f"✓ Forward pass successful")
    print(f"\n   Output keys: {list(outputs.keys())}")
    print(f"   SOH pred shape: {outputs['soh_pred'].shape}")
    print(f"   Alpha shape: {outputs['alpha'].shape}")
    print(f"   Beta shape: {outputs['beta'].shape}")
    print(f"   Ea shape: {outputs['Ea'].shape}")

    print(f"\n   Sample values:")
    print(f"      α_base: {outputs['alpha_base'][0].item():.6f}")
    print(f"      Temp factor: {outputs['temp_factor'][0].item():.4f}")
    print(f"      Impedance factor: {outputs['impedance_factor'][0].item():.4f}")
    print(f"      α_effective: {outputs['alpha'][0].item():.6f}")
    print(f"      β: {outputs['beta'][0].item():.4f}")
    print(f"      Ea: {outputs['Ea'][0].item():.0f} J/mol")

except NameError:
    print("⚠️  Skipping model test (CONFIG not found)")
    print("   Run 01_environment_setup.py first")

print()


# ========================================
# ENHANCED ARCHITECTURE COMPLETE
# ========================================

print("=" * 70)
print("✅ STEP 8 ENHANCED COMPLETE: Multi-Physics Architecture")
print("=" * 70)

print("\n🎯 Enhanced Physics Constraints:")
print("   1. ✅ Arrhenius temperature dependence")
print("   2. ✅ Impedance-based degradation correlation")
print("   3. ✅ Combined multi-physics model")
print("   4. ✅ Two-head architecture (SOH + Physics)")

print("\n📊 Physics Model:")
print("   α_effective = α_base × exp(Ea/R(1/T_ref - 1/T)) × f(Re, Rct)")
print("   dSOH/dt = -α_effective × β × (1-α×t)^(β-1)")

print("\n💡 Next: Create Step 9 Enhanced to train with multi-physics")
print("   This will require modified data loading")

print("\n" + "=" * 70)
