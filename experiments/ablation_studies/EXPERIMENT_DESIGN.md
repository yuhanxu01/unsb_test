# Ablation Study: Detailed Experiment Design

## Scientific Motivation

This ablation study aims to answer two key research questions:

1. **Component Analysis**: What are the individual contributions of different loss components in Schrödinger Bridge (SB) for paired MRI contrast transfer?
2. **Supervision Strategy**: Where should we apply supervision - at the intermediate diffusion state or at the final output?

---

## Loss Component Definitions

### OT_input: Intermediate State Supervision
```python
loss_OT_input = tau * mean((real_A_noisy - real_B)^2)
```

**What it does**:
- Supervises the **intermediate noisy state** from forward diffusion
- `real_A_noisy` is computed through multiple diffusion steps: A → X₁ → X₂ → ... → Xₜ
- When `use_ot_input=True`, this diffusion is computed **with gradient**
- Directly constrains the forward diffusion process to move toward ground truth

**Implementation**:
- Forward diffusion computed in `forward()` with `compute_noisy_with_grad=True`
- Uses gradient checkpointing: only keeps gradient for final Xₜ, detaches intermediate states
- Memory-efficient: `Xt = (1-inter) * Xt.detach() + inter * Xt_1 + noise`

### OT_output: Final Output Supervision
```python
loss_OT_output = tau * mean((fake_B - real_B)^2)
```

**What it does**:
- Supervises the **final network output**
- `fake_B` is the result after full diffusion process
- Standard supervised learning: minimizes L2 distance to ground truth
- Does NOT require gradient in forward diffusion (uses `no_grad` version)

### Entropy Loss: Energy-Based Regularization
```python
loss_entropy = -tau * ET_XY
```

**What it does**:
- Energy-based regularization from Schrödinger Bridge formulation
- `ET_XY = E(X,X|X,X) - logsumexp(E(X,X|X,X'))`
- Encourages smooth transport between domains
- Based on netE (energy network)

### L2_intermediate_single: Single-Step Intermediate Supervision
```python
loss_L2_inter_single = tau * mean((netG(X_{i-1}) - real_B)^2)
```

**What it does**:
- Supervises network output at a **single intermediate diffusion step**
- Default step: `i = T//2` (middle of diffusion process)
- `X_{i-1}` is an intermediate noisy state during forward diffusion
- `netG(X_{i-1})` is the network's prediction from that intermediate state
- Directly constrains intermediate predictions to match ground truth
- **Gradient-enabled**: Computed with gradient to enable backpropagation

**Implementation**:
- Forward diffusion computed with selective gradient: gradient enabled only at target step
- Uses `--use_l2_intermediate_single` flag
- Step can be customized with `--intermediate_step` parameter

### L2_intermediate_multi: Multi-Step Intermediate Supervision
```python
loss_L2_inter_multi = tau * mean([mean((netG(X_i) - real_B)^2) for all i in 1..T])
```

**What it does**:
- Supervises network outputs at **all intermediate diffusion steps**
- Averages L2 loss across all steps from 1 to current timestep T
- Similar to "deep supervision" in neural networks
- Provides stronger supervision signal throughout the diffusion process
- **Gradient-enabled**: All intermediate steps computed with gradient

**Implementation**:
- Forward diffusion computed with gradient for all steps
- Uses `--use_l2_intermediate_multi` flag
- May be more memory-intensive due to multiple gradient computations

---

## Experiment Breakdown (16 Total)

### Group 1: Fully Paired (100% data, from scratch)

| Exp | OT_input | OT_output | Entropy | Training | Research Question |
|-----|----------|-----------|---------|----------|-------------------|
| 1   | ✓        |           |         | 1-600    | Can intermediate state supervision alone work? |
| 2   | ✓        |           | ✓       | 1-600    | Does entropy help intermediate supervision? |
| 3   |          | ✓         |         | 1-600    | Can output supervision alone work? |
| 4   |          | ✓         | ✓       | 1-600    | Does entropy help output supervision? |

**Comparison**:
- **Exp1 vs Exp3**: Intermediate vs output supervision (both without entropy)
- **Exp2 vs Exp4**: Intermediate vs output supervision (both with entropy)
- **Exp1 vs Exp2**: Effect of adding entropy to intermediate supervision
- **Exp3 vs Exp4**: Effect of adding entropy to output supervision

### Group 2: Two-Stage (10% data, pretrained)

| Exp | OT_input | OT_output | Entropy | Training | Research Question |
|-----|----------|-----------|---------|----------|-------------------|
| 5   | ✓        |           |         | 401-600  | Low-data: intermediate supervision? |
| 6   | ✓        |           | ✓       | 401-600  | Low-data: intermediate + entropy? |
| 7   |          | ✓         |         | 401-600  | Low-data: output supervision? |
| 8   |          | ✓         | ✓       | 401-600  | Low-data: output + entropy? |

**Comparison**:
- **Exp5 vs Exp7**: Intermediate vs output (10% data, no entropy)
- **Exp6 vs Exp8**: Intermediate vs output (10% data, with entropy)
- **Exp5 vs Exp1**: Effect of data scarcity on intermediate supervision
- **Exp7 vs Exp3**: Effect of data scarcity on output supervision

### Group 3: Two-Stage (100% data, pretrained)

| Exp | OT_input | OT_output | Entropy | Training | Research Question |
|-----|----------|-----------|---------|----------|-------------------|
| 9   | ✓        |           |         | 401-600  | Pretrained: intermediate supervision? |
| 10  | ✓        |           | ✓       | 401-600  | Pretrained: intermediate + entropy? |
| 11  |          | ✓         |         | 401-600  | Pretrained: output supervision? |
| 12  |          | ✓         | ✓       | 401-600  | Pretrained: output + entropy? |

**Comparison**:
- **Exp9 vs Exp11**: Intermediate vs output (pretrained, no entropy)
- **Exp10 vs Exp12**: Intermediate vs output (pretrained, with entropy)
- **Exp1 vs Exp9**: From-scratch vs pretrained (intermediate, no entropy)
- **Exp3 vs Exp11**: From-scratch vs pretrained (output, no entropy)

---

### Group 4: L2 Intermediate Single-Step (100% data, from scratch)

| Exp | L2_inter_single | OT_output | Entropy | Training | Research Question |
|-----|-----------------|-----------|---------|----------|-------------------|
| 11  | ✓               |           |         | 1-600    | Can single-step intermediate supervision work? |
| 12  | ✓               |           | ✓       | 1-600    | Does entropy help L2 intermediate supervision? |
| 13  | ✓               | ✓         |         | 1-600    | Combined intermediate + final supervision? |
| 14  | ✓               | ✓         | ✓       | 1-600    | All three components together? |

**Comparison**:
- **Exp11 vs Exp1**: L2 intermediate vs OT output (which supervision strategy is better?)
- **Exp11 vs Exp12**: Effect of adding entropy to L2 intermediate
- **Exp13 vs Exp1**: Both intermediate & final vs final only
- **Exp13 vs Exp11**: Adding final supervision to intermediate
- **Exp14**: Full combination - does everything together work best?

### Group 5: L2 Intermediate Multi-Step (100% data, from scratch)

| Exp | L2_inter_multi | Entropy | Training | Research Question |
|-----|----------------|---------|----------|-------------------|
| 15  | ✓              |         | 1-600    | Multi-step intermediate supervision alone? |
| 16  | ✓              | ✓       | 1-600    | Multi-step + entropy regularization? |

**Comparison**:
- **Exp15 vs Exp11**: Multi-step vs single-step intermediate supervision
- **Exp15 vs Exp1**: Multi-step intermediate vs final output only
- **Exp16 vs Exp15**: Effect of entropy on multi-step supervision
- **Exp16 vs Exp12**: Multi-step vs single-step (both with entropy)

---

## Expected Insights

### 1. Supervision Location (OT_input vs OT_output)
- **Hypothesis**: OT_output should work better as it directly supervises the final task
- **OT_input** may help learn better diffusion dynamics but might be harder to optimize

### 2. Role of Entropy
- **Hypothesis**: Entropy regularization provides smoother transport
- May be more important when supervision is weaker (OT_input)

### 3. Data Efficiency
- **Group 2 (10% data)**: Which supervision strategy is more data-efficient?
- **Group 3 (100% data)**: Can pretrained model benefit from paired fine-tuning?

### 4. Training Dynamics
- **Fully Paired**: Learn everything from paired data
- **Two-Stage**: Build on unpaired pretrained model, adapt with paired data

### 5. L2 Intermediate Supervision (New)
- **Hypothesis**: Supervising intermediate network predictions may provide stronger gradient signals
- **Single-step (T//2)**: Provides supervision at middle of diffusion process
  - Potentially more efficient than multi-step
  - May help network learn better trajectory
- **Multi-step (all steps)**: Deep supervision across entire diffusion
  - Stronger supervision signal
  - Higher memory cost
  - May lead to better or more stable training

**Key Questions**:
1. **Exp11 vs Exp1**: Does intermediate supervision outperform final output supervision?
2. **Exp15 vs Exp11**: Is multi-step better than single-step?
3. **Exp13 vs Exp1 vs Exp11**: Should we use both intermediate and final, or just one?
4. **Memory/Speed Trade-off**: How much slower is multi-step vs single-step?

**Expected Outcomes**:
- L2 intermediate may provide clearer gradient signals than OT_output
- Multi-step may be more stable but slower
- Combining intermediate + final (Exp13) might be the best approach
- Entropy regularization may still help (Exp12, Exp14, Exp16)

---

## Implementation Notes

### Memory Efficiency for OT_input

When `use_ot_input=True`, forward diffusion is computed with gradient:

```python
# Gradient checkpointing: only keep gradient for current step
Xt = (1-inter) * Xt.detach() + inter * Xt_1 + noise
#                    ^^^^^^^ Detach previous state to save memory
#                                    ^^^^ Keep gradient for network output
```

This allows gradient to flow through `real_A_noisy` without storing all intermediate states.

### Conditional Gradient Computation

```python
compute_noisy_with_grad = use_ot_input and self.opt.isTrain
```

- Only enabled for OT_input experiments
- Other experiments use faster `no_grad` version
- Automatic switching based on experiment config

### L2 Intermediate Implementation

**Single-Step Mode** (`--use_l2_intermediate_single`):
```python
# In forward(), only compute gradient at target step
if t == intermediate_step:  # Default: T//2
    Xt_1 = self.netG(Xt, time_idx_t, z)  # Keep gradient
    self.intermediate_preds.append(Xt_1)
else:
    with torch.no_grad():
        Xt_1 = self.netG(Xt, time_idx_t, z)
```

**Multi-Step Mode** (`--use_l2_intermediate_multi`):
```python
# In forward(), compute gradient for all steps
for t in range(1, T+1):
    Xt_1 = self.netG(Xt, time_idx_t, z)  # Keep gradient for all steps
    self.intermediate_preds.append(Xt_1)
```

**Loss Computation**:
```python
# Single-step
loss_L2_inter_single = tau * mean((intermediate_preds[0] - real_B)²)

# Multi-step
loss_L2_inter_multi = tau * mean([mean((pred - real_B)²) for pred in intermediate_preds])
```

**Key Differences from OT_input**:
- **OT_input**: Supervises the intermediate noisy state `X_{i-1}` directly
- **L2_intermediate**: Supervises the network prediction `netG(X_{i-1})` from intermediate state
- L2_intermediate has gradients through the network, providing clearer optimization signals

---

## Metrics to Compare

All experiments log:
- **SSIM**: Structural similarity
- **PSNR**: Peak signal-to-noise ratio
- **NRMSE**: Normalized root mean square error
- **Loss components**: OT_output, Entropy, L2_inter_single, L2_inter_multi (when applicable)

Compare across:
- Training efficiency (loss curves)
- Final performance (SSIM/PSNR)
- Generalization (validation metrics)
