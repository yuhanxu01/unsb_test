# Ablation Study: 10 Experiments

## Structure
**3 Training Strategies × 3-4 Loss Combinations = 10 Experiments**

### Training Strategies
| Group | Name | Init | Paired Data | Epochs |
|-------|------|------|-------------|--------|
| A | Fully Paired | Scratch | 100% | 1-600 |
| B | Two-Stage 10% | Pretrained | 10% | 401-600 |
| C | Two-Stage 100% | Pretrained | 100% | 401-600 |

### Loss Combinations
| # | OT_output | Entropy | Baseline |
|---|-----------|---------|----------|
| 1 | ✓ | - | - |
| 2 | ✓ | ✓ | - |
| 3 | - | ✓ | - |
| 4 | - | - | ✓ (SB only) |

## Experiments
| Exp | Strategy | Loss | Name |
|-----|----------|------|------|
| 1 | A | OT_output | `ablation_exp1_fully_pair_OT_output` |
| 2 | A | OT_output+E | `ablation_exp2_fully_pair_OT_output_E` |
| 3 | A | Entropy | `ablation_exp3_fully_pair_Entropy` |
| 4 | A | Baseline | `ablation_exp4_fully_pair_Baseline` |
| 5 | B | OT_output | `ablation_exp5_twostage_10p_OT_output` |
| 6 | B | OT_output+E | `ablation_exp6_twostage_10p_OT_output_E` |
| 7 | B | Entropy | `ablation_exp7_twostage_10p_Entropy` |
| 8 | C | OT_output | `ablation_exp8_twostage_100p_OT_output` |
| 9 | C | OT_output+E | `ablation_exp9_twostage_100p_OT_output_E` |
| 10 | C | Entropy | `ablation_exp10_twostage_100p_Entropy` |

## Loss Details

### OT_output
`loss_OT_output = τ * ||fake_B - real_B||²`
- Supervises network's final output
- Guides generation toward ground truth

### Entropy
`loss_entropy = -(T-t)/T * τ * ET_XY`
- Energy-based regularization
- Computed via netE (energy network)

### Baseline
- Standard SB loss only
- No additional supervision
- Reference for ablation comparison

## Changes from Original Design

### Removed: OT_input
**Reason**: Training with gradient-enabled forward diffusion caused:
- OOM (13GB+ GPU memory)
- Requires iterative netG calls during training (non-standard)
- Gradient backprop dependency issues

**Original concept**: Supervise intermediate diffusion state
`loss_OT_input = τ * ||real_A_noisy - real_B||²`

This required real_A_noisy to have gradients through iterative network calls, which conflicted with standard diffusion model training (direct sampling).
