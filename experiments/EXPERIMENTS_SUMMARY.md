# Ablation Study: 12 Experiments

## Structure
**3 Training Strategies × 4 Loss Combinations = 12 Experiments**

### Training Strategies
| Group | Name | Init | Paired Data | Epochs |
|-------|------|------|-------------|--------|
| A | Fully Paired | Scratch | 100% | 1-600 |
| B | Two-Stage 10% | Pretrained | 10% | 401-600 |
| C | Two-Stage 100% | Pretrained | 100% | 401-600 |

### Loss Combinations
| # | OT_input | OT_output | Entropy |
|---|----------|-----------|---------|
| 1 | ✓ | - | - |
| 2 | ✓ | - | ✓ |
| 3 | - | ✓ | - |
| 4 | - | ✓ | ✓ |

## Experiments
| Exp | Strategy | Loss | Gradient | Name |
|-----|----------|------|----------|------|
| 1 | A | OT_input | ✓ | `ablation_exp1_fully_pair_OT_input` |
| 2 | A | OT_input+E | ✓ | `ablation_exp2_fully_pair_OT_input_E` |
| 3 | A | OT_output | - | `ablation_exp3_fully_pair_OT_output` |
| 4 | A | OT_output+E | - | `ablation_exp4_fully_pair_OT_output_E` |
| 5 | B | OT_input | ✓ | `ablation_exp5_twostage_10p_OT_input` |
| 6 | B | OT_input+E | ✓ | `ablation_exp6_twostage_10p_OT_input_E` |
| 7 | B | OT_output | - | `ablation_exp7_twostage_10p_OT_output` |
| 8 | B | OT_output+E | - | `ablation_exp8_twostage_10p_OT_output_E` |
| 9 | C | OT_input | ✓ | `ablation_exp9_twostage_100p_OT_input` |
| 10 | C | OT_input+E | ✓ | `ablation_exp10_twostage_100p_OT_input_E` |
| 11 | C | OT_output | - | `ablation_exp11_twostage_100p_OT_output` |
| 12 | C | OT_output+E | - | `ablation_exp12_twostage_100p_OT_output_E` |

## Innovation: OT_input with Gradient
**Problem**: Iterative forward diffusion in training causes gradient accumulation → OOM
**Solution**: Use closed-form sampling (I²SB-style) instead of loop

**Affected**: Exp 1, 2, 5, 6, 9, 10
**Loss**: `τ * ||X_t - real_B||²` where `X_t` has gradient to enable intermediate state supervision
