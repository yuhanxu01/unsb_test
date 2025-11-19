#!/bin/bash
# Rename experiments to new numbering

# Fully Paired (exp1-4)
mv exp3_fully_pair_OT_output.sh exp1_fully_pair_OT_output.sh
mv exp4_fully_pair_OT_output_E.sh exp2_fully_pair_OT_output_E.sh

# Two-Stage 10% (exp5-7)
mv exp7_twostage_10p_OT_output.sh exp5_twostage_10p_OT_output.sh
mv exp8_twostage_10p_OT_output_E.sh exp6_twostage_10p_OT_output_E.sh

# Two-Stage 100% (exp8-10)
mv exp11_twostage_100p_OT_output.sh exp8_twostage_100p_OT_output.sh
mv exp12_twostage_100p_OT_output_E.sh exp9_twostage_100p_OT_output_E.sh

echo "Renamed experiments successfully"
