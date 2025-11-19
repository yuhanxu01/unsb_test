#!/bin/bash

# ========================================
# Launch all 10 ablation study experiments in parallel
# ========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================"
echo "Launching All 16 Ablation Study Experiments"
echo "========================================"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Array to store job IDs
declare -a job_ids

# Function to submit a job and store its ID
submit_job() {
    local exp_script=$1
    local exp_name=$2

    echo "Submitting: $exp_name"

    # Submit the job and capture the job ID
    job_output=$(sbatch "$SCRIPT_DIR/$exp_script" 2>&1)

    if [ $? -eq 0 ]; then
        # Extract job ID from output (format: "Submitted batch job 12345")
        job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')
        job_ids+=("$job_id")
        echo "  ✓ Job ID: $job_id"
    else
        echo "  ✗ Failed to submit: $job_output"
    fi
    echo ""
}

echo "=========================================="
echo "Group A: Fully Paired (from scratch, 100% data)"
echo "=========================================="
echo ""

submit_job "exp1_fully_pair_OT_output.sh" "Exp 1: OT Output"
submit_job "exp2_fully_pair_OT_output_E.sh" "Exp 2: OT Output + Entropy"
submit_job "exp3_fully_pair_Entropy.sh" "Exp 3: Entropy Only"
submit_job "exp4_fully_pair_Baseline.sh" "Exp 4: Baseline (SB only)"

echo "=========================================="
echo "Group B: Two-Stage 10% (pretrained, 10% data)"
echo "=========================================="
echo ""

submit_job "exp5_twostage_10p_OT_output.sh" "Exp 5: OT Output"
submit_job "exp6_twostage_10p_OT_output_E.sh" "Exp 6: OT Output + Entropy"
submit_job "exp7_twostage_10p_Entropy.sh" "Exp 7: Entropy Only"

echo "=========================================="
echo "Group C: Two-Stage 100% (pretrained, 100% data)"
echo "=========================================="
echo ""

submit_job "exp8_twostage_100p_OT_output.sh" "Exp 8: OT Output"
submit_job "exp9_twostage_100p_OT_output_E.sh" "Exp 9: OT Output + Entropy"
submit_job "exp10_twostage_100p_Entropy.sh" "Exp 10: Entropy Only"

echo "=========================================="
echo "Group D: L2 Intermediate Single-Step (fully paired)"
echo "=========================================="
echo ""

submit_job "exp11_fully_pair_L2_inter_single.sh" "Exp 11: L2 Inter Single"
submit_job "exp12_fully_pair_L2_inter_single_E.sh" "Exp 12: L2 Inter Single + Entropy"
submit_job "exp13_fully_pair_L2_inter_single_OT_output.sh" "Exp 13: L2 Inter Single + OT Output"
submit_job "exp14_fully_pair_L2_inter_single_OT_output_E.sh" "Exp 14: L2 Inter Single + OT Output + Entropy"

echo "=========================================="
echo "Group E: L2 Intermediate Multi-Step (fully paired)"
echo "=========================================="
echo ""

submit_job "exp15_fully_pair_L2_inter_multi.sh" "Exp 15: L2 Inter Multi"
submit_job "exp16_fully_pair_L2_inter_multi_E.sh" "Exp 16: L2 Inter Multi + Entropy"

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total jobs submitted: ${#job_ids[@]}"
echo "Job IDs: ${job_ids[*]}"
echo ""

if [ ${#job_ids[@]} -gt 0 ]; then
    echo "To check status of all jobs:"
    echo "  squeue -j $(IFS=,; echo "${job_ids[*]}")"
    echo ""
    echo "To cancel all jobs:"
    echo "  scancel $(IFS=' '; echo "${job_ids[*]}")"
    echo ""
fi

echo "=========================================="
echo "Launch complete!"
echo "=========================================="
