#!/bin/bash
# Wrapper script to run full training pipeline with proper logging

cd /home/ely/Projects/2026-winter/lab/final-proj/playground

echo "=========================================="
echo "Starting Full Model Training Pipeline"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Run training with unbuffered output
conda run -n spark_env python -u data_transformation.py

echo ""
echo "=========================================="
echo "Training Complete"
echo "End time: $(date)"
echo "=========================================="
