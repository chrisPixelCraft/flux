#!/bin/bash
"""
Convenience script for running BezierAdapter training with common configurations.

This script provides easy access to different training configurations and
handles environment setup automatically.
"""

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  setup         Setup environment and download data"
    echo "  dev          Run development training (fast, small dataset)"
    echo "  full         Run full training (production settings)"
    echo "  distributed  Run distributed training (multi-GPU)"
    echo "  resume       Resume training from checkpoint"
    echo "  test         Run model testing and validation"
    echo ""
    echo "Options:"
    echo "  --data-root PATH    Path to dataset directory (default: ./data)"
    echo "  --output-dir PATH   Path to output directory (default: ./outputs)"
    echo "  --num-gpus N        Number of GPUs for distributed training (default: 4)"
    echo "  --batch-size N      Override batch size"
    echo "  --learning-rate LR  Override learning rate"
    echo "  --resume-from PATH  Path to checkpoint for resuming"
    echo "  --flux-model NAME   FLUX model variant (default: flux-dev)"
    echo "  --compile           Enable torch.compile optimization"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                              # Setup environment"
    echo "  $0 dev --batch-size 2                # Development training"
    echo "  $0 full --output-dir ./exp1          # Full training"
    echo "  $0 distributed --num-gpus 8          # 8-GPU distributed training"
    echo "  $0 resume --resume-from ./checkpoint.pt"
}

# Setup environment and data
setup_environment() {
    print_color $BLUE "Setting up BezierAdapter training environment..."
    
    # Check if conda environment exists
    if ! conda env list | grep -q "easycontrol"; then
        print_color $RED "Error: conda environment 'easycontrol' not found"
        print_color $YELLOW "Please create the environment first:"
        print_color $YELLOW "  conda create -n easycontrol python=3.10"
        print_color $YELLOW "  conda activate easycontrol"
        exit 1
    fi
    
    # Activate conda environment
    print_color $GREEN "Activating conda environment 'easycontrol'..."
    eval "$(conda shell.bash hook)"
    conda activate easycontrol
    
    # Install dependencies
    print_color $GREEN "Installing dependencies..."
    pip install -r requirements.txt
    
    # Check if data exists
    if [ ! -d "${DATA_ROOT}" ]; then
        print_color $BLUE "Creating data directory: ${DATA_ROOT}"
        mkdir -p "${DATA_ROOT}"
    fi
    
    # Run preprocessing if needed
    if [ ! -f "${DATA_ROOT}/train_metadata.json" ]; then
        print_color $BLUE "Running data preprocessing..."
        
        # Download dataset
        if [ -f "./get_dataset.sh" ]; then
            print_color $GREEN "Downloading font dataset..."
            bash ./get_dataset.sh
        else
            print_color $YELLOW "Warning: get_dataset.sh not found, skipping dataset download"
        fi
        
        # Extract Bézier curves
        if [ -f "./bezier_extraction.py" ]; then
            print_color $GREEN "Extracting Bézier curves from fonts..."
            python bezier_extraction.py --output-dir "${DATA_ROOT}"
        else
            print_color $YELLOW "Warning: bezier_extraction.py not found, skipping curve extraction"
        fi
    else
        print_color $GREEN "Preprocessed data found, skipping preprocessing"
    fi
    
    print_color $GREEN "Environment setup completed!"
}

# Check system requirements
check_requirements() {
    print_color $BLUE "Checking system requirements..."
    
    # Check Python version
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    print_color $GREEN "Python version: $python_version"
    
    # Check PyTorch installation
    if ! python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
        print_color $RED "Error: PyTorch not installed"
        exit 1
    fi
    
    # Check CUDA availability
    if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null | grep -q "True"; then
        gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_color $GREEN "CUDA available: Yes"
        print_color $GREEN "GPU count: $gpu_count"
        print_color $GREEN "GPU name: $gpu_name"
        
        # Check GPU memory
        gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        print_color $GREEN "GPU memory: ${gpu_memory}MB"
        
        if [ "$gpu_memory" -lt 12000 ]; then
            print_color $YELLOW "Warning: GPU memory < 12GB, consider reducing batch size"
        fi
    else
        print_color $YELLOW "CUDA not available, training will use CPU (very slow)"
    fi
    
    print_color $GREEN "System requirements check completed!"
}

# Run development training
run_dev_training() {
    print_color $BLUE "Starting development training..."
    
    python train_bezier_adapter.py \
        --config-type dev \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUTPUT_DIR}/dev_training" \
        --flux-model "${FLUX_MODEL}" \
        --log-level INFO \
        ${BATCH_SIZE_ARG} \
        ${LEARNING_RATE_ARG} \
        ${COMPILE_ARG} \
        "$@"
}

# Run full training
run_full_training() {
    print_color $BLUE "Starting full training..."
    
    python train_bezier_adapter.py \
        --config-type full \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUTPUT_DIR}/full_training" \
        --flux-model "${FLUX_MODEL}" \
        --mixed-precision \
        --log-level INFO \
        ${BATCH_SIZE_ARG} \
        ${LEARNING_RATE_ARG} \
        ${COMPILE_ARG} \
        "$@"
}

# Run distributed training
run_distributed_training() {
    print_color $BLUE "Starting distributed training on ${NUM_GPUS} GPUs..."
    
    # Check if multiple GPUs are available
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [ "$gpu_count" -lt "$NUM_GPUS" ]; then
        print_color $RED "Error: Requested $NUM_GPUS GPUs but only $gpu_count available"
        exit 1
    fi
    
    # Use torchrun for distributed training
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=12355 \
        train_bezier_adapter.py \
        --config-type distributed \
        --num-gpus $NUM_GPUS \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUTPUT_DIR}/distributed_training" \
        --flux-model "${FLUX_MODEL}" \
        --mixed-precision \
        --log-level INFO \
        ${BATCH_SIZE_ARG} \
        ${LEARNING_RATE_ARG} \
        ${COMPILE_ARG} \
        "$@"
}

# Resume training from checkpoint
resume_training() {
    if [ -z "$RESUME_FROM" ]; then
        print_color $RED "Error: --resume-from PATH is required for resume command"
        exit 1
    fi
    
    if [ ! -f "$RESUME_FROM" ]; then
        print_color $RED "Error: Checkpoint file not found: $RESUME_FROM"
        exit 1
    fi
    
    print_color $BLUE "Resuming training from: $RESUME_FROM"
    
    python train_bezier_adapter.py \
        --config-type full \
        --resume-from "$RESUME_FROM" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUTPUT_DIR}/resumed_training" \
        --flux-model "${FLUX_MODEL}" \
        --mixed-precision \
        --log-level INFO \
        ${BATCH_SIZE_ARG} \
        ${LEARNING_RATE_ARG} \
        ${COMPILE_ARG} \
        "$@"
}

# Run model testing
run_testing() {
    print_color $BLUE "Running model testing and validation..."
    
    # Find latest checkpoint
    latest_checkpoint=""
    if [ -d "${OUTPUT_DIR}" ]; then
        latest_checkpoint=$(find "${OUTPUT_DIR}" -name "best_checkpoint.pt" -o -name "checkpoint_epoch_*.pt" | sort | tail -1)
    fi
    
    if [ -n "$latest_checkpoint" ]; then
        print_color $GREEN "Found checkpoint: $latest_checkpoint"
        RESUME_FROM="$latest_checkpoint"
    else
        print_color $YELLOW "No checkpoint found, testing with random initialization"
    fi
    
    python -c "
import sys
sys.path.insert(0, 'src')
from flux.training.config import get_development_config
from flux.training.dataset import split_dataset
from flux.modules.bezier_flux_model import FluxBezierAdapter
from flux.util import load_flow_model, configs
import torch

print('Running BezierAdapter validation tests...')

# Load configuration
config = get_development_config()
config.data.data_root = '${DATA_ROOT}'

# Test dataset loading
try:
    train_dataset, val_dataset, test_dataset = split_dataset(config.data.data_root)
    print(f'✓ Dataset loading successful: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test')
except Exception as e:
    print(f'✗ Dataset loading failed: {e}')
    exit(1)

# Test model initialization
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    flux_params = configs['${FLUX_MODEL}'].params
    bezier_config = config.get_model_config_dict()
    model = FluxBezierAdapter(flux_params, bezier_config=bezier_config)
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f'✓ Model initialization successful')
    print(f'  Trainable parameters: {trainable_params:,}')
    print(f'  Total parameters: {total_params:,}')
    print(f'  Parameter efficiency: {trainable_params/total_params*100:.1f}%')
except Exception as e:
    print(f'✗ Model initialization failed: {e}')
    exit(1)

print('All validation tests passed!')
"
}

# Default values
DATA_ROOT="./data"
OUTPUT_DIR="./outputs"
NUM_GPUS=4
FLUX_MODEL="flux-dev"
BATCH_SIZE=""
LEARNING_RATE=""
RESUME_FROM=""
COMPILE_ARG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            BATCH_SIZE_ARG="--batch-size $BATCH_SIZE"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            LEARNING_RATE_ARG="--learning-rate $LEARNING_RATE"
            shift 2
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --flux-model)
            FLUX_MODEL="$2"
            shift 2
            ;;
        --compile)
            COMPILE_ARG="--compile"
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        setup|dev|full|distributed|resume|test)
            COMMAND="$1"
            shift
            break
            ;;
        *)
            print_color $RED "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if command was provided
if [ -z "$COMMAND" ]; then
    print_color $RED "Error: No command provided"
    print_usage
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Execute command
case $COMMAND in
    setup)
        setup_environment
        check_requirements
        ;;
    dev)
        check_requirements
        run_dev_training "$@"
        ;;
    full)
        check_requirements
        run_full_training "$@"
        ;;
    distributed)
        check_requirements
        run_distributed_training "$@"
        ;;
    resume)
        check_requirements
        resume_training "$@"
        ;;
    test)
        check_requirements
        run_testing "$@"
        ;;
    *)
        print_color $RED "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac