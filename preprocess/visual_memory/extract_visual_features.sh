#!/bin/bash
###############################################################################
# Script to extract visual features with automatic split processing and merge
# Usage:
#   ./extract_visual_features.sh --person <person> --gpu <gpu_list> [--num_frames <frames>]
#
# Examples:
#   ./extract_visual_features.sh --person A1_JAKE --gpu 0,1,2,3 --num_frames 16
#   ./extract_visual_features.sh --person A1_JAKE --gpu 0,0,1,1,2,2 --num_frames 5
#   ./extract_visual_features.sh --person A1_JAKE --gpu 0
###############################################################################

set -euo pipefail

# Array to store background process IDs (declared early for trap)
declare -a pids=()

# Trap function to handle Ctrl+C (SIGINT)
cleanup() {
    echo ""
    echo "Caught Ctrl+C! Stopping all processes..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    echo "All processes stopped."
    exit 1
}

# Set up the trap for SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Default values
PERSON="A1_JAKE"
GPU_LIST=""
NUM_FRAMES=16

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --person)
            PERSON="$2"
            shift 2
            ;;
        --gpu)
            GPU_LIST="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --person <person> --gpu <gpu_list> [--num_frames <frames>]"
            exit 1
            ;;
    esac
done

# Convert GPU_LIST to array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_SPLITS=${#GPUS[@]}

NUM_SPLITS=${#GPUS[@]}

echo "=========================================="
echo "Visual Feature Extraction"
echo "=========================================="
echo "Person:      $PERSON"
echo "GPU List:    $GPU_LIST"
echo "Num Splits:  $NUM_SPLITS"
echo "Num Frames:  $NUM_FRAMES"
echo "=========================================="
echo ""

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."  # Go to project root

# Create log directory (align with 2_preprocess.sh pattern)
LOG_DIR=".log/preprocess/${PERSON}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Log directory: $LOG_DIR"
echo ""

# Check if we're doing split processing
if [ "$NUM_SPLITS" -eq 1 ]; then
    LOG_FILE="$LOG_DIR/visual_features_gpu${GPUS[0]}_$TIMESTAMP.log"
    echo "Running on single GPU ${GPUS[0]} (no splitting)..."
    echo "Log file: $LOG_FILE"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} python preprocess/visual_memory/extract_visual_features.py \
        --person "$PERSON" \
        --num_frames "$NUM_FRAMES" 2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Processing completed successfully!"
    
else
    echo "Running split processing on $NUM_SPLITS processes..."
    echo ""
    
    # Reset pids array for this run
    pids=()
    
    # Launch parallel jobs for each GPU in the list
    for ((i=0; i<$NUM_SPLITS; i++)); do
        GPU_ID=${GPUS[$i]}
        LOG_FILE="$LOG_DIR/visual_features_split${i}_gpu${GPU_ID}_$TIMESTAMP.log"
        
        echo "Starting split $i on GPU $GPU_ID..."
        echo "  Log file: $LOG_FILE"
        
        # Run in background with GPU assignment and redirect output to log file
        CUDA_VISIBLE_DEVICES=$GPU_ID python preprocess/visual_memory/extract_visual_features.py \
            --person "$PERSON" \
            --split_id "$i" \
            --num_splits "$NUM_SPLITS" \
            --num_frames "$NUM_FRAMES" \
            > "$LOG_FILE" 2>&1 &
        
        # Store the PID
        pids+=($!)
        
        # Small delay to avoid race conditions
        sleep 2
    done
    
    echo ""
    echo "All $NUM_SPLITS processes launched. Waiting for completion..."
    echo ""
    
    # Wait for all background processes and check their exit status
    failed=0
    for ((i=0; i<${#pids[@]}; i++)); do
        pid=${pids[$i]}
        
        if wait $pid; then
            echo "✓ Split $i (PID $pid) completed successfully"
        else
            echo "✗ Split $i (PID $pid) failed!"
            failed=1
        fi
    done
    
    echo ""
    
    # Check if any splits failed
    if [ $failed -eq 1 ]; then
        echo "Error: One or more splits failed. Aborting merge."
        exit 1
    fi
    
    echo "All splits completed successfully!"
    echo ""
    echo "Merging split files..."
    
    LOG_FILE="$LOG_DIR/visual_features_merge_$TIMESTAMP.log"
    echo "Log file: $LOG_FILE"
    
    # Merge the split files
    python preprocess/visual_memory/extract_visual_features.py \
        --person "$PERSON" \
        --num_splits "$NUM_SPLITS" \
        --merge 2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "All processing completed successfully!"
fi

echo ""
echo "Output file: output/metadata/visual_memory/$PERSON/visual_embeddings.pkl"
echo ""
