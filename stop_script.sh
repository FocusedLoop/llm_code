#!/bin/bash

# Check if argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./stop_script.sh <CHECKPOINT_LIMIT>"
    exit 1
fi
CHECKPOINT_LIMIT=$1

# Define directories
FILE_PATH="/home/joshua/llms/stop_training.py"
LOG_FILE="/home/joshua/llms/output.log"
TMUX_SESSION="model_train"

# Start new tmux session and run script
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Stopping existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
    sleep 1
fi

echo "Starting tmux session '$TMUX_SESSION'..."
tmux new-session -d -s "$TMUX_SESSION" "python \"$FILE_PATH\" $CHECKPOINT_LIMIT > \"$LOG_FILE\" 2>&1"

# Confirm process started and tail logs
echo "Tailing..."
sleep 2
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Error: tmux session '$TMUX_SESSION' failed to start!"
    exit 1
fi
tail -f "$LOG_FILE"