#!/bin/bash

# Check for an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <file.ipynb | file.py>"
    exit 1
fi

# Define directories
BASE_DIR="/home/joshua/llms/llama3.1_PC_Build"
INPUT_NAME="$1"
FILE_PATH="$BASE_DIR/$INPUT_NAME"
LOG_FILE="$BASE_DIR/output.log"
LOG_CACHE="$BASE_DIR/TRAIN_CACHE.json"
TMUX_SESSION="model_train"

# Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File '$INPUT_NAME' not found in '$BASE_DIR'!"
    exit 1
fi

# Convert notebook to Python script or just run python script
if [[ "$INPUT_NAME" == *.ipynb ]]; then
    SCRIPT_NAME="${INPUT_NAME%.ipynb}.py"
    SCRIPT_PATH="$BASE_DIR/$SCRIPT_NAME"
    echo "Converting $INPUT_NAME to Python script..."
    if ! jupyter nbconvert --to script "$FILE_PATH" > /dev/null 2>&1; then
        echo "Error: Failed to convert '$INPUT_NAME'!"
        exit 1
    fi
    FILE_PATH="$SCRIPT_PATH"
elif [[ "$INPUT_NAME" == *.py ]]; then
    echo "Running Python script '$INPUT_NAME'..."
else
    echo "Error: Unsupported file type. Please provide a .ipynb or .py file."
    exit 1
fi

# Start new tmux session and run script
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Stopping existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
    sleep 1
fi
echo "Starting tmux session '$TMUX_SESSION'..."
tmux new-session -d -s "$TMUX_SESSION" "python \"$FILE_PATH\" > \"$LOG_FILE\" 2>&1"

# Start caching training time and epoch in the background: follow the log look for resume statement and track epoch value
# total_training_time is 2 hours ahead
# Calculate average of length of completing an epoch
# Calculate per epoch iteration completion average
# last digit of epoch is only cached in the list, needs to th the whole float

# Initialize the cache file if it doesn't exist
if [ ! -f "$LOG_CACHE" ]; then
    echo '{"start_time": null, "last_epoch": null, "total_training_time": 0, "epoch_list": {}}' > "$LOG_CACHE"
fi


total_training_time=0
start_time=""
last_epoch=""

# Monitor the log and cache the epochs and times
monitor_training_log() {
    start_time=$(date +%s)
    tail -n 0 -f "$LOG_FILE" | while read -r line; do
        if [[ "$line" == *"No valid checkpoint found. Training from scratch."* || "$line" == *"Resuming from latest checkpoint"* ]]; then
            start_time=$(date +%s)
        fi
        
        if [[ "$line" =~ epoch[^0-9]*([0-9]+(\.[0-9]+)?) ]]; then
            current_epoch="${BASH_REMATCH[1]}"
            current_time=$(date +%s)

            if [ -n "$start_time" ]; then
                elapsed_time=$((current_time - start_time))
                total_training_time=$((total_training_time + elapsed_time))
            fi
            start_time=$current_time

            jq --arg start_time "$start_time" \
               --arg total_training_time "$total_training_time" \
               --argjson last_epoch "$current_epoch" \
               --arg current_time "$current_time" \
               '
               .start_time = ($start_time | tonumber) |
               .total_training_time = ($total_training_time | tonumber) |
               .last_epoch = $last_epoch |
               .epoch_list[$current_time] = $last_epoch
               ' "$LOG_CACHE" > "${LOG_CACHE}.tmp" && mv "${LOG_CACHE}.tmp" "$LOG_CACHE"
        fi
    done
}
monitor_training_log &

# Confirm process started and tail
echo "Tailing..."
sleep 2
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Error: tmux session '$TMUX_SESSION' failed to start!"
    exit 1
fi
tail -f "$LOG_FILE"