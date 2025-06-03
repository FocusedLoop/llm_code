#!/bin/bash

# Check for an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <huggingface model path>"
    exit 1
fi

tmux new -d -s download_session "python download_model.py \"$1\" > output.log 2>&1"
echo "Script started in tmux session 'download_session'. Check output.log for details."

# COMMANDS
# List sessions:            tmux list-session
# Attach to session:        tmux attach -t download_session
# Kill the tmux session:    tmux kill-session -t download_session