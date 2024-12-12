#!/bin/bash

# Check if hostname matches the pattern 'gpu-q-<number>'
if [[ $(hostname) =~ ^gpu-[a-z]-[0-9]+$ ]]; then
    echo "Hostname matches the pattern: $HOSTNAME"
    # Get VENV_PATH
    source ../../../.env

    # Check if the virtual environment exists
    if [[ -d "$VENV_PATH" ]]; then
        echo "Activating virtual environment at $VENV_PATH."
        source "$VENV_PATH/bin/activate"
    else
        echo "Virtual environment not found at $VENV_PATH."
        exit 1
    fi
else
    echo "Hostname does not match the pattern. Skipping virtual environment activation."
fi
