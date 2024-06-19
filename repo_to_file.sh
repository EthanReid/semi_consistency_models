#!/bin/bash

# Define the directory of the git repo
REPO_DIR="/Users/ethanreid/Documents/DevProjects/Research/ConsistencyModels/consistency_models"

# Define the output file
OUTPUT_FILE="/Users/ethanreid/Documents/DevProjects/Research/ConsistencyModels/repo_contents.txt"

# Check if the output file already exists and remove it to avoid appending to old data
if [ -f "$OUTPUT_FILE" ]; then
    rm "$OUTPUT_FILE"
fi

# Traverse through the repo directory and dump the content of each file into the output file
find "$REPO_DIR" -type f -not -path '*/\.*' | while read -r file; do
    echo "======================" >> "$OUTPUT_FILE"
    echo "FILE: $file" >> "$OUTPUT_FILE"
    echo "======================" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "\n\n" >> "$OUTPUT_FILE"
done

echo "All file contents have been dumped into $OUTPUT_FILE"