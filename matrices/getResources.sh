#!/bin/bash

set -e

RESOURCE_FILE="${1:-matrices/mat_resources.txt}"
RAW_DIR="${RAW_DIR:-matrices/raw}"

mkdir -p "$RAW_DIR"

if [ ! -f "$RESOURCE_FILE" ]; then
    echo "Error: resource file not found: $RESOURCE_FILE"
    exit 1
fi

while IFS= read -r link; do
    if [ -z "$link" ]; then
        continue
    fi

    case "$link" in
        \#*)
            continue
            ;;
    esac

    filename=$(basename "$link")
    matrix_name="${filename%.tar.gz}"

    echo "Step 1: Processing matrix $matrix_name"

    if [ -f "$RAW_DIR/$filename" ]; then
        echo "Step 2: Archive already exists: $RAW_DIR/$filename"
    else
        echo "Step 2: Downloading $filename"
        wget -P "$RAW_DIR" "$link"
    fi

    if [ -d "$RAW_DIR/$matrix_name" ]; then
        echo "Step 3: Directory already extracted: $RAW_DIR/$matrix_name"
    else
        echo "Step 3: Extracting $filename"
        tar -xzf "$RAW_DIR/$filename" -C "$RAW_DIR"
    fi

    echo "Step 4: Done with $matrix_name"
    echo ""

done < "$RESOURCE_FILE"