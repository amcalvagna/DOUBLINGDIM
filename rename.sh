#!/bin/bash
# Script to remove a given substring from all filenames in a directory

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <directory> <substring> [--execute]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/dir 'unwanted_text'           # Dry run (preview only)"
    echo "  $0 /path/to/dir 'unwanted_text' --execute # Actually rename files"
    exit 1
fi

DIRECTORY="$1"
SUBSTRING="$2"
REPLACE="$3"
EXECUTE=false

# Check for --execute flag
if [ "$4" = "--execute" ]; then
    EXECUTE=true
fi

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist"
    exit 1
fi

# Count files
count=0
renamed=0

echo "Directory: $DIRECTORY"
echo "Substring to remove: '$SUBSTRING'"
if [ "$EXECUTE" = true ]; then
    echo "Mode: LIVE (files will be renamed)"
else
    echo "Mode: DRY RUN (no changes will be made)"
fi
echo "------------------------------------------------------------"

# Loop through all files in directory
for filepath in "$DIRECTORY"/*; do
    # Skip if not a file
    [ ! -f "$filepath" ] && continue
    
    # Get filename without path
    filename=$(basename "$filepath")
    
    # Check if substring exists in filename
    if [[ "$filename" != *"$SUBSTRING"* ]]; then
        continue
    fi
    
    # Remove substring
    newname="${filename//$SUBSTRING/$REPLACE}"
    
    # Skip if new name is empty
    if [ -z "$newname" ]; then
        echo "⚠️  Skipping '$filename': result would be empty"
        continue
    fi
    
    # Check if target already exists
    if [ -f "$DIRECTORY/$newname" ]; then
        echo "⚠️  Skipping '$filename': '$newname' already exists"
        continue
    fi
    
    count=$((count + 1))
    
    if [ "$EXECUTE" = true ]; then
        # Actually rename
        mv "$filepath" "$DIRECTORY/$newname"
        echo "[RENAMED] '$filename' → '$newname'"
        renamed=$((renamed + 1))
    else
        # Dry run - just show
        echo "[DRY RUN] '$filename' → '$newname'"
    fi
done

echo "------------------------------------------------------------"
if [ "$EXECUTE" = true ]; then
    echo "Renamed $renamed file(s)"
else
    echo "Found $count file(s) to rename"
    echo "Run with --execute flag to actually rename files"
fi