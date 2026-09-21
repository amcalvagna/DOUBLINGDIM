#!/usr/bin/env bash

# Usage:
#   ./manage_tree.sh <topdir> <subdir1> [subdir2 ...]
#
# Example:
#   ./manage_tree.sh data images logs results

set -e  # exit on any error

TOPDIR="$1"
shift  # remaining arguments are subdirs

if [ -z "$TOPDIR" ] || [ $# -eq 0 ]; then
    echo "Usage: $0 <topdir> <subdir1> [subdir2 ...]"
    exit 1
fi

for SUBDIR in "$@"; do
    FULLPATH="./${TOPDIR}/${SUBDIR}"

    if [ -d "$FULLPATH" ]; then
        echo "Directory exists: $FULLPATH"
        echo "Cleaning contents..."
        rm -rf "${FULLPATH:?}"/*   # safely remove contents only
    else
        echo "Creating directory: $FULLPATH"
        mkdir -p "$FULLPATH"
    fi
done

echo "Done."
