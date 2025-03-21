#!/bin/sh
set -e

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN environment variable is not set"
    exit 1
fi

exec "$@"
