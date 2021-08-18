#!/bin/bash

DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH="$DIR/src:$PYTHONPATH"

python3 -m pwproc "$@"
