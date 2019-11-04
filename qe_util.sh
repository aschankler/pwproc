#!/bin/bash

DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH="$DIR"

case "$1" in

template)
    shift
    python3 $DIR/template/template.py $@
    ;;
scf)
    shift
    python3 $DIR/pwproc/scf.py $@
    ;;
relax)
    shift
    python3 $DIR/pwproc/relax.py $@
    ;;
fermi)
    shift
    python3 $DIR/pwproc/fermi.py $@
    ;;
bands)
    shift
    python3 $DIR/pwproc/bands.py $@
    ;;
*)
    echo "Available modes are 'template,' 'scf', 'relax', 'fermi' and 'bands'"
    exit 1
    ;;
esac
