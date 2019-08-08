#!/bin/bash

DIR=$(dirname "$(readlink -f "$0")")

case "$1" in

template)
    shift
    python3 $DIR/template_sub.py $@
    ;;
energy)
    shift
    $DIR/get_energy.sh $@
    ;;
geometry)
    shift
    $DIR/get_geom.sh $@
    ;;
bands)
    shift
    python3 $DIR/parse_bands.py $@
    ;;
*)
    echo "Available modes are 'template,' 'energy,' 'geometry' and 'bands'"
    exit 1
    ;;
esac
