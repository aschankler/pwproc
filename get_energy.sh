#!/bin/bash

E_LINES="$(grep '![[:space:]]*total energy' $@)"

echo "$(sed -n 's/!.*=[[:space:]]*\([-.[:digit:]]\)/ \1/p' <<< "$E_LINES")"
