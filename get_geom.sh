#!/bin/bash

# Select the geometry output chunk
RAW_OUTPUT=$(sed -n '/Begin final coordinates/,/End final coordinates/p' $1)

# Parse the output
ALAT=$(echo "$RAW_OUTPUT" | sed -n 's/.*(alat=\(.*\))/\1/p')
LATTICE=$(sed -n '/./{H;$!d} ; x ; s/.*CELL_PARAMETERS[^\n]*\n\(.*\)/\1/p' <<< "$RAW_OUTPUT")
GEOM=$(sed -n '/./{H;$!d} ; x ; s/.*ATOMIC_POSITIONS (.*)\n\(.*\)\nEnd final coordinates/\1/p' <<< "$RAW_OUTPUT")

if [ $ALAT ]; then
    ALAT="ALAT = $ALAT\n\n"
fi

if [ $LATTICE ]; then
    LATTICE="LATTICE = \"\"\"$LATTICE\"\"\"\n\n"
fi


OUT="$ALAT$LATTICE
GEOMETRY = \"\"\"$GEOM\"\"\"
"

echo "$OUT"
