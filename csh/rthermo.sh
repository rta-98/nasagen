#!/usr/bin/env bash 

set -ex 

dir1="$1"
#outDir="$2" 

#mkdir -p "$outDir" 

THERMO="/home2/rappe/thermo.csh" 

cd "$dir1" || exit 1 
for file in "$dir1"/*.chk; do
  [ -f "$file" ] || continue 
  filename="${file%.chk}" 
  csh "$THERMO" "$filename" 
done  
echo "Done." 
