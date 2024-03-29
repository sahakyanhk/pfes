#!/bin/bash

input_pdb_path=$1
chain=$2

SS=$(dssp ${input_pdb_path} | sed -n '/#/,$p' | awk -v chain=$chain '$3==chain {print substr($0, 17,1)}' | tr ' ' '-' | tr -d  '\n')
max_helix=$(grep -Eo 'H+' <<< "$SS" | awk '{print length($1)}' | sort -rnk1 | head -1) #returns lenght of the longest HHH... string in $SS

echo -n "$SS",$((max_helix + 1))