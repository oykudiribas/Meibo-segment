#!/bin/bash
# volkbay[2023]
FILES="../../img/raw/*"
IND=0
TOGGLE=0
for f in $FILES
do
  if [[ $TOGGLE -eq 0 ]]
  then
    mv "$f" $(printf '../../img/raw/patient%04d_OD.jpeg' $IND)
  else
    mv "$f" $(printf '../../img/raw/patient%04d_OS.jpeg' $IND)
    ((IND++))
  fi
  TOGGLE=$((1-$TOGGLE))
done
