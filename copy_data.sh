#!/bin/bash
echo "Używać z katalogu w którym są te MOP02 itd" 
if [ -z "$1" ]; then
	echo "usage $0 <destination>";
	exit 1;
fi;

DESTINATION="$1"
rm -rf .tmp_paths
mkdir -p "$DESTINATION"
for j in $(for i in $(ls); do cut -d' ' -f1 $i/$(basename $i).txt | awk '{print $0}'; done); do  echo $j |  tr ',' ' ' | awk '{print $1,$2} '| tr '\\' '/' | tr ' ' '/' >> .tmp_paths; done;

while read i; do
 cp --parents $(echo $i | sed -s 's/p1/P1/' | sed -s 's/mop/Mop/') $DESTINATION
done <.tmp_paths;

for i in $(ls); do cp --parents -r $i/MOP* $DESTINATION; done;

rm .tmp_paths

