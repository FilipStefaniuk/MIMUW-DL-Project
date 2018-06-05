#!/bin/bash
set -euox pipefail
SRC=${2:-"4"}

cp models/ps360958_$SRC.py models/ps360958_$1.py
cat configs/ps360958_$SRC.json | sed "s/ps360958_$SRC/ps360958_$1/g" > configs/ps360958_$1.json
cat mains/ps360958_main$SRC.py | sed "s/ps360958_$SRC/ps360958_$1/g" > mains/ps360958_main$1.py


