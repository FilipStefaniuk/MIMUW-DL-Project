#!/bin/bash
set -euox pipefail
cp models/ps360958_4.py models/ps360958_$1.py
cat configs/ps360958_4.json | sed "s/ps360958_4/ps360958_$1/g" > configs/ps360958_$1.json
cat mains/ps360958_main4.py | sed "s/ps360958_4/ps360958_$1/g" > mains/ps360958_main$1.py


