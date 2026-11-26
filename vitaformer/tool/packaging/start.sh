#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -x runtime/env/bin/python ] || { echo "Run ./install.sh first."; exit 1; }
exec runtime/env/bin/python -m vitaformer app "$@"
