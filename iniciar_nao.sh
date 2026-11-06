#!/bin/sh
# Inicializa apenas o conector de camera, sem comandos de movimento.
set -eu
BASE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ -d /opt/aldebaran/lib/python2.7/site-packages ]; then
    export PYTHONPATH="/opt/aldebaran/lib/python2.7/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="/opt/aldebaran/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
SERVER_URL="${1:-http://192.168.16.162:8765}"
exec python -u "$BASE_DIR/nao_client.py" --server "$SERVER_URL" --robot 127.0.0.1 --key-file "$BASE_DIR/.access-key" --node-id-file "$BASE_DIR/.node-id"
