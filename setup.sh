#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
RUN=0

if [[ "${1:-}" == "--run" ]]; then
  RUN=1
  shift
fi

[ -d "$VENV_DIR" ] || "$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[OK] venv: $VENV_DIR (python: $(python -V))"
echo "[OK] To keep it active after: source $VENV_DIR/bin/activate"

if [[ $RUN -eq 1 ]]; then
  [[ -f main.py ]] || { echo "[ERR] main.py not found."; exit 1; }
  echo "[OK] Running: python main.py $*"
  python -m main "$@"
fi
