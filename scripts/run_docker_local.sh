#!/usr/bin/env bash
# Local end-to-end smoke test: build the Docker image, run it against
# input/tasks.json with the /input and /output volumes, and verify
# output/results.json comes out valid. Dummy Fireworks credentials are only
# passed when every task in input/tasks.json classifies as "math" (solved
# locally, so the LLM is never actually called) -- no real secrets involved.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_NAME="hackathon-agent"
INPUT_DIR="$(pwd)/input"
OUTPUT_DIR="$(pwd)/output"
TASKS_FILE="$INPUT_DIR/tasks.json"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not in PATH." >&2
    exit 1
fi

if [ ! -f "$TASKS_FILE" ]; then
    echo "ERROR: $TASKS_FILE not found." >&2
    exit 1
fi

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

PYTHON_BIN="python3"
if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

echo "==> Building Docker image..."
docker build -t "$IMAGE_NAME" .

echo "==> Checking whether $TASKS_FILE contains only local math tasks..."
ONLY_MATH=$("$PYTHON_BIN" - "$TASKS_FILE" <<'PYEOF'
import json
import sys

sys.path.insert(0, ".")
from router import TaskType, classify_task

with open(sys.argv[1], "r", encoding="utf-8") as f:
    tasks = json.load(f)

only_math = bool(tasks) and all(classify_task(t.get("prompt", "")) == TaskType.MATH for t in tasks)
print("true" if only_math else "false")
PYEOF
)

ENV_ARGS=()
if [ "$ONLY_MATH" = "true" ]; then
    echo "==> All tasks are local math tasks; using dummy Fireworks credentials (never actually used)."
    ENV_ARGS=(
        -e "FIREWORKS_API_KEY=dummy-key"
        -e "FIREWORKS_BASE_URL=http://dummy.invalid/v1"
        -e "ALLOWED_MODELS=dummy-model"
    )
else
    echo "==> $TASKS_FILE has non-math tasks; real FIREWORKS_API_KEY/FIREWORKS_BASE_URL/ALLOWED_MODELS"
    echo "    must already be exported in this shell before running this script."
fi

echo "==> Running container..."
docker run --rm \
    "${ENV_ARGS[@]}" \
    -v "$INPUT_DIR:/input" \
    -v "$OUTPUT_DIR:/output" \
    "$IMAGE_NAME"

echo "==> Verifying $OUTPUT_DIR/results.json exists..."
if [ ! -f "$OUTPUT_DIR/results.json" ]; then
    echo "ERROR: $OUTPUT_DIR/results.json was not created." >&2
    exit 1
fi

echo "==> Contents of $OUTPUT_DIR/results.json:"
cat "$OUTPUT_DIR/results.json"
