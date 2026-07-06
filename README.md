# hackathon-agent

Generalist AI agent MVP. Reads tasks from `tasks.json`, processes each one,
and writes the results to `results.json`. The processing step currently uses
a mock agent (`main.MockAgent`) — no real Fireworks calls are made yet.

## Environment variables

| Variable              | Required | Default                                     |
| ---------------------- | -------- | -------------------------------------------- |
| `FIREWORKS_API_KEY`    | no (unused until Fireworks is wired in) | -                |
| `FIREWORKS_MODEL`      | no (unused until Fireworks is wired in) | -                |
| `FIREWORKS_BASE_URL`   | no       | `https://api.fireworks.ai/inference/v1`       |
| `INPUT_PATH`           | no       | `/input/tasks.json`                           |
| `OUTPUT_PATH`          | no       | `/output/results.json`                        |

No API keys or model names are hardcoded in the source.

## Task format

`tasks.json` is a JSON array of tasks:

```json
[
  { "task_id": "t1", "prompt": "..." },
  { "task_id": "t2", "prompt": "..." }
]
```

`results.json` is a JSON array of results, one per task, in the same order:

```json
[
  { "task_id": "t1", "answer": "..." },
  { "task_id": "t2", "answer": "..." }
]
```

Rules:
- The `/output` directory is created automatically if it doesn't exist.
- `results.json` is always valid JSON.
- If a single task fails (missing prompt, agent error), its `answer` is
  `"Unable to answer the task."` — the rest of the batch still runs.
- If the input file is missing or not valid JSON, the process exits with a
  non-zero status and writes nothing.
- On success, the process exits with status `0`.

## Run locally

```bash
pip install -r requirements.txt
INPUT_PATH=./tasks.json OUTPUT_PATH=./results.json python main.py
```

## Run with Docker

```bash
docker build -t hackathon-agent .
docker run --rm \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_MODEL=... \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  hackathon-agent
```

## Tests

Tests use `pytest` (not included in `requirements.txt`, which only lists
production dependencies):

```bash
pip install pytest
pytest
```
