# hackathon-agent

Generalist AI agent MVP. Reads tasks from `tasks.json`, routes each one to a
handler, optionally calls Fireworks AI, and writes the results to
`results.json`.

## Environment variables

| Variable              | Required | Default                                     |
| ---------------------- | -------- | -------------------------------------------- |
| `FIREWORKS_API_KEY`    | yes (for `llm` tasks) | -                              |
| `FIREWORKS_MODEL`      | yes (for `llm` tasks) | -                              |
| `FIREWORKS_BASE_URL`   | no       | `https://api.fireworks.ai/inference/v1`       |
| `INPUT_PATH`           | no       | `/input/tasks.json`                           |
| `OUTPUT_PATH`          | no       | `/output/results.json`                        |

No API keys or model names are hardcoded in the source.

## Task format

`tasks.json` is either a JSON array of tasks, or an object with a `tasks` key:

```json
[
  { "id": "1", "type": "math", "input": "2 + 2 * 3" }
]
```

Supported `type` values: `math`, `llm`, `sentiment`, `summary`, `ner`, `code`, `logic`.
Only `math` and `llm` have real logic today; the rest are placeholders that
return `{"status": "not_implemented"}`.

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
