# hackathon-agent

## 1. What the agent does

`GeneralPurposeAgent` reads a batch of tasks, classifies each `prompt` into
one of eight categories (`router.classify_task`), and resolves it:

- **math**: tried locally first (`math_handler.try_solve_math`, regex +
  controlled arithmetic, no `eval()`) to save LLM tokens. Only falls back to
  an LLM call when the phrasing can't be solved with confidence.
- **factual, sentiment, summarization, ner, code_debugging,
  code_generation, logic**: each gets a short, category-specific prompt
  (see `handlers/`) sent to Fireworks AI through an OpenAI-compatible
  client (`fireworks_client.py`).

`main.py` reads the task list from `/input/tasks.json`, processes each task
through the agent, and writes the answers to `/output/results.json`. If a
single task fails for any reason (missing prompt, missing credentials, LLM
error), its answer becomes `"Unable to answer the task."` instead of
aborting the whole batch. The Fireworks client and its config are created
lazily and reused across tasks — a run made only of math tasks never touches
the Fireworks environment variables at all.

## 2. `/input/tasks.json` structure

A JSON array of tasks:

```json
[
  { "task_id": "t1", "prompt": "..." },
  { "task_id": "t2", "prompt": "..." }
]
```

A sample file with one task per required category (math, sentiment,
summary, factual) is provided at [`input/tasks.json`](input/tasks.json).

## 3. `/output/results.json` structure

A JSON array of results, one per task, in the same order as the input:

```json
[
  { "task_id": "t1", "answer": "..." },
  { "task_id": "t2", "answer": "..." }
]
```

Rules:
- The `/output` directory is created automatically if it doesn't exist.
- `results.json` is always valid JSON.
- If the input file is missing or not valid JSON, the process exits with a
  non-zero status and writes nothing. On success, it exits with status `0`.

## 4. Environment variables

| Variable              | Required | Default                |
| ---------------------- | -------- | ------------------------ |
| `FIREWORKS_API_KEY`    | yes, for any task that needs the LLM | - |
| `FIREWORKS_BASE_URL`   | yes, for any task that needs the LLM | - |
| `ALLOWED_MODELS`       | yes, for any task that needs the LLM | - |
| `INPUT_PATH`           | no       | `/input/tasks.json`      |
| `OUTPUT_PATH`          | no       | `/output/results.json`   |

No API keys or model names are hardcoded in the source. `ALLOWED_MODELS` is
a comma-separated list (e.g. `model-a,model-b`); `fireworks_client.choose_model`
picks the first entry that looks like a small/efficient model (name
containing `small`, `mini`, `8b`, `7b`, `qwen`, or `llama`), otherwise the
first entry in the list.

## 5. Build the image

```bash
docker build -t hackathon-agent .
```

## 6. Run locally with volumes

```bash
docker run --rm \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=... \
  -e ALLOWED_MODELS=... \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  hackathon-agent
```

This reads `./input/tasks.json` (the sample file is already there) and
writes `./output/results.json`.

## Local testing

Run the unit tests (mocked/faked LLM calls only, no real network access, no
Fireworks credentials required — `requirements-dev.txt` is test-only and is
never installed into the production image):

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Run the agent directly with Python, against the sample input, without
building the Docker image:

```bash
FIREWORKS_API_KEY=... FIREWORKS_BASE_URL=... ALLOWED_MODELS=... \
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json \
python main.py
cat ./output/results.json
```

Try it with no Fireworks variables set at all — the math task in the sample
input still resolves correctly (solved locally), while the other tasks fall
back to `"Unable to answer the task."` instead of crashing the run:

```bash
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python main.py
cat ./output/results.json
```
