# Dataset Inspector Agent
---
## Acomplished So Far
- Basic Script
- Ollama Setup

## Next Step
- CV Tools

## Models Used
-  Small Nvidia 2070 8GB:
     - Qwen3.5:9b
-  Medium A5000x2 (2x24 GB):
-  Look at the Peasants (1-8 x H200):
---
An LLM agent for auditing image datasets. Point it at a folder of images (and optionally labels), ask questions in plain English, and it decides which tools to run and reports back. Example questions:

- "Are there near-duplicates?"
- "Which classes look mislabeled?"
- "Show me the blurriest 20 images."

This is a learning project, built in four stages. Each stage teaches a core agent concept:

| Stage | Concept |
|---|---|
| 1. The bare loop | The tool-calling cycle |
| 2. Real CV tools | Tool design |
| 3. State and scale | Memory and context management |
| 4. Evaluation | Measuring the agent instead of eyeballing it |

The agent loop is written by hand with the `ollama` Python library against a self-hosted [Ollama](https://ollama.com) server, with no agent framework, so every step is visible.

---

## Setup

**Requirements:** Python 3.11+, Ollama installed and running (`ollama serve`)

```bash
pip install ollama pillow numpy opencv-python torch open_clip_torch faiss-cpu
# or use transformers instead of open_clip_torch for DINOv2
```

### Model choice

You need two capabilities, which may or may not come from the same model:

- **Tool calling** for the agent itself. In the Ollama model library, filter by the "tools" tag. Qwen-family models are a solid starting point.
- **Vision** for `view_image`. Filter by the "vision" tag.

```bash
ollama pull <tool-capable-model>
ollama pull <vision-model>     # skip if your agent model does both
```

Start with a mid-sized model (roughly 7–14B) that fits comfortably in VRAM alongside your embedding model. Local models vary a lot in how reliably they call tools, and that variation is part of what you'll learn.

### Set the context window explicitly

Ollama's default context window (`num_ctx`) is small, and when a conversation exceeds it, the oldest content is **silently truncated**. Agents fill context fast with tool results, so always pass it explicitly:

```python
ollama.chat(model=MODEL, messages=messages, tools=TOOLS,
            options={"num_ctx": 16384})
```

Larger values cost VRAM, so pick the biggest your GPU handles alongside the other models.

**Test dataset:** [Imagenette](https://github.com/fastai/imagenette), a 10-class ImageNet subset of about 13k images. Carve out a 500-image `dev` subset for fast debugging.

### Repo layout

```
inspector/
  agent.py        # the loop
  tools/          # one file per tool group
  registry.py     # tool schemas + dispatch
  cache/          # embeddings, results
  evals/          # stage 4
  logs/           # full transcripts of every run
LESSONS.md        # failure modes and fixes (start in Stage 3)
```

### Logging from day one

Dump every message, tool call, and tool result to a JSONL file per run in `logs/`. Reading these transcripts is how you'll understand what the agent is actually doing.

---

## Stage 1: The bare loop

**Goal:** a working agent with trivial tools, and a clear picture of the request → tool call → result → repeat cycle.

### Build

- [ ] `run_agent(user_prompt)` calls `ollama.chat(...)` with the conversation and tool definitions (JSON-schema function specs).
- [ ] If `response.message.tool_calls` is non-empty, append the assistant message, run each requested tool, append each result as a `{"role": "tool", "tool_name": ..., "content": ...}` message, then call again.
- [ ] Stop when the model replies with no tool calls. That reply is the final answer.
- [ ] Cap the loop at N iterations (start with 15) so it can't run away.
- [ ] Validate tool arguments before running. Local models sometimes produce wrong argument names, wrong types, or calls to tools that don't exist. Return a clear error message as the tool result rather than crashing.

### Tools

- [ ] `list_images(path, limit)` returns file paths and a count.
- [ ] `get_image_stats(path)` returns size, mode, file size, and whether it's corrupted.
- [ ] `view_image(path, question)` lets the agent look at an image. Tool results in Ollama are text, so choose one approach:
  - **Describe-by-proxy (simpler):** the tool sends the image and question to your vision model and returns its text description.
  - **Inject the image (if your agent model has vision):** the tool returns a short note, and the loop appends a user message with the image in its `images` field.

**Done when:** asking "how many images are in this folder, and are any unusual in size?" gets a correct answer.

**Things to notice:**
- How the model chooses tools from the descriptions alone.
- What happens when a tool raises an exception. Return errors as tool results and watch whether it recovers.
- How often the model calls tools badly. Keep a tally, since it's a useful number for comparing models later.

---

## Stage 2: Real CV tools

**Goal:** learn tool design, which is the most important skill in agent building.

### Tools

- [ ] `compute_embeddings(path)` embeds everything with CLIP or DINOv2, caches it to disk, and returns only a summary ("embedded 13,394 images, cached at X").
- [ ] `find_near_duplicates(threshold, top_k)` uses a FAISS search over the cache and returns the top-k pairs with similarity scores.
- [ ] `find_outliers(top_k)` returns images far from their class centroid, or with low kNN density.
- [ ] `score_quality(path_or_all)` computes Laplacian variance for blur, mean/std brightness for exposure, and returns the worst offenders.
- [ ] `check_label_agreement(top_k)` runs zero-shot CLIP or a pretrained classifier against the provided labels and returns the most confident disagreements.

### Tool design rules

Follow these, and test them by breaking them:

1. Tools return summaries and top-k lists, never raw arrays.
2. Descriptions say what the tool does, when to use it, and what it returns.
3. Expensive work happens once and is cached. The agent should never embed the same dataset twice.
4. Outputs include enough context to act on, such as paths, scores, and labels.
5. Keep the tool list short and argument schemas simple. Smaller local models degrade noticeably as the number of tools and optional arguments grows.

**VRAM note:** the LLM, the vision model, and your embedding model all compete for the same GPU. Compute embeddings once up front, then free that model, or run it on CPU if memory is tight. Ollama unloads idle models after a timeout, and reloading costs seconds per call.

**Done when:** "find likely label errors and show me the three worst" produces a sensible answer, and the agent looks at those images with `view_image` before concluding.

**Experiment:** make one tool's description vague and watch the agent misuse it. This shows you directly how much descriptions matter.

---

## Stage 3: State and scale

**Goal:** handle long, multi-step investigations without losing the thread.

### Build

- [ ] A findings store (SQLite or JSON) with `record_finding(type, paths, note)` and `get_findings()` tools, so the agent keeps its conclusions outside the context window.
- [ ] A `write_report(path)` tool that produces a markdown summary of everything found.
- [ ] Context management: track token usage (`prompt_eval_count` in each response) and truncate or summarize old tool results before you approach `num_ctx`. Don't rely on Ollama's silent truncation, which drops the system prompt and original task first.

### Stress tests

- [ ] Run "do a full audit of this dataset" on all of Imagenette.
- [ ] Watch for these failure modes:
  - context overflow (the agent "forgets" its task or tools)
  - repeated identical tool calls
  - premature "done"
  - hallucinated findings that were never recorded
  - tool calls written as plain text in the reply instead of real tool calls
- [ ] Fix each failure mode as you hit it, and note the fix in `LESSONS.md`. That file will be the most valuable thing you produce.

**Done when:** a full audit runs end to end and produces a report whose claims all trace back to tool outputs in the log.

---

## Stage 4: Evaluation

**Goal:** measure the agent instead of eyeballing it.

### Build

- [ ] A corruption script that takes a clean subset and injects known problems:
  - about 50 exact and near duplicates (crops, JPEG recompression, small rotations)
  - about 5% flipped labels
  - 20 corrupted or truncated files
  - 30 heavily blurred or darkened images
- [ ] Save the ground truth to `evals/manifest.json`.
- [ ] Run the full audit, parse the findings store, and compute precision and recall per problem type.

### Baselines

- [ ] Your Stage 2 tools run directly as a script, with no agent. This tells you what the agent adds beyond orchestration.
- [ ] [`cleanlab`](https://github.com/cleanlab/cleanlab) for label noise.

### Questions to answer

- Does the agent beat the plain script, or just add latency and variance?
- How consistent is it across 5 runs with the same seed data?
- What does a full audit cost in tokens (`prompt_eval_count` + `eval_count`) and wall-clock time?
- How do results change across model sizes? Running the same eval on two or three local models is easy with Ollama and makes a good results table.

**Done when:** you have a small results table and an honest answer to "was the agent worth it?", whichever way it comes out.

---

## Stretch ideas

- Let the agent write and run its own analysis code in a sandbox, instead of only calling fixed tools.
- Add a human-in-the-loop step, where it proposes removals and you approve them.
- Point it at one of your own work datasets. That's the real test.
- Try constrained output (Ollama's `format` parameter with a JSON schema) for the final report, and compare reliability against free-text.

---

## Time budget

| Stage | Rough estimate |
|---|---|
| 1. The bare loop | A weekend |
| 2. Real CV tools | A weekend |
| 3. State and scale | A week of evenings |
| 4. Evaluation | A week of evenings |

Stage 1 will feel almost too easy, but don't skip the logging. It pays off in Stage 3.
