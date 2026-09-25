"""
This is my basic Loop. Dummy tools for now
Minimal agent loop against Ollama. No frameworks, stdlib only.

    ollama pull qwen3        # or llama3.1, mistral-nemo, any tool-capable model
    python agent.py

Logging:
    LOG_LEVEL=DEBUG python agent.py     # also logs full message payloads
    Logs go to the console and to agent.log (set LOG_FILE to change).
"""
import json
import logging
import os
import subprocess
import time
import urllib.request
from datetime import datetime

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
MAX_STEPS = 10


# ---------- Logging ----------

def setup_logging():
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file = logging.FileHandler(os.environ.get("LOG_FILE", "agent.log"), encoding="utf-8")
    file.setFormatter(fmt)
    file.setLevel(logging.DEBUG)                 # file always gets everything

    log = logging.getLogger("agent")
    log.setLevel(logging.DEBUG)
    console.setLevel(level)
    log.addHandler(console)
    log.addHandler(file)
    return log

log = setup_logging()

def short(text, n=300):
    """Truncate long strings for readable INFO logs."""
    text = str(text).replace("\n", "\\n")
    return text if len(text) <= n else text[:n] + f"... (+{len(text) - n} chars)"


# ---------- Tools: plain Python functions + a JSON schema for each ----------

def get_time() -> str:
    return datetime.now().isoformat()

def calculator(expression: str) -> str:
    # Restricted eval: numbers and operators only
    if not set(expression) <= set("0123456789+-*/(). %"):
        return "error: invalid characters"
    return str(eval(expression, {"__builtins__": {}}))

def run_shell(command: str) -> str:
    # Dangerous in real life -- keep it for demo, or gate it with a confirm prompt
    if input(f"  allow `{command}`? [y/N] ").lower() != "y":
        return "user denied"
    r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return (r.stdout + r.stderr)[-4000:]

TOOLS = {"get_time": get_time, "calculator": calculator, "run_shell": run_shell}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_time", "description": "Get the current local date and time",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command and return its output",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]


# ---------- The model call ----------

def chat(messages):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "tools": TOOL_SCHEMAS, "stream": False}).encode()
    log.debug("request: %d messages, last=%s", len(messages), short(messages[-1], 1000))

    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    elapsed = time.perf_counter() - t0

    log.info("model responded in %.2fs (prompt=%s tok, output=%s tok)",
             elapsed, data.get("prompt_eval_count", "?"), data.get("eval_count", "?"))
    log.debug("raw response: %s", short(data["message"], 2000))
    return data["message"]


# ---------- The loop ----------

def run_agent(user_input, messages):
    log.info("=== new task: %s", short(user_input))
    messages.append({"role": "user", "content": user_input})

    for step in range(1, MAX_STEPS + 1):
        log.info("--- step %d/%d: calling %s", step, MAX_STEPS, MODEL)
        try:
            msg = chat(messages)
        except Exception:
            log.exception("model call failed")
            raise
        messages.append(msg)                      # keep assistant turn (incl. tool_calls)

        if msg.get("thinking"):                   # reasoning models (qwen3, etc.)
            log.info("thinking: %s", short(msg["thinking"]))
        if msg.get("content"):
            log.info("assistant: %s", short(msg["content"]))

        calls = msg.get("tool_calls") or []
        if not calls:                             # no tools requested -> final answer
            log.info("=== done after %d step(s)", step)
            return msg.get("content", "")

        log.info("model requested %d tool call(s)", len(calls))
        for call in calls:                        # execute each tool, feed result back
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):             # some models return a JSON string
                args = json.loads(args or "{}")

            log.info("tool call: %s(%s)", name, json.dumps(args))
            t0 = time.perf_counter()
            try:
                if name not in TOOLS:
                    raise KeyError(f"unknown tool '{name}'")
                result = TOOLS[name](**args)
                log.info("tool result (%.2fs): %s", time.perf_counter() - t0, short(result))
            except Exception as e:
                result = f"error: {e}"
                log.warning("tool %s failed: %s", name, e)
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

    log.warning("=== stopped: hit MAX_STEPS (%d)", MAX_STEPS)
    return "(stopped: hit MAX_STEPS)"


if __name__ == "__main__":
    log.info("starting agent: model=%s url=%s", MODEL, OLLAMA_URL)
    history = [{"role": "system",
                "content": "You are a helpful assistant. Use tools when they help."}]
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in {"exit", "quit"}:
            break
        if q:
            print("agent>", run_agent(q, history))
    log.info("agent exiting")
