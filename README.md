# Lymerorium

**A local AI debate engine that beats single-model reasoning by making seven models argue.**

Most AI tools send your question to one model and trust the answer. Lymerorium does something different: it runs seven specialized agents across three different model families, has them debate the question, fact-check each other with live tools, and only then synthesizes a verdict. The entire system runs locally on a Jetson Orin Nano Super and a single laptop GPU. No cloud, no API bills.

On the MMLU knowledge benchmark, this pushed accuracy from **54% (single model) to 80% (full swarm)** at n=50, a statistically significant gain (McNemar exact test, p = 0.0023).

The project began as a camera assistant for a robot hand and grew into a full distributed reasoning system with its own benchmarking harness.

---

## Highlights

- **+26 points on MMLU** over a single-model baseline, verified with a proper paired significance test.
- **100% local.** Three nodes, two consumer machines, zero cloud inference cost.
- **Three model families** (Qwen, Llama, Gemma) for genuine cognitive diversity, not just repeated sampling of one model.
- **Fast when it can be.** A pre-debate gate answers 90% of easy questions instantly and only spins up the full debate when models disagree.
- **Built to not fall over.** Circuit breakers, a memory-pressure guard, and graceful degradation when a node drops.
- **Live vision.** Point a camera at the world and ask the swarm about what it sees.

---

## How it works

**The fast path (about 90% of questions).** A pre-debate gate runs three quick independent answers. If they agree unanimously, that answer is returned immediately. This keeps the system responsive on questions that do not need a debate.

**The debate path (about 10% of questions).** When the quick answers disagree, seven agents split into three groups and argue across multiple rounds:

1. Each agent states a position, backed by evidence it pulls from memory and live tools.
2. Agents that just repeat themselves are temporarily silenced, keeping the debate moving forward.
3. When two agents sharply disagree, a targeted web search fires on the exact point of contention.
4. Groups exchange summaries between rounds so good arguments cross-pollinate.
5. If the group agrees too quickly, a forced challenge round guards against premature consensus.
6. A Synthesizer agent reads every argument and writes the final verdict with a calibrated confidence score.

The whole pipeline is a LangGraph state machine, so every step is explicit, inspectable, and resumable.

---

## Results

Six evaluation runs were conducted between June 15 and June 26, 2026. The first three were thrown out due to infrastructure faults (a downed vision node, a memory leak in the Jetson inference server, and a model accidentally pinned to CPU). Run 4 was the first clean run with all three model families live. Runs 5 and 6 added a self-consistency control arm and fixed confidence calibration.

**Run 6, the final evaluation (n=50, four arms, two benchmarks):**

MMLU (multiple-choice knowledge):
- Single-model baseline (Qwen3:8b): **0.54**
- Self-consistency (same model sampled three times, majority vote): **0.42**
- Full swarm (heterogeneous debate): **0.80**

The 26-point swarm gain over baseline is statistically significant (McNemar exact test, p = 0.0023). Critically, self-consistency scored *below* baseline, which confirms the improvement comes from genuine model diversity in the debate, not from simply sampling one model more often.

The pre-debate gate fired on 90% of MMLU questions. Gated and fully-debated questions scored identically (both 0.80), showing the gate routes easy questions correctly without giving up any accuracy.

**Known limitation, reported honestly.** On StrategyQA (binary yes/no commonsense), the swarm regressed from 0.62 to 0.52, a roughly 10-point drop that held consistent across every run. On short yes/no questions, debate can amplify a shared wrong intuition rather than correct it. This is a real failure mode of 8B-scale models on short-answer tasks and is documented here rather than hidden.

The complete machine-readable results are committed at `bench/evalcard_20260626_111506.json`.

---

## Hardware

Three nodes, each running a different model family:

- **Jetson Orin Nano Super (8 GB)** runs Gemma 4 E2B via llama-server. Handles the camera, vision tasks, and fast routing calls.
- **Laptop GPU (8 GB VRAM)** runs Qwen3:8b via Ollama, driving the heavy debate agents and the final synthesis.
- **Laptop GPU (shared)** runs Llama 3.1:8b via Ollama, adding a second, independent model family.

Running three distinct families means three different training histories and three different sets of blind spots, which is exactly why the swarm catches errors that any single model would miss.

---

## Agents

Eight agents, each with a fixed personality and a curated tool set:

- **Skeptic** (Qwen3:8b) probes weak evidence and challenges assumptions
- **Technologist** (Qwen3:8b) verifies numbers and runs symbolic math
- **Realist** (Llama3.1:8b) anchors the debate to real-world constraints
- **Economist** (Llama3.1:8b) models incentives and runs quantitative checks
- **Ethicist** (Llama3.1:8b) raises societal and fairness concerns
- **Visionary** (Gemma E2B) maps future possibilities and searches broadly
- **Contrarian** (Gemma E2B) defends the least-popular position on purpose
- **Synthesizer** (Qwen3:8b) reads every argument and writes the verdict

---

## Tools

Each agent gets a subset of tools matched to its role. Tools run in parallel before each speak round.

- **web_search / web_search_and_fetch** search the web and optionally fetch full page content
- **wiki_summary** returns a Wikipedia article summary
- **wikidata_entity** pulls structured facts from Wikidata
- **openalex_search** searches open academic literature (free, no key)
- **semantic_scholar_search** searches papers via Semantic Scholar
- **sympy_solve** runs a symbolic math solver for algebra and calculus
- **python_eval** evaluates sandboxed Python arithmetic
- **pint_convert** converts physical units
- **frankfurter_fx** fetches live foreign exchange rates
- **camera_snapshot** asks the vision model about the current camera frame
- **search_knowledge_graph** searches the local debate-history graph

Optional keyed tools: Tavily for higher-quality search and Wolfram Alpha.

---

## Memory and knowledge graph

Past verdicts are stored as vector embeddings (all-MiniLM-L6-v2 via usearch). When a new question arrives, the system checks whether a recent, similar debate already answered it, and three agents vote on whether enough has changed to justify a fresh debate.

Entities and relationships extracted from each debate are stored in a local NetworkX graph and can optionally sync to a Neo4j AuraDB instance to persist across restarts.

---

## Reliability

- **Circuit breaker** on every LLM endpoint, opening after five consecutive failures and resetting after sixty seconds.
- **Resource guard** that samples GPU VRAM and system RAM before each call and aborts gracefully instead of letting the machine hit an out-of-memory crash.
- **Background swarm** that continuously pre-warms the knowledge graph between requests.
- **Live streaming** of agent speak events to the browser over server-sent events.
- **MCP server** exposing every swarm tool as a standard endpoint for external agent clients.

---

## Quick start

Requirements: Python 3.11 or newer, Ollama on the laptop, llama-server on the Jetson.

```bash
git clone https://github.com/your-username/Lymerorium
cd Lymerorium
pip install -r swarm_core/requirements.txt
cp swarm_core/.env.example swarm_core/.env
# fill in your Jetson IP, model names, and any optional API keys
```

Start the vision node on the Jetson:

```bash
llama-server -m gemma-4-E2B-it-Q4_K_M.gguf --port 8080 --ctx-size 8192 --ctx-checkpoints 0 -np 1
```

Start the reasoning models on the laptop:

```bash
ollama serve
ollama pull qwen3:8b
ollama pull llama3.1:8b
```

Launch the web app:

```bash
python app.py
# open http://localhost:5000
```

Or run a single debate from the terminal, no camera needed:

```bash
python swarm_core/main.py "Your question here" 2
```

---

## Reproducing the benchmarks

Fetch the datasets:

```bash
python bench/fetch_data.py
```

Run the full eval card (four arms, two benchmarks):

```bash
python bench/eval_card.py --bench mmlu,strategyqa --n 50 --arms baseline,self_consistency,swarm
```

Run the lighter benchmark (GSM8K math and TruthfulQA):

```bash
python bench/run_bench.py --bench gsm8k,truthfulqa --n 20 --arms baseline,swarm
```

Smoke test (one question, confirms the plumbing works):

```bash
python bench/run_bench.py --smoke
```

---

## Project structure

```
app.py                      Flask web app, camera feed, and swarm integration
jetson_watchdog.ps1         Restarts llama-server on the Jetson if it goes down
swarm_core/
    simulation.py           LangGraph debate state machine
    agents.py               Agent personalities, LLM routing, circuit breakers
    api.py                  Public interface called by app.py
    tools.py                All agent tools
    memory.py               Vector memory for past verdicts
    knowledge_graph.py      NetworkX and Neo4j knowledge graph
    oracle_chat.py          Single-shot question answering via the swarm
    resource_guard.py       Proactive GPU and RAM safety guard
    background.py           Continuous background debate loop
    mcp_server.py           MCP tool server
    config.py               Environment-configurable settings
bench/
    eval_card.py            Full four-arm evaluation harness
    run_bench.py            Lighter two-arm benchmark
    metrics.py              Accuracy, calibration, McNemar, and routing metrics
    fetch_data.py           Dataset downloader
    evalcard_20260626_111506.json   Run 6 results (n=50)
```

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

