"""Tools available to swarm agents: web search, camera, knowledge graph, filesystem, git, time."""


import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


import knowledge_graph as kg


# Knowledge-graph tools

def search_knowledge_graph(query: str) -> str:
    """Full-text search over local NetworkX graph nodes."""
    query_lower = query.lower()
    matches = []
    with kg._graph_lock:
        for node, data in kg._local_graph.nodes(data=True):
            if query_lower in str(node).lower() or query_lower in data.get("text", "").lower():
                matches.append({"node": node, **data})
    if not matches:
        return "No matches found."
    return json.dumps(matches[:10], indent=2, default=str)


def get_agent_opinions(agent_name: str) -> str:
    history = kg.get_agent_history(agent_name)
    if not history:
        return f"No recorded opinions for {agent_name}."
    return "\n".join(history)


def get_entity_relationships(entity: str) -> str:
    with kg._graph_lock:
        if entity not in kg._local_graph:
            return f"Entity '{entity}' not found in knowledge graph."
        out_edges = [(dst, data.get("relation", "?")) for _, dst, data in kg._local_graph.out_edges(entity, data=True)]
        in_edges = [(src, data.get("relation", "?")) for src, _, data in kg._local_graph.in_edges(entity, data=True)]
    return json.dumps({
        "entity": entity,
        "outgoing": [{"target": t, "relation": r} for t, r in out_edges],
        "incoming": [{"source": s, "relation": r} for s, r in in_edges],
    }, indent=2)


def run_neo4j_query(cypher: str) -> str:
    results = kg.query_neo4j(cypher)
    if not results:
        return "No results (or Neo4j unavailable)."
    return json.dumps(results, indent=2, default=str)


# Report writer

def write_report(topic: str, verdict: str, summary: str, rounds: list[dict]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"sim_{timestamp}.md"
    lines = [
        "# Swarm Simulation Report",
        f"**Topic:** {topic}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "", "## Verdict", verdict,
        "", "## Summary", summary,
        "", "## Debate Transcript",
    ]
    for msg in rounds:
        lines.append(f"\n**[Round {msg['round']}] {msg['agent']}:**")
        lines.append(msg["content"])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path.resolve())


# Web search

def web_search_tool(query: str, max_results: int = 5) -> str:
    """DuckDuckGo search - returns snippet list."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No web results found for: {query}"
        return "\n".join(
            f"- [{r.get('title', '')}] {r.get('body', '')}  |  {r.get('href', '')}"
            for r in results
        )
    except Exception as e:
        return f"[web_search error: {e}]"


# Fetch (MCP reference)

def fetch_url(url: str) -> str:
    """Fetch a URL and return clean readable text (HTML tags stripped)."""
    try:
        import requests as _req
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SwarmAgent/1.0)"}
        resp = _req.get(url, headers=headers, timeout=12, allow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        # strip script / style blocks
        text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>",   " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000]
    except Exception as e:
        return f"[fetch_url error: {e}]"


def web_search_and_fetch(query: str) -> str:
    """Search DuckDuckGo, fetch the full text of the top result.
    Returns full article content + other snippets - far richer than search alone."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return f"No results for: {query}"
        snippets = "\n".join(
            f"- [{r.get('title','')}] {r.get('body','')}"
            for r in results[:3]
        )
        top_url = results[0].get("href") or results[0].get("url", "")
        if top_url:
            full = fetch_url(top_url)
            return f"Full article ({top_url}):\n{full}\n\n---\nOther snippets:\n{snippets}"
        return snippets
    except Exception as e:
        return f"[web_search_and_fetch error: {e}]"


# Filesystem (MCP reference)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # vision-app/

def _safe_path(path: str) -> Path | None:
    """Resolve path and ensure it stays within the project root."""
    try:
        resolved = (Path.cwd() / path).resolve()
        if str(resolved).startswith(str(_PROJECT_ROOT)):
            return resolved
        return None
    except Exception:
        return None


def read_file(path: str) -> str:
    """Read a local file within the project directory."""
    safe = _safe_path(path)
    if safe is None:
        return f"[read_file: access denied or path outside project - {path}]"
    if not safe.exists():
        return f"[read_file: file not found - {path}]"
    if not safe.is_file():
        return f"[read_file: not a file - {path}]"
    try:
        return safe.read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception as e:
        return f"[read_file error: {e}]"


def list_directory(directory: str = ".") -> str:
    """List files and folders in a directory within the project."""
    safe = _safe_path(directory)
    if safe is None:
        return f"[list_directory: access denied - {directory}]"
    if not safe.exists():
        return f"[list_directory: not found - {directory}]"
    try:
        entries = sorted(safe.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = []
        for e in entries[:60]:
            tag = "FILE" if e.is_file() else "DIR "
            size = f" ({e.stat().st_size:,} bytes)" if e.is_file() else ""
            lines.append(f"  [{tag}] {e.name}{size}")
        return f"Contents of {safe}:\n" + "\n".join(lines)
    except Exception as e:
        return f"[list_directory error: {e}]"


# Git (MCP reference)

def git_log(n: int = 10, repo_path: str = ".") -> str:
    """Recent git commit history - author, date, message."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log",
             f"-{n}", "--pretty=format:%h  %ad  %an  %s", "--date=short"],
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode != 0:
            return f"[git_log: {result.stderr.strip() or 'not a git repo'}]"
        return result.stdout.strip() or "[git_log: no commits found]"
    except FileNotFoundError:
        return "[git_log: git not installed]"
    except Exception as e:
        return f"[git_log error: {e}]"


def git_diff(repo_path: str = ".") -> str:
    """Current uncommitted changes (file-level summary)."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--stat"],
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode != 0:
            return f"[git_diff: {result.stderr.strip()}]"
        return result.stdout.strip() or "[git_diff: no changes]"
    except Exception as e:
        return f"[git_diff error: {e}]"


# Time (MCP reference)

def get_current_time(timezone: str = "UTC") -> str:
    """Current date and time in the requested timezone."""
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")
            timezone = "UTC (fallback - unknown zone requested)"
        now = datetime.now(tz)
        return (
            f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Timezone: {timezone}\n"
            f"Day of week: {now.strftime('%A')}\n"
            f"ISO 8601: {now.isoformat()}"
        )
    except Exception as e:
        return f"[get_current_time error: {e}] - {datetime.utcnow().isoformat()}Z (UTC fallback)"


# Camera

def camera_snapshot(question: str, image_b64: str) -> str:
    """Ask the LLM about the current camera frame (base64 JPEG)."""
    if not image_b64:
        return "[camera_snapshot: no image available]"
    try:
        import requests as _req
        from config import MAX_LLM_CALL_SECONDS, get_vision_llm_config
        cfg = get_vision_llm_config()  # always use Gemma - Qwen3-14B is text-only

        # OpenAI-compatible (local, openai, groq)
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]}],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 256,
        }
        headers = {"Content-Type": "application/json"}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        resp = _req.post(cfg["base_url"], json=payload, headers=headers,
                         timeout=MAX_LLM_CALL_SECONDS)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"] or ""
        content = raw.strip()
        if "<channel|>" in content:
            content = content.split("<channel|>")[-1].strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if not content:
            m = re.search(r"<think>([\s\S]*?)(?:</think>|$)", raw)
            content = m.group(1).strip() if m else raw.strip()
        return content
    except Exception as e:
        return f"[camera_snapshot error: {e}]"


# Safe arithmetic evaluator - agents verify their math rather than hallucinating numbers.
# Only pure arithmetic; no imports, no function calls, no attribute access.
def python_eval(expression: str) -> str:
    """Evaluate a safe arithmetic expression (no file/network access)."""
    import ast
    _SAFE = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd,
    }
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        for node in ast.walk(tree):
            if type(node) not in _SAFE:
                return "[python_eval: only arithmetic operators allowed]"
        return str(eval(compile(tree, "<expr>", "eval")))  # noqa: S307 - sandbox verified above
    except Exception as e:
        return f"[python_eval error: {e}]"


# Wikipedia summary - direct factual lookups for knowledge-heavy tasks (MMLU, TruthfulQA).
def wiki_summary(topic: str) -> str:
    """Fetch a Wikipedia extract for a topic (up to 1500 chars)."""
    try:
        import requests as _req
        _HDRS = {"User-Agent": "SwarmAgent/1.0 (educational research)"}
        slug = topic.strip().replace(" ", "_")
        resp = _req.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
            headers=_HDRS, timeout=10,
        )
        if resp.status_code == 404:
            # Fall back to search for top result slug
            sr = _req.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": topic,
                        "format": "json", "srlimit": 1},
                headers=_HDRS, timeout=10,
            )
            hits = sr.json().get("query", {}).get("search", [])
            if not hits:
                return f"[wiki_summary: no article found for '{topic}']"
            slug = hits[0]["title"].replace(" ", "_")
            resp = _req.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
                headers=_HDRS, timeout=10,
            )
        resp.raise_for_status()
        extract = resp.json().get("extract", "")
        return extract[:1500] or "[wiki_summary: empty extract]"
    except Exception as e:
        return f"[wiki_summary error: {e}]"


# ── Tier 1: free, no key, zero latency or public REST ──────────────────────────

def sympy_solve(expression: str) -> str:
    """Solve or simplify a math/algebra expression using SymPy (fully local, no network)."""
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application,
        )
        from sympy import solve, simplify, symbols

        _TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
        expr = expression.strip()

        if "=" in expr:
            lhs_s, rhs_s = expr.split("=", 1)
            lhs = parse_expr(lhs_s.strip(), transformations=_TRANSFORMS)
            rhs = parse_expr(rhs_s.strip(), transformations=_TRANSFORMS)
            diff = lhs - rhs
            free = sorted(diff.free_symbols, key=str)
            if not free:
                return f"Statement is {'true' if diff == 0 else 'false'}"
            sol = solve(diff, free[0])
            return f"{free[0]} = {sol}"
        else:
            result = simplify(parse_expr(expr, transformations=_TRANSFORMS))
            return str(result)
    except ImportError:
        return "[sympy_solve: install sympy - pip install sympy]"
    except Exception as e:
        return f"[sympy_solve error: {e}]"


def pint_convert(value: str, from_unit: str, to_unit: str) -> str:
    """Convert between 700+ physical units (fully local via pint, zero latency)."""
    try:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        qty = float(value) * ureg(from_unit)
        converted = qty.to(to_unit)
        return f"{value} {from_unit} = {converted:.6g}"
    except ImportError:
        return "[pint_convert: install pint - pip install pint]"
    except Exception as e:
        return f"[pint_convert error: {e}]"


def openalex_search(query: str) -> str:
    """Search 250M academic papers via OpenAlex (no auth, no rate limit)."""
    try:
        import requests as _req
        resp = _req.get(
            "https://api.openalex.org/works",
            params={
                "search": query, "per-page": 3,
                "select": "title,publication_year,cited_by_count,abstract_inverted_index",
            },
            headers={"User-Agent": "SwarmAgent/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return f"[openalex_search: no results for '{query}']"
        lines = []
        for work in results:
            title = work.get("title", "Unknown")
            year = work.get("publication_year", "?")
            cited = work.get("cited_by_count", 0)
            aii = work.get("abstract_inverted_index") or {}
            if aii:
                positions = [(pos, word) for word, pos_list in aii.items() for pos in pos_list]
                abstract = " ".join(w for _, w in sorted(positions))[:400]
            else:
                abstract = "(no abstract)"
            lines.append(f"• [{year}] {title} (cited {cited}×)\n  {abstract}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"[openalex_search error: {e}]"


def semantic_scholar_search(query: str) -> str:
    """Search 200M+ papers via Semantic Scholar (no key needed, 5 000 req/day free)."""
    try:
        import requests as _req
        resp = _req.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "fields": "title,abstract,year,citationCount,tldr", "limit": 3},
            headers={"User-Agent": "SwarmAgent/1.0"},
            timeout=12,
        )
        resp.raise_for_status()
        results = resp.json().get("data", [])
        if not results:
            return f"[semantic_scholar: no results for '{query}']"
        lines = []
        for paper in results:
            title = paper.get("title", "Unknown")
            year = paper.get("year", "?")
            cited = paper.get("citationCount", 0)
            tldr = (paper.get("tldr") or {}).get("text", "")
            abstract = paper.get("abstract", "")
            summary = tldr or (abstract[:300] if abstract else "(no abstract)")
            lines.append(f"• [{year}] {title} (cited {cited}×)\n  {summary}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"[semantic_scholar_search error: {e}]"


def wikidata_entity(entity: str) -> str:
    """Look up structured facts about an entity from Wikidata (no key needed)."""
    try:
        import requests as _req
        _HDRS = {"User-Agent": "SwarmAgent/1.0"}
        search = _req.get(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbsearchentities", "search": entity,
                    "language": "en", "format": "json", "limit": 1, "type": "item"},
            headers=_HDRS, timeout=10,
        )
        search.raise_for_status()
        hits = search.json().get("search", [])
        if not hits:
            return f"[wikidata_entity: '{entity}' not found]"
        h = hits[0]
        qid   = h["id"]
        label = h.get("label", entity)
        desc  = h.get("description", "")
        aliases = [a["value"] for a in h.get("aliases", [])][:3]
        out = f"{label} ({qid}): {desc}"
        if aliases:
            out += f"\nAlso known as: {', '.join(aliases)}"
        return out
    except Exception as e:
        return f"[wikidata_entity error: {e}]"


def frankfurter_fx(base: str = "USD", target: str = "EUR") -> str:
    """Get ECB exchange rate (Frankfurter API, no key, updated daily)."""
    try:
        import requests as _req
        resp = _req.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": base.upper(), "symbols": target.upper()},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = data["rates"].get(target.upper(), "N/A")
        return f"1 {base.upper()} = {rate} {target.upper()} (ECB, {data.get('date', '?')})"
    except Exception as e:
        return f"[frankfurter_fx error: {e}]"


# ── Tier 2: free tier with optional API key - gracefully no-ops without one ────

def tavily_search(query: str) -> str:
    """AI-native web search via Tavily (set TAVILY_API_KEY env var - 1 000 free/month)."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        # Transparent fallback so the tool phase still produces evidence
        return web_search_tool(query)
    try:
        from tavily import TavilyClient
        resp = TavilyClient(api_key=api_key).search(
            query, search_depth="advanced", max_results=4,
        )
        lines = []
        if resp.get("answer"):
            lines.append(f"Direct answer: {resp['answer']}")
        for r in resp.get("results", [])[:3]:
            lines.append(f"• {r.get('title', '')} - {r.get('content', '')[:300]}")
        return "\n".join(lines) or "[tavily_search: no results]"
    except ImportError:
        return web_search_tool(query)   # graceful fallback
    except Exception as e:
        return f"[tavily_search error: {e}]"


def wolfram_query(query: str) -> str:
    """Computational knowledge via Wolfram Alpha LLM API - returns agent-ready text.
    Uses the /llm-api endpoint: designed for LLM consumption, clean prose output,
    no XML/image parsing needed. Set WOLFRAM_APP_ID env var (free at developer.wolframalpha.com).
    """
    app_id = os.environ.get("WOLFRAM_APP_ID", "")
    if not app_id:
        return "[wolfram_query: set WOLFRAM_APP_ID in .env - free key at developer.wolframalpha.com]"
    try:
        import requests as _req
        resp = _req.get(
            "https://www.wolframalpha.com/api/v1/llm-api",
            params={"input": query, "appid": app_id, "maxchars": 1000},
            timeout=12,
        )
        if resp.status_code == 501:
            return f"[wolfram_query: query not understood - '{query}']"
        resp.raise_for_status()
        return resp.text.strip()[:1200]
    except Exception as e:
        return f"[wolfram_query error: {e}]"


# Tool registry

TOOL_MAP = {
    # ── Knowledge graph ─────────────────────────────────────────────────────────
    "search_knowledge_graph":  search_knowledge_graph,
    "get_agent_opinions":      get_agent_opinions,
    "get_entity_relationships": get_entity_relationships,
    "run_neo4j_query":         run_neo4j_query,
    "write_report":            write_report,
    # ── Web search ──────────────────────────────────────────────────────────────
    "web_search":              web_search_tool,
    "web_search_and_fetch":    web_search_and_fetch,
    "fetch":                   fetch_url,
    "fetch_url":               fetch_url,
    "tavily_search":           tavily_search,           # AI-native, falls back to DDG
    # ── Structured knowledge ────────────────────────────────────────────────────
    "wiki_summary":            wiki_summary,            # Wikipedia prose
    "wikidata_entity":         wikidata_entity,         # Wikidata structured facts
    "openalex_search":         openalex_search,         # 250M academic papers
    "semantic_scholar_search": semantic_scholar_search, # 200M papers + AI TLDRs
    # ── Math / computation ──────────────────────────────────────────────────────
    "python_eval":             python_eval,             # safe arithmetic
    "sympy_solve":             sympy_solve,             # symbolic algebra (local)
    "pint_convert":            pint_convert,            # 700+ unit conversions (local)
    "wolfram_query":           wolfram_query,           # Wolfram Alpha (key optional)
    "frankfurter_fx":          frankfurter_fx,          # ECB exchange rates (no key)
    # ── Vision ──────────────────────────────────────────────────────────────────
    "camera_snapshot":         camera_snapshot,
    # ── Filesystem / git (open-ended / dev topics) ──────────────────────────────
    "read_file":               read_file,
    "list_directory":          list_directory,
    "git_log":                 git_log,
    "git_diff":                git_diff,
    "get_current_time":        get_current_time,
    "time":                    get_current_time,
}


def dispatch_tool(tool_name: str, **kwargs) -> str:
    fn = TOOL_MAP.get(tool_name)
    if not fn:
        return f"Unknown tool: {tool_name}"
    return fn(**kwargs)
