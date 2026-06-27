"""Progressive Tooled Debate - parallel execution, early exit, parallel tool fetch."""


import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, TypedDict

import numpy as np

from langgraph.graph import END, StateGraph

import agents as ag
import knowledge_graph as kg
import tools
from memory import MEMORY, AGENT_MEMORY

# Constants

CONVERGENCE_THRESHOLD = 0.68   # semantic cosine similarity threshold
MAX_PIVOTS = 2

# SSE streaming - set by api.run_simulation, cleared on exit.
# Groups finished in speak_round_node push messages here so the
# SSE endpoint can stream them live without polling the final state.
_ACTIVE_STREAM: list | None = None


def _set_stream(sink: list | None) -> None:
    global _ACTIVE_STREAM
    _ACTIVE_STREAM = sink

# GroupDebate - 3 groups debate internally, then cross-pollinate
DEBATE_GROUPS: list[list[str]] = [
    ["Skeptic", "Contrarian"],           # Challenge group: probes + challenges consensus
    ["Visionary", "Technologist"],        # Build group: futures + implementation
    ["Realist", "Economist", "Ethicist"], # Ground group: constraints + incentives + ethics
]
# DyTopo: which group pairs share summaries each round (round % 3 selects the pair)
_CROSS_PAIRS = [(0, 1), (1, 2), (0, 2)]
# DySCo dynamic sparse communication (replaces fixed single-pair rotation + summary trim).
_DYSCO_BUDGET = 2        # max group->group summary edges kept per round
_DYSCO_MIN_VALUE = 0.05  # drop edges below this value (groups already aligned, fewer tokens needed)
_PAIR_OF: dict = {frozenset(p): i for i, p in enumerate(_CROSS_PAIRS)}

# SELENE: skip agent if claim overlaps too much with their previous claim
NOVELTY_THRESHOLD = 0.78   # cosine similarity - above this = near-paraphrase / repetitive
MAX_SILENT_ROUNDS = 1      # consecutive rounds an agent can be silenced before forced back in
_HIGH_VAR_THRESHOLD = 0.10  # pairwise sim variance above this = scattered reasoning (Level 27)

# Per-agent tool list - matched to each agent's role and the benchmark types they face.
#
# Reasoning group (Qwen3:8b):
#   Skeptic    - evidence scrutiniser:  deep web + structured academic knowledge
#   Technologist - computation verifier: web + symbolic math + unit conversion
#   Synthesizer  - oracle, no tool phase
#
# Diversity group (Llama3.1:8b):
#   Realist    - ground-truth verifier: web + structured facts + arithmetic
#   Economist  - quantitative:          web + symbolic math + finance/FX
#   Ethicist   - source validator:      web + academic papers
#
# Vision group (Gemma E2B):
#   Visionary  - broad knowledge seeker: web + academic search + wiki
#   Contrarian - devil's advocate:       web + structured facts + KG recall
_WORKER_TOOLS: dict[str, list[str]] = {
    "Skeptic":      ["web_search_and_fetch", "wiki_summary", "openalex_search",
                     "wikidata_entity", "search_knowledge_graph"],
    "Technologist": ["web_search_and_fetch", "sympy_solve", "python_eval", "pint_convert"],
    "Realist":      ["web_search", "wiki_summary", "wikidata_entity",
                     "sympy_solve", "python_eval"],
    "Economist":    ["web_search", "sympy_solve", "python_eval",
                     "frankfurter_fx", "pint_convert"],
    "Ethicist":     ["web_search_and_fetch", "wiki_summary", "openalex_search",
                     "semantic_scholar_search", "search_knowledge_graph"],
    "Visionary":    ["web_search_and_fetch", "wiki_summary", "semantic_scholar_search",
                     "openalex_search"],
    "Contrarian":   ["web_search", "wiki_summary", "wikidata_entity",
                     "search_knowledge_graph"],
}

# All agents that fetch evidence during tool_phase
TOOL_AGENTS = list(_WORKER_TOOLS.keys())

# Maps each agent to a model lane: reasoning (Qwen3), diversity (Llama3.1), vision (Gemma E2B).
_AGENT_LANE: dict[str, str] = {
    "Skeptic":      "reasoning",
    "Technologist": "reasoning",
    "Synthesizer":  "reasoning",
    "Realist":      "diversity",
    "Economist":    "diversity",
    "Ethicist":     "diversity",
    "Visionary":    "vision",
    "Contrarian":   "vision",
}


def _lane_for(agent_name: str) -> str:
    return _AGENT_LANE.get(agent_name, "reasoning")


# State

class SimState(TypedDict):
    topic: str
    entities: list[str]
    round: int
    max_rounds: int
    phase: str                              # "debate" | "evidence" | "done"
    messages: Annotated[list[dict], operator.add]
    evidence_pool: dict
    convergence_score: float
    pivot_count: int
    image_b64: str
    verdict: str
    report_path: str
    # GroupDebate
    group_summaries: dict                   # {group_key: summary_text}
    # SELENE - novelty gating
    agent_last_claim: dict                  # {agent_name: last claim text}
    agent_silent_count: dict                # {agent_name: consecutive silent rounds}
    # Dynamic agent selection + DyTopo reputation tracking
    agent_filter: list                      # debate-agent subset; empty = all 7
    last_pair_idx: int                      # DyTopo pair used in last round
    # MoA layered refinement
    layer_context: str                      # previous round's claims for inter-layer injection
    # Anti-sycophancy
    forced_challenge_done: bool             # True after one forced evidence round
    # DRIFTJudge
    drift_correction_done: bool             # True after one drift re-anchor injection
    # FREE-MAD - consensus-free critique mode
    free_mad_mode: bool                     # True when topic has contested stored verdict
    free_mad_scores: dict                   # {agent_name: score} for score-based winner
    # Belief-Update Calibration - detect overconfidence / false convergence
    agent_confidence_history: dict          # {agent_name: [conf_round1, conf_round2, ...]}
    overconfidence_corrected: bool          # True after one overconfidence correction fires
    # DCI Epistemic Acts - typed deliberation acts
    unresolved_challenges: int              # CHALLENGE acts not yet matched by GROUND/BRIDGE
    # Single-answer mode: "open" (discursive verdict) | "numeric" | "choice"
    answer_mode: str


# Helpers

def _compute_convergence(messages: list[dict]) -> float:
    """Semantic similarity between recent messages via embeddings (replaces Jaccard)."""
    return MEMORY.convergence_score(messages)


def _detect_drift(topic: str, messages: list[dict]) -> bool:
    """
    DRIFTJudge: detect if recent debate content has drifted
    away from the original topic. Returns True if drift is detected.
    Uses last 6 non-error messages to minimise token cost.
    """
    recent = [m["content"][:200] for m in messages[-6:] if not m.get("error")]
    if len(recent) < 3:
        return False
    combined = "\n".join(recent)
    raw = ag.call_llm(
        "You detect topic drift in multi-agent debates.",
        f"Original topic: {topic}\n\nRecent agent statements:\n{combined}\n\n"
        f"Are agents still substantively discussing the original topic, "
        f"or have they drifted to tangential subjects?\n"
        f"Reply with ONLY: ON_TOPIC or DRIFTED",
        max_tokens=15, enable_thinking=False, temperature=0.1,
    ).strip().upper()
    return "DRIFTED" in raw


def _extract_claim(text: str) -> str:
    """Pull the CLAIM line out of a structured agent response."""
    for line in text.split("\n"):
        if line.upper().startswith("CLAIM:"):
            return line[6:].strip()
    return ""


def _reflect_one(name: str, topic: str) -> tuple[str, str]:
    """Reflect on past positions before speaking. Returns (name, reflection)."""
    past = AGENT_MEMORY.get_relevant(name, topic, top_k=3)
    principles = AGENT_MEMORY.get_reflections(name, topic, rtype="principle", top_k=2)
    procedures = AGENT_MEMORY.get_reflections(name, topic, rtype="procedure", top_k=1)

    if not past and not principles and not procedures:
        return name, ""

    agent = ag.get_agent(name)
    parts = []
    if past:
        parts.append("Past claims on related topics:\n" + "\n".join(f"- {p}" for p in past))
    if principles:
        parts.append("Principles learned from past debates:\n" + "\n".join(f"- {p}" for p in principles))
    if procedures:
        parts.append("Procedural adjustments for this domain:\n" + "\n".join(f"- {p}" for p in procedures))

    memory_txt = "\n\n".join(parts)
    reflection = ag.call_llm(
        agent["personality"],
        f"{memory_txt}\n\n"
        f"New debate topic: {topic}\n\n"
        f"In one sentence: what do you refine, double down on, or apply from the above? Be specific.",
        max_tokens=80,
        enable_thinking=False,
    )
    if reflection.startswith("[") and len(reflection) < 300:
        return name, ""
    return name, reflection


# Single-agent speak (thread-safe)

def _speak_one(
    name: str,
    state: SimState,
    entity_ctx: str,
    conversation: str,
    evidence_txt: str,
    reflection: str = "",
    layer_context: str = "",
    anti_conformity_ctx: str = "",  # FREE-MAD: injected after standard prompt
) -> dict:
    """One agent generates its response. Safe to call from multiple threads."""
    agent = ag.get_agent(name)
    history = kg.get_agent_history(name)
    history_txt = "\n".join(history[-3:]) if history else "No prior statements."

    reflection_block = (
        f"\nYour self-reflection from past debates:\n{reflection}\n"
        if reflection else ""
    )
    # MoA: inject previous round's claims as refinement context
    layer_block = (
        f"\nPositions from the previous round - refine, challenge, or build on these:\n{layer_context}\n"
        if layer_context else ""
    )

    user_prompt = (
        f"Topic under debate: {state['topic']}\n\n"
        f"Knowledge graph context:\n{entity_ctx}\n\n"
        f"Your past positions in this debate:\n{history_txt}"
        f"{reflection_block}"
        f"{layer_block}\n"
        f"Current debate (most recent):\n{conversation}"
        f"{evidence_txt}\n\n"
        f"Round {state['round']} of {state['max_rounds']} | Phase: {state['phase'].upper()}.\n\n"
        f"Reply AS YOURSELF in this exact structure - one sentence each, no extras:\n"
        f"CLAIM: [your main point - a specific, arguable position]\n"
        f"EVIDENCE: [one fact, data point, or observation that supports your claim]\n"
        f"REBUTTAL: [pre-empt the strongest objection someone could raise against you]\n"
        f"CONFIDENCE: [integer 0-100 - how certain are you of your claim right now]\n"
        f"EPISTEMIC_ACT: [choose ONE: PROPOSE (new claim) | CHALLENGE (dispute another's specific claim) | BRIDGE (find common ground between two views) | GROUND (add concrete evidence for an existing claim) | SYNTHESIZE (integrate multiple positions)]"
    )
    if anti_conformity_ctx:
        user_prompt += anti_conformity_ctx

    # Single-answer mode: require a parseable final answer (so oracle_node can vote on it)
    # and give the model genuine chain-of-thought + a larger budget - multi-step answers
    # (esp. arithmetic) collapse without it. "open" mode keeps the original fast,
    # thinking-suppressed debate behaviour untouched.
    answer_mode = state.get("answer_mode", "open")
    speak_tokens = 680
    speak_thinking = False
    if answer_mode == "numeric":
        user_prompt += ("\n\nWork the problem step by step, then end with a line "
                        "'FINAL: <the single numeric answer, digits only>'.")
        speak_tokens, speak_thinking = 1536, True
    elif answer_mode == "choice":
        user_prompt += ("\n\nReason briefly, then end with a line "
                        "'FINAL: <the single correct option letter>'.")
        speak_tokens, speak_thinking = 900, True

    # Agents see peer CONFIDENCE scores in conversation; instruct them to weight accordingly.
    # Only injected for single-answer tasks - open debate intentionally stays flat.
    if answer_mode in ("numeric", "choice") and conversation != "(group debate just started)":
        user_prompt += (
            "\n\nWhen evaluating the peer claims above: treat CONFIDENCE ≥70 as strong "
            "signal, CONFIDENCE <40 as weak - weight your response accordingly."
        )

    lane = _lane_for(name)
    response = ag.call_llm(agent["personality"], user_prompt,
                           max_tokens=min(speak_tokens, 300),
                           temperature=agent.get("temperature", 0.8),
                           enable_thinking=speak_thinking and lane != "vision",
                           lane=lane)

    # Detect LLM error responses - call_llm returns "[ExceptionStr]" on failure.
    # Don't store error strings in the KG or agent memory as real opinions.
    is_error = (
        response.startswith("[") and "CLAIM:" not in response and len(response) < 300
    )
    if not is_error:
        fallback_entity = kg.topic_node_id(state["topic"])
        for entity in (state["entities"] if state["entities"] else [fallback_entity]):
            kg.add_agent_opinion(name, entity, response, state["round"])

    # Belief-Update Calibration: extract self-reported confidence
    import re as _re_conf
    _conf_m = _re_conf.search(r'CONFIDENCE:\s*(\d+)', response)
    confidence = int(_conf_m.group(1)) if _conf_m else 50

    return {"agent": name, "content": response, "round": state["round"],
            "error": is_error, "confidence": confidence}


# SELENE novelty check

def _claim_novelty(new_claim: str, old_claim: str) -> float:
    """
    Cosine similarity between claim embeddings (same encoder as convergence_score).
    Low = novel contribution, high = repetitive - SELENE uses this to gate agents.
    """
    if not old_claim or not new_claim:
        return 0.0
    try:
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            vecs = enc.encode([new_claim, old_claim],
                              convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))
    except Exception:
        return 0.0


# GroupDebate internal-phase helper

def _group_speak_phase(
    group_names: list[str],
    active_names: list[str],          # SELENE-filtered subset
    state: SimState,
    entity_ctx: str,
    evidence_txt: str,
    reflections: dict[str, str],
    cross_summary: str,               # DyTopo: summary from the paired group
    layer_context: str = "",          # MoA: accumulated claims from the previous round
    anti_conformity_ctx: str = "",    # FREE-MAD: anti-conformity block for each agent
) -> tuple[list[dict], str]:
    """
    Intra-group debate: agents see each other + a cross-group summary.
    Returns (messages, group_summary).
    """
    if not active_names:
        active_names = group_names     # SELENE fallback: never fully silence a group

    group_conv = "(group debate just started)"
    if cross_summary:
        group_conv = f"External group's position: {cross_summary}\n\n(your group responds)"

    group_msgs: list[dict] = []
    for peer_idx, name in enumerate(active_names):
        msg = _speak_one(name, state, entity_ctx, group_conv, evidence_txt,
                         reflections.get(name, ""), layer_context, anti_conformity_ctx)
        group_msgs.append(msg)
        # Don't inject LLM error strings into the conversation - skip silently.
        # Show "Peer N" instead of role names to prevent identity-driven sycophancy.
        # Real names still tracked in KG, evidence pool, and vote.
        if not msg.get("error"):
            claim = _extract_claim(msg["content"]) or msg["content"][:150]
            group_conv += f"\n[Peer {peer_idx + 1}]: {claim}"

    # Summarize this group's position for cross-pollination - exclude error responses
    combined = "\n".join(
        f"{m['agent']}: {_extract_claim(m['content']) or m['content'][:200]}"
        for m in group_msgs
        if not m.get("error")
    )
    if not combined:
        return group_msgs, ""  # all agents errored - skip summary LLM call
    summary = ag.call_llm(
        "You summarize debate positions concisely.",
        f"Group debate:\n{combined}\n\n"
        "What is this group's core position and main internal disagreement? 2 sentences max.",
        max_tokens=200, temperature=0.4, enable_thinking=False,
    )
    return group_msgs, summary


# MARS meta-cognitive reflection

def _mars_reflect_background(verdict_text: str, messages: list[dict], topic: str) -> None:
    """Generate typed memory entries (PRINCIPLE + PROCEDURE) per agent in a background thread."""
    import threading

    def _run():
        # Count participation - only reflect for agents who actively contributed
        participation: dict[str, int] = {}
        for m in messages:
            if not m.get("error") and m["agent"] not in ("Synthesizer", "Contrarian", "SYSTEM"):
                participation[m["agent"]] = participation.get(m["agent"], 0) + 1

        active_agents = [a for a, cnt in participation.items() if cnt >= 2][:4]  # max 4
        if not active_agents:
            return

        verdict_snippet = verdict_text[:400]

        for name in active_agents:
            agent_msgs = [
                m["content"][:200] for m in messages
                if m["agent"] == name and not m.get("error")
            ]
            agent_debate = "\n".join(agent_msgs[-3:])

            try:
                # PRINCIPLE: generalizable lesson
                principle = ag.call_llm(
                    ag.get_agent(name)["personality"],
                    f"Topic debated: {topic}\n\n"
                    f"Your contributions:\n{agent_debate}\n\n"
                    f"Final verdict:\n{verdict_snippet}\n\n"
                    f"In ONE sentence: what generalizable principle does this debate reveal "
                    f"that you should remember across future topics? "
                    f"Start with 'When' or 'Always' or 'Never'.",
                    max_tokens=80, enable_thinking=False, temperature=0.3,
                )
                AGENT_MEMORY.save_reflection(name, topic, "principle", principle)

                # PROCEDURE: specific adjustment for this domain
                procedure = ag.call_llm(
                    ag.get_agent(name)["personality"],
                    f"Topic debated: {topic}\n\n"
                    f"Your contributions:\n{agent_debate}\n\n"
                    f"Final verdict:\n{verdict_snippet}\n\n"
                    f"In ONE sentence: what specific adjustment should you make next time "
                    f"a question in this domain comes up? Be concrete - name the adjustment.",
                    max_tokens=80, enable_thinking=False, temperature=0.3,
                )
                AGENT_MEMORY.save_reflection(name, topic, "procedure", procedure)
                print(f"[MARS] {name} - principle + procedure stored")
            except Exception as exc:
                print(f"[MARS] {name} reflection error: {exc}")

    threading.Thread(target=_run, daemon=True).start()


# FREE-MAD helpers

_FREE_MAD_W = (20, 25, 30, 20)  # w1 initial, w2 shift-penalty, w3 shift-gain, w4 maintain


def _free_mad_suffix(other_agent_claims: list[str]) -> str:
    """Anti-conformity prompt block injected per agent. Majority opinion is explicitly forbidden as evidence."""
    others = "\n".join(f"- {c}" for c in other_agent_claims) if other_agent_claims else "(none yet)"
    return (
        f"\n\nOther agents' current positions:\n{others}\n\n"
        f"MANDATORY 5-STEP CRITIQUE (do not skip steps):\n"
        f"STEP 1 - Initial position: restate your core claim and reasoning chain.\n"
        f"STEP 2 - Analyse others: for each agent above, identify one concrete error or valid point.\n"
        f"STEP 3 - Self-examination: do those same errors exist in your own reasoning?\n"
        f"STEP 4 - Decision: RETAIN or REVISE your position, with a one-sentence justification.\n"
        f"STEP 5 - Final CLAIM line: must start with 'CLAIM:'\n\n"
        f"CRITICAL: You may NOT use majority opinion as evidence. "
        f"If you cannot definitively prove others are correct, RETAIN your own conclusion."
    )


def _update_free_mad_scores(
    scores: dict, messages: list[dict], last_claims: dict, round_num: int
) -> dict:
    """Score each agent's round contribution. Rewards initial positions and maintained stances; penalises opinion shifts."""
    w1, w2, w3, w4 = _FREE_MAD_W
    f = 1.0 / (round_num + 1)
    new_scores = dict(scores)
    for msg in messages:
        if msg.get("error"):
            continue
        name = msg["agent"]
        claim = _extract_claim(msg["content"])
        if not claim:
            continue
        prev = last_claims.get(name, "")
        if round_num == 1 or not prev:
            new_scores[name] = new_scores.get(name, 0.0) + w1 * f
        elif _claim_novelty(claim, prev) < 0.5:
            # Opinion shift detected - reward new position, penalise old
            new_scores[name] = new_scores.get(name, 0.0) + w3 * f
            old_key = f"{name}_prev"
            new_scores[old_key] = new_scores.get(old_key, 0.0) - w2 * f
        else:
            # Maintained position
            new_scores[name] = new_scores.get(name, 0.0) + w4 * f
    return new_scores


# DART: Disagreement-Triggered Tool Recruitment

def _dart_dispute_search(messages: list[dict], topic: str) -> dict:
    """Find the most-opposed claim pair and fire a targeted web search on the dispute. No LLM calls."""
    recent_claims: list[tuple[str, str]] = []
    for m in messages[-12:]:
        if m.get("error") or m["agent"] in ("Synthesizer", "Contrarian", "SYSTEM"):
            continue
        claim = _extract_claim(m["content"])
        if claim and len(claim.split()) >= 5:
            recent_claims.append((m["agent"], claim))

    if len(recent_claims) < 2:
        return {}

    try:
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            vecs = enc.encode([c for _, c in recent_claims],
                              convert_to_numpy=True, normalize_embeddings=True)
        n = len(vecs)
        max_dist, best_i, best_j = 0.0, 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                dist = 1.0 - float(np.dot(vecs[i], vecs[j]))
                if dist > max_dist:
                    max_dist, best_i, best_j = dist, i, j

        if max_dist < 0.70:
            return {}

        a1, c1 = recent_claims[best_i]
        a2, c2 = recent_claims[best_j]
        print(f"[DART] Sharp dispute dist={max_dist:.2f}: {a1} vs {a2} - recruiting tool")
        result = tools.web_search_tool(f"{c1[:80]} versus {c2[:80]}")
        return {f"DART_{a1}_vs_{a2}": f"[DART: {a1} vs {a2} dispute resolution]\n{result[:400]}"}
    except Exception as exc:
        print(f"[DART] Error: {exc}")
        return {}


# EGSR: Per-Round Evidence Re-Anchoring

def _egsr_anchor(agent_name: str, evidence_pool: dict, last_claim: str) -> str:
    """Re-surface the most relevant existing evidence for an agent's current claim. No LLM calls."""
    if not evidence_pool or not last_claim:
        return ""
    items = [
        (k, v) for k, v in evidence_pool.items()
        if v and not v.startswith("[") and len(v.strip()) > 20
    ]
    if not items:
        return ""
    try:
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            claim_vec = enc.encode(last_claim, convert_to_numpy=True, normalize_embeddings=True)
            ev_texts = [v[:200] for _, v in items]
            ev_vecs = enc.encode(ev_texts, convert_to_numpy=True, normalize_embeddings=True)
        sims = np.dot(ev_vecs, claim_vec)
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) < 0.28:
            return ""  # not relevant enough to inject
        snippet = ev_texts[best_idx][:120].strip()
        return f"[EGSR re-anchor - most relevant evidence for your position: {snippet}]"
    except Exception:
        return ""


# DySCo: Dynamic Trust-Aware Sparse Communication

def _dysco_select_edges(prev_summaries: dict, topic: str,
                        round_num: int) -> tuple[dict, int]:
    """Score each group-to-group communication edge by trust * divergence * relevance. Keep only the top edges. No LLM calls."""
    from reputation import get_agent_trust, best_pair_idx

    rotation_fallback = best_pair_idx(round_num)
    have = [(int(k), v) for k, v in prev_summaries.items() if v]
    if not have:
        return {}, rotation_fallback   # round 1 / cold start: groups stay independent

    n_groups = len(DEBATE_GROUPS)
    try:
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            vecs = enc.encode([v for _, v in have] + [topic],
                              convert_to_numpy=True, normalize_embeddings=True)
    except Exception:
        return {}, rotation_fallback

    topic_vec = vecs[-1]
    sum_vec = {gi: vecs[i] for i, (gi, _) in enumerate(have)}
    sum_txt = {gi: v for gi, v in have}

    def _gtrust(gidx: int) -> float:
        members = DEBATE_GROUPS[gidx]
        return sum(get_agent_trust(a) for a in members) / max(1, len(members))

    edges: list = []   # (value, sender, receiver)
    for s, s_vec in sum_vec.items():
        relevance = max(0.0, float(np.dot(s_vec, topic_vec)))
        trust = _gtrust(s)
        for r in range(n_groups):
            if r == s:
                continue
            divergence = (1.0 - float(np.dot(s_vec, sum_vec[r]))) if r in sum_vec else 1.0
            value = trust * relevance * divergence
            if value >= _DYSCO_MIN_VALUE:
                edges.append((value, s, r))

    if not edges:
        return {}, rotation_fallback

    edges.sort(key=lambda e: -e[0])
    selected = edges[:max(1, _DYSCO_BUDGET)]

    cross: dict = {}
    for value, s, r in selected:
        if r not in cross:                 # keep highest-value incoming edge per receiver
            cross[r] = sum_txt[s]

    _, top_s, top_r = selected[0]
    dominant_pair_idx = _PAIR_OF.get(frozenset({top_s, top_r}), rotation_fallback)
    print(f"[DySCo] kept {len(selected)}/{len(edges)} edge(s): "
          + ", ".join(f"{s}->{r}={v:.2f}" for v, s, r in selected))
    return cross, dominant_pair_idx


# Nodes

def setup_node(state: SimState) -> dict:
    kg.seed_topic(state["topic"], state["entities"])
    for agent in ag.SWARM:
        kg.add_entity(agent["name"], "agent", style=agent["style"])
    return {
        "round": 1,
        "phase": "debate",
        "evidence_pool": {},
        "convergence_score": 0.0,
        "pivot_count": 0,
        "group_summaries": {},
        "agent_last_claim": {},
        "agent_silent_count": {},
        "last_pair_idx": 0,
        "layer_context": "",
        "forced_challenge_done": False,
        "drift_correction_done": False,
        "free_mad_mode": False,
        "free_mad_scores": {},
        "agent_confidence_history": {},
        "overconfidence_corrected": False,
        "unresolved_challenges": 0,
    }


def speak_round_node(state: SimState) -> dict:
    """Run one debate round: groups speak in parallel, then cross-pollinate via summaries."""
    round_num = state["round"]
    phase = state["phase"]
    topic = state["topic"]

    # Resource guard: hard-stop before a memory-starved host OOM-crashes mid-round.
    import resource_guard
    resource_guard.assert_safe(f"speak_round r{round_num}")

    entity_ctx = kg.get_entity_context(state["entities"])
    evidence_txt = ""
    if phase == "evidence" and state["evidence_pool"]:
        lines = [f"  {a}: {v}" for a, v in state["evidence_pool"].items()]
        evidence_txt = "\n\nShared evidence gathered by all agents:\n" + "\n".join(lines)

    # DART: if agents sharply disagreed last round (cosine dist > 0.70),
    # fire a targeted web search on the disputed sub-topic BEFORE this round starts so
    # agents can respond to the dispute resolution evidence immediately.
    dart_additions: dict = {}
    if state.get("messages"):
        dart_additions = _dart_dispute_search(state["messages"], topic)
    if dart_additions:
        dart_lines = "\n".join(f"  {k}: {v}" for k, v in dart_additions.items())
        evidence_txt += f"\n\n[DART - targeted dispute resolution]:\n{dart_lines}"
        print(f"[DART] {len(dart_additions)} dispute resolution(s) injected into round {round_num}")

    # Self-reflection (sequential for low-end hardware)
    all_worker_names = [n for g in DEBATE_GROUPS for n in g]
    reflections: dict[str, str] = {}
    if phase == "debate":
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut_map = {pool.submit(_reflect_one, n, state["topic"]): n
                       for n in all_worker_names}
            for fut in as_completed(fut_map):
                try:
                    name, ref = fut.result()
                    if ref:
                        reflections[name] = ref
                except Exception:
                    pass

    # Re-anchor each agent with the most relevant existing evidence before they speak.
    if state.get("evidence_pool"):
        _last_cls = state.get("agent_last_claim") or {}
        for _ename in all_worker_names:
            _anchor = _egsr_anchor(_ename, state["evidence_pool"], _last_cls.get(_ename, ""))
            if _anchor:
                reflections[_ename] = (
                    _anchor + ("\n" + reflections[_ename] if _ename in reflections else "")
                )

    # SELENE + dynamic filter
    last_claims = state.get("agent_last_claim") or {}
    silent_counts = state.get("agent_silent_count") or {}
    # Dynamic Role Assignment: query-specific agent subset
    filter_set = set(state.get("agent_filter") or [])

    def _active_for_group(group: list[str]) -> list[str]:
        active = []
        for name in group:
            if filter_set and name not in filter_set:
                continue                      # excluded for this query
            if silent_counts.get(name, 0) >= MAX_SILENT_ROUNDS:
                active.append(name)          # SELENE: forced back in
            elif silent_counts.get(name, 0) > 0:
                pass                          # SELENE: still silenced
            else:
                active.append(name)
        eligible = [n for n in group if not filter_set or n in filter_set]
        return active or eligible or group    # never fully empty a group

    # Select which group summaries to share this round based on trust and divergence.
    prev_summaries = state.get("group_summaries") or {}
    cross_inputs, pair_idx = _dysco_select_edges(prev_summaries, state["topic"], round_num)

    # MoA: pass previous round's accumulated claims into each group
    incoming_layer_ctx = state.get("layer_context", "")

    # FREE-MAD: build anti-conformity context from previous round's claims
    free_mad_mode = state.get("free_mad_mode", False)
    anti_conformity_ctx = ""
    if free_mad_mode:
        prior_claims = [
            f"{m['agent']}: {_extract_claim(m['content'])}"
            for m in state["messages"][-12:]
            if not m.get("error") and _extract_claim(m["content"])
        ]
        anti_conformity_ctx = _free_mad_suffix(prior_claims)
        print(f"[FREE-MAD] Round {round_num} - anti-conformity mode active, "
              f"{len(prior_claims)} prior claims injected")

    # Run 3 groups sequentially (hardware-safe)
    round_messages: list[dict] = []
    new_group_summaries: dict[str, str] = {}

    # Groups run concurrently now - different groups mostly hit different models
    # (Qwen/Llama/Gemma), so they overlap. Per-lane semaphores in call_llm keep any
    # single model from being hit by two calls at once.
    with ThreadPoolExecutor(max_workers=len(DEBATE_GROUPS)) as pool:
        futures = {
            pool.submit(
                _group_speak_phase,
                DEBATE_GROUPS[g_idx],
                _active_for_group(DEBATE_GROUPS[g_idx]),
                state, entity_ctx, evidence_txt, reflections,
                cross_inputs.get(g_idx, ""),
                incoming_layer_ctx,
                anti_conformity_ctx,  # FREE-MAD
            ): g_idx
            for g_idx in range(len(DEBATE_GROUPS))
        }
        for fut in as_completed(futures):
            g_idx = futures[fut]
            try:
                msgs, summary = fut.result()
                round_messages.extend(msgs)
                new_group_summaries[str(g_idx)] = summary
                # SSE: push this group's messages as soon as they're ready
                if _ACTIVE_STREAM is not None:
                    _ACTIVE_STREAM.extend(msgs)
            except Exception as exc:
                print(f"[GroupDebate] Group {g_idx} error: {exc}")

    # SELENE: update novelty state for next round
    new_last_claims = dict(last_claims)
    new_silent = dict(silent_counts)

    for msg in round_messages:
        name = msg["agent"]
        claim = _extract_claim(msg["content"])
        if not claim:
            continue
        overlap = _claim_novelty(claim, new_last_claims.get(name, ""))
        new_last_claims[name] = claim
        if overlap > NOVELTY_THRESHOLD:
            new_silent[name] = new_silent.get(name, 0) + 1
            print(f"[SELENE] {name} silenced next round (overlap={overlap:.2f})")
        else:
            new_silent[name] = 0   # reset - agent is contributing novelty

    # Persist claims to agent journal
    for msg in round_messages:
        claim = _extract_claim(msg["content"])
        if claim:
            AGENT_MEMORY.save_position(msg["agent"], state["topic"], claim)

    # MoA: build layer_context for the NEXT round from this round's claims
    claim_lines = []
    for msg in round_messages:
        if not msg.get("error"):
            claim = _extract_claim(msg["content"]) or msg["content"][:120]
            claim_lines.append(f"- {msg['agent']}: {claim}")
    new_layer_ctx = "\n".join(claim_lines) if claim_lines else ""

    # FREE-MAD: update score table after each round
    updated_free_mad_scores = {}
    if free_mad_mode:
        updated_free_mad_scores = _update_free_mad_scores(
            state.get("free_mad_scores", {}), round_messages, last_claims, round_num
        )

    # Belief-Update Calibration: accumulate per-agent confidence history
    conf_history: dict[str, list] = dict(state.get("agent_confidence_history") or {})
    for msg in round_messages:
        if not msg.get("error") and "confidence" in msg:
            agent_name = msg["agent"]
            conf_history.setdefault(agent_name, []).append(msg["confidence"])

    # DCI: count typed epistemic acts to track unresolved CHALLENGEs.
    # CHALLENGE acts that lack a corresponding GROUND or BRIDGE response block convergence.
    import re as _re_dci
    _challenge_count = 0
    _response_count = 0
    for msg in round_messages:
        if msg.get("error"):
            continue
        act_m = _re_dci.search(r'EPISTEMIC_ACT:\s*(PROPOSE|CHALLENGE|BRIDGE|GROUND|SYNTHESIZE)',
                               msg["content"], _re_dci.IGNORECASE)
        if act_m:
            act = act_m.group(1).upper()
            if act == "CHALLENGE":
                _challenge_count += 1
            elif act in ("GROUND", "BRIDGE"):
                _response_count += 1
    new_unresolved = max(0, _challenge_count - _response_count)

    result = {
        "messages": round_messages,
        "round": round_num + 1,
        "group_summaries": new_group_summaries,
        "agent_last_claim": new_last_claims,
        "agent_silent_count": new_silent,
        "last_pair_idx": pair_idx,
        "layer_context": new_layer_ctx,
        "agent_confidence_history": conf_history,
        "unresolved_challenges": new_unresolved,
    }
    if updated_free_mad_scores:
        result["free_mad_scores"] = updated_free_mad_scores
    if dart_additions:
        merged_pool = dict(state.get("evidence_pool") or {})
        merged_pool.update(dart_additions)
        result["evidence_pool"] = merged_pool
    return result


def _disagreement_state(messages: list[dict]) -> tuple[str, float, float]:
    """Classify the debate into one of four epistemic states using mean + variance of pairwise similarity."""
    recent = [m["content"] for m in messages[-8:] if not m.get("error")]
    if len(recent) < 2:
        return "convergent_disagreement", 0.0, 0.0
    try:
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            vecs = enc.encode(recent, convert_to_numpy=True, normalize_embeddings=True)
        n = len(vecs)
        sims = [float(np.dot(vecs[i], vecs[j])) for i in range(n) for j in range(i + 1, n)]
        if not sims:
            return "convergent_disagreement", 0.0, 0.0
        mean_sim = sum(sims) / len(sims)
        var_sim = float(np.var(sims))
        high_sim = mean_sim >= CONVERGENCE_THRESHOLD
        high_var = var_sim >= _HIGH_VAR_THRESHOLD
        if high_sim and not high_var:
            return "convergent_agreement", mean_sim, var_sim
        if high_sim and high_var:
            return "divergent_agreement", mean_sim, var_sim
        if not high_sim and high_var:
            return "divergent_disagreement", mean_sim, var_sim
        return "convergent_disagreement", mean_sim, var_sim
    except Exception:
        return "convergent_disagreement", 0.0, 0.0


def convergence_check_node(state: SimState) -> dict:
    # FREE-MAD: in consensus-free mode, never exit early - run all
    # rounds and let the score-based winner in oracle_node decide the outcome.
    if state.get("free_mad_mode", False):
        return {"convergence_score": 0.0}

    try:
        score = _compute_convergence(state["messages"])
    except Exception as e:
        print(f"[ConvergenceCheck] Encoder failed, defaulting to 0.0: {e}")
        score = 0.0

    # Force one evidence round when agents converge before gathering external data.
    if score >= CONVERGENCE_THRESHOLD and not state.get("forced_challenge_done", False):
        print(f"[AntiSycophancy] Premature convergence (score={score:.3f}, "
              f"round={state['round']}) - forcing one evidence round")
        return {
            "convergence_score": CONVERGENCE_THRESHOLD - 0.01,
            "forced_challenge_done": True,
        }

    # Re-anchor and force one more round if convergence happened on a drifted topic.
    # Threshold is set above CONVERGENCE_THRESHOLD to avoid false positives.
    _DRIFT_TRIGGER = 0.85
    if (
        score >= _DRIFT_TRIGGER
        and state.get("forced_challenge_done", False)
        and not state.get("drift_correction_done", False)
    ):
        try:
            drifted = _detect_drift(state["topic"], state["messages"])
        except Exception as e:
            print(f"[DRIFTJudge] Check failed: {e}")
            drifted = False

        if drifted:
            print(f"[DRIFTJudge] Drift detected at round {state['round']} - re-anchoring")
            re_anchor = {
                "agent": "SYSTEM",
                "content": (
                    f"[DRIFT CORRECTION] The debate has lost focus. "
                    f"All agents must return to the original topic: \"{state['topic']}\". "
                    f"Ignore tangential threads and address the core question directly."
                ),
                "round": state["round"],
                "error": False,
            }
            return {
                "convergence_score": CONVERGENCE_THRESHOLD - 0.01,
                "drift_correction_done": True,
                "messages": [re_anchor],
            }

    # Force one more round when convergence looks like overconfidence rather than genuine agreement.
    if (
        score >= CONVERGENCE_THRESHOLD
        and state.get("forced_challenge_done", False)
        and state.get("drift_correction_done", False)
        and not state.get("overconfidence_corrected", False)
    ):
        conf_history = state.get("agent_confidence_history") or {}
        deltas = []
        for agent_conf_list in conf_history.values():
            if len(agent_conf_list) >= 2:
                deltas.append(abs(agent_conf_list[-1] - agent_conf_list[-2]))
        if deltas:
            median_delta = sorted(deltas)[len(deltas) // 2]
            if median_delta < 5:
                print(f"[ConfCalib] Overconfidence detected - median confidence delta "
                      f"{median_delta:.1f} < 5 across {len(deltas)} agents. "
                      f"Forcing one more evidence round.")
                return {
                    "convergence_score": CONVERGENCE_THRESHOLD - 0.01,
                    "overconfidence_corrected": True,
                }

    # Block early exit if unresolved challenges remain.
    if score >= CONVERGENCE_THRESHOLD and state.get("unresolved_challenges", 0) > 0:
        n = state["unresolved_challenges"]
        print(f"[DCI] {n} unresolved CHALLENGE act(s) - blocking convergence, forcing one more round")
        return {"convergence_score": CONVERGENCE_THRESHOLD - 0.01}

    # Classify debate into epistemic states using mean + variance of pairwise similarity.
    _dis_state, _ms, _sv = _disagreement_state(state["messages"])
    print(f"[DisRouting] state={_dis_state} mean={_ms:.3f} var={_sv:.3f}")
    if _dis_state == "convergent_agreement":
        return {"convergence_score": _ms}
    if _dis_state == "divergent_agreement":
        # Agents converged on the same conclusion via incompatible reasoning chains.
        # Exit is allowed (score is above threshold) but inject a caution for the Synthesizer.
        _caution = {
            "agent": "SYSTEM",
            "content": (
                "[DIVERGENT AGREEMENT] Agents reached the same conclusion via incompatible "
                "reasoning chains - the consensus may be superficial. Synthesizer: verify "
                "that agreement is substantive and not coincidental before issuing the verdict."
            ),
            "round": state["round"],
            "error": False,
        }
        return {"convergence_score": _ms, "messages": [_caution]}
    if _dis_state == "divergent_disagreement":
        # Positions are scattering in all directions without structure - inject a focus message.
        # convergence_score = _ms < CONVERGENCE_THRESHOLD, so route_after_convergence sends to tool_phase.
        _focus = {
            "agent": "SYSTEM",
            "content": (
                f"[DISAGREEMENT RESET] Debate has scattered across incompatible framings "
                f"without converging on any sub-question. All agents: identify the ONE "
                f"sub-question whose resolution most advances the core topic: "
                f"\"{state['topic']}\". Address only that sub-question this round."
            ),
            "round": state["round"],
            "error": False,
        }
        return {"convergence_score": _ms, "messages": [_focus]}
    # convergent_disagreement - uniform low-similarity disagreement, standard no-exit path
    return {"convergence_score": score}


def tool_phase_node(state: SimState) -> dict:
    """Gather targeted evidence per agent based on their current claim position."""
    topic = state["topic"]
    image_b64 = state.get("image_b64", "")

    # PROClaim: extract specific claims from last round's messages
    recent_agent_claims: dict[str, str] = {}
    for msg in state["messages"][-14:]:
        if not msg.get("error"):
            claim = _extract_claim(msg["content"])
            if claim and len(claim.split()) >= 5:
                recent_agent_claims[msg["agent"]] = claim

    existing_pool: dict = dict(state.get("evidence_pool") or {})

    def _evidence_is_novel(text: str) -> bool:
        """PROClaim novelty gate: reject evidence too similar to what we already have."""
        if not existing_pool:
            return True
        try:
            pool_samples = list(existing_pool.values())[:8]
            with MEMORY._encode_lock:
                enc = MEMORY._get_encoder()
                vecs = enc.encode([text] + pool_samples,
                                  convert_to_numpy=True, normalize_embeddings=True)
            max_sim = max(float(np.dot(vecs[0], vecs[i + 1])) for i in range(len(pool_samples)))
            return max_sim < 0.80   # admit if novelty > 0.20 (1 - 0.80)
        except Exception:
            return True

    def _call_tool(tool_name: str, query: str, agent_name: str) -> str:
        # ── Tools that need state (image, topic) rather than the query string ──
        if tool_name == "camera_snapshot":
            return tools.camera_snapshot(
                f"What does this scene tell us about: {topic}?", image_b64)
        if tool_name in ("search_knowledge_graph", "get_agent_opinions"):
            return tools.search_knowledge_graph(topic)
        if tool_name == "get_entity_relationships":
            first_word = query.split()[0] if query else topic
            return tools.get_entity_relationships(first_word)
        # ── Filesystem / git - kept for open-ended / dev topics ────────────────
        if tool_name == "list_directory":
            return tools.list_directory(".")
        if tool_name == "read_file":
            return tools.list_directory(".")
        if tool_name == "git_log":
            return tools.git_log(n=5)
        if tool_name == "git_diff":
            return tools.git_diff()
        if tool_name == "get_current_time":
            return tools.get_current_time("UTC")
        # ── Multi-arg tools - parse args from query string ─────────────────────
        if tool_name == "frankfurter_fx":
            import re as _re
            curs = _re.findall(r"\b([A-Z]{3})\b", query.upper())
            return tools.frankfurter_fx(
                curs[0] if curs else "USD",
                curs[1] if len(curs) >= 2 else "EUR",
            )
        if tool_name == "pint_convert":
            import re as _re
            m = _re.search(
                r"(-?\d+\.?\d*)\s+([\w/]+)\s+(?:to|in|into)\s+([\w/]+)", query, _re.I)
            if m:
                return tools.pint_convert(m.group(1), m.group(2), m.group(3))
            return tools.wolfram_query(query)   # Wolfram handles natural-language units
        # ── All single-string tools - dispatch via TOOL_MAP ────────────────────
        # This covers: web_search, web_search_and_fetch, wiki_summary,
        # wikidata_entity, openalex_search, semantic_scholar_search,
        # tavily_search, wolfram_query, sympy_solve, python_eval, fetch_url
        fn = tools.TOOL_MAP.get(tool_name)
        if fn:
            try:
                return fn(query)
            except TypeError:
                return fn()     # zero-arg tool (e.g. time)
        # Unknown tool - safe fallback
        return tools.search_knowledge_graph(topic)

    def fetch_one(agent_name: str) -> tuple[str, str]:
        tool_names = _WORKER_TOOLS.get(agent_name, ["web_search"])

        # PROClaim: use agent's specific claim as the search query for web tools
        agent_claim = recent_agent_claims.get(agent_name, "")
        claim_query = f"{agent_claim[:120]} evidence counterargument" if agent_claim else ""

        parts = []
        for tn in tool_names:
            try:
                if tn in ("web_search", "web_search_and_fetch") and claim_query:
                    query = claim_query
                    print(f"[PROClaim] {agent_name} - claim-reactive query: {query[:80]}")
                else:
                    query = f"{topic} {agent_name.lower()} perspective"

                res = _call_tool(tn, query, agent_name)

                if tn in ("web_search", "web_search_and_fetch") and not _evidence_is_novel(res[:400]):
                    print(f"[PROClaim] {agent_name} evidence redundant (sim >= 0.80) - skipping")
                    parts.append(f"[{tn}: evidence redundant with existing pool - skipped]")
                else:
                    parts.append(f"[{tn}]\n{res[:300]}")
            except Exception as exc:
                parts.append(f"[{tn}: error - {exc}]")
        return agent_name, "\n\n".join(parts)[:700]

    pool: dict = dict(existing_pool)

    with ThreadPoolExecutor(max_workers=len(TOOL_AGENTS)) as executor:
        futures = [executor.submit(fetch_one, n) for n in TOOL_AGENTS]
        for fut in as_completed(futures):
            try:
                name, result = fut.result()
                pool[name] = result
            except Exception as exc:
                print(f"[ToolPhase] fetch error: {exc}")

    return {"evidence_pool": pool, "phase": "evidence"}


def pivot_refine_gate_node(state: SimState) -> dict:
    try:
        score = _compute_convergence(state["messages"])
    except Exception as e:
        print(f"[PivotGate] Encoder failed, forcing oracle: {e}")
        score = CONVERGENCE_THRESHOLD  # treat as converged so we exit to oracle
    pivot_count = state.get("pivot_count", 0)
    if score >= CONVERGENCE_THRESHOLD or pivot_count >= MAX_PIVOTS:
        return {"convergence_score": score, "phase": "done"}
    return {"convergence_score": score, "pivot_count": pivot_count + 1, "phase": "evidence"}


def _disco_uq(messages: list[dict]) -> float:
    """Calibrate confidence using semantic entropy of final-round claims. Returns a multiplier in [0.5, 1.0]."""
    last_round = max((m["round"] for m in messages if not m.get("error")), default=0)
    final_msgs = [m for m in messages if m["round"] == last_round and not m.get("error")
                  and m["agent"] not in ("Synthesizer", "Contrarian", "SYSTEM")]
    claims = [_extract_claim(m["content"]) or m["content"][:200] for m in final_msgs]
    if len(claims) < 2:
        return 1.0
    try:
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            vecs = enc.encode(claims, convert_to_numpy=True, normalize_embeddings=True)
        n = len(vecs)
        # Gram kernel of unit-normalised embeddings is PSD with unit diagonal, so
        # trace(K) = n. Clamp tiny negatives to keep it a valid similarity kernel.
        kernel = np.clip(vecs @ vecs.T, 0.0, 1.0)
        rho = kernel / n                                  # density matrix, trace = 1
        eig = np.linalg.eigvalsh(rho)                     # real eigvals (real symmetric matrix)
        eig = eig[eig > 1e-12]                             # drop numerical-zero modes
        if eig.size == 0:
            return 1.0
        entropy = float(-np.sum(eig * np.log(eig)))       # von Neumann entropy
        s_norm = entropy / np.log(n)                       # normalise to [0, 1]
        s_norm = min(1.0, max(0.0, s_norm))
        return 1.0 - 0.5 * s_norm                          # map to [0.5, 1.0]
    except Exception:
        return 1.0


def _audit_reasoning_tree(messages: list[dict]) -> str:
    """Audit the minority position in the final round. Returns a block for Synthesizer if minority is better reasoned."""
    last_round = max((m["round"] for m in messages if not m.get("error")), default=0)
    final_msgs = [
        m for m in messages
        if m["round"] == last_round and not m.get("error")
        and m["agent"] not in ("Synthesizer", "Contrarian", "SYSTEM")
    ]
    if len(final_msgs) < 3:
        return ""

    agent_claims = [
        (m["agent"], _extract_claim(m["content"]) or m["content"][:150])
        for m in final_msgs
    ]
    texts = [c for _, c in agent_claims]

    try:
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            vecs = enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        centroid = vecs.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        sims_to_centroid = np.dot(vecs, centroid)
        minority_idx = int(np.argmin(sims_to_centroid))
        majority_mean_sim = float(np.mean(np.delete(sims_to_centroid, minority_idx)))
        minority_sim = float(sims_to_centroid[minority_idx])
        if majority_mean_sim - minority_sim < 0.15:
            return ""   # not divergent enough to bother auditing
    except Exception:
        return ""

    minority_agent, minority_claim = agent_claims[minority_idx]
    majority_summary = "; ".join(
        cl for a, cl in agent_claims if a != minority_agent
    )[:300]

    skeptic = ag.get_agent("Skeptic")
    audit = ag.call_llm(
        skeptic["personality"],
        f"Majority position (consensus): {majority_summary}\n\n"
        f"Minority position ({minority_agent}): {minority_claim}\n\n"
        f"Is the minority position BETTER REASONED - stronger evidence, fewer unsupported "
        f"assumptions, more internally consistent - than the majority?\n"
        f"Reply ONLY: VINDICATED or DISMISSED - one sentence reason.",
        max_tokens=60, enable_thinking=False, temperature=0.1,
    )
    if "VINDICATED" in audit.upper():
        print(f"[AgentAuditor] Minority vindicated: {minority_agent} - {minority_claim[:60]}")
        return (
            f"MINORITY POSITION VINDICATED - AgentAuditor reasoning-tree audit:\n"
            f"  {minority_agent}: {minority_claim}\n"
            f"  Auditor finding: {audit.strip()}\n"
            f"This minority branch has stronger reasoning than the majority. "
            f"Weight it heavily even if it contradicts consensus.\n\n"
        )
    print(f"[AgentAuditor] Minority dismissed ({minority_agent})")
    return ""


def _trace_level_synthesis(messages: list[dict]) -> str:
    """Cluster compatible EVIDENCE lines across agents and merge them into reasoning chains for Synthesizer."""
    evidence_items: list[tuple[str, str]] = []
    for m in messages:
        if m.get("error") or m["agent"] in ("Synthesizer", "SYSTEM"):
            continue
        for line in m["content"].split("\n"):
            if line.upper().startswith("EVIDENCE:"):
                ev = line[9:].strip()
                if len(ev.split()) >= 5:
                    evidence_items.append((m["agent"], ev))

    if len(evidence_items) < 3:
        return ""

    try:
        texts = [ev for _, ev in evidence_items]
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            vecs = enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        n = len(vecs)
        assigned = [False] * n
        clusters: list[list[int]] = []
        for i in range(n):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if not assigned[j] and float(np.dot(vecs[i], vecs[j])) >= 0.60:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)

        merged: list[str] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            agents = sorted({evidence_items[idx][0] for idx in cluster})
            evs = [evidence_items[idx][1] for idx in cluster[:3]]
            merged.append(f"[{' + '.join(agents)}] " + " | ".join(evs))

        if not merged:
            return ""

        print(f"[TraceSynth] {len(merged)} compatible reasoning chain(s) merged")
        return (
            "TRACE-LEVEL SYNTHESIS - compatible reasoning segments across agents:\n"
            + "\n".join(f"  - {c[:200]}" for c in merged[:4])
            + "\nThese chains represent convergent evidence from independent agents "
            "- they form the strongest factual foundation for the verdict.\n\n"
        )
    except Exception:
        return ""


def _counterfactual_probe(topic: str, messages: list[dict], final_conf: dict) -> str:
    """Probe the top confident claims with counterfactual scenarios. Returns a warning block if any fail."""
    import re as _re_cf

    # Collect highest-confidence claims from the full debate (deduplicated)
    seen: set[str] = set()
    candidates: list[tuple[int, str, str]] = []   # (confidence, agent, claim)
    for msg in reversed(messages):
        if msg.get("error") or msg["agent"] in ("Synthesizer", "Contrarian", "SYSTEM"):
            continue
        claim = _extract_claim(msg["content"])
        if not claim or claim in seen or len(claim.split()) < 5:
            continue
        seen.add(claim)
        candidates.append((final_conf.get(msg["agent"], 50), msg["agent"], claim))

    if not candidates:
        return ""

    # Probe the most confident claims - overconfidence is the highest hallucination risk
    candidates.sort(key=lambda x: -x[0])
    top = candidates[:2]
    claims_txt = "\n".join(f"CLAIM_{i+1} [{a}, conf {c}/100]: {cl}"
                           for i, (c, a, cl) in enumerate(top))

    # Step 1: generate counterfactual scenarios
    cf_scenarios = ag.call_llm(
        "You generate counterfactual reasoning challenges.",
        f"Topic: {topic}\n\nDebate claims to probe:\n{claims_txt}\n\n"
        f"For EACH claim write one counterfactual: "
        f"'What if [the opposite condition held] - would this claim still be valid?' "
        f"Keep each counterfactual to one sentence. "
        f"Format: CLAIM_1: <counterfactual> | CLAIM_2: <counterfactual>",
        max_tokens=160, enable_thinking=False, temperature=0.4,
    )
    if cf_scenarios.startswith("[") and len(cf_scenarios) < 200:
        return ""   # LLM unavailable - skip probe, don't block verdict

    # Step 2: Skeptic evaluates each claim under its counterfactual
    skeptic = ag.get_agent("Skeptic")
    evaluation = ag.call_llm(
        skeptic["personality"],
        f"Topic: {topic}\n\nOriginal claims:\n{claims_txt}\n\n"
        f"Counterfactual scenarios:\n{cf_scenarios}\n\n"
        f"For each claim decide: does it SURVIVE the counterfactual (the reasoning still holds), "
        f"or does it FAIL (the counterfactual exposes a flaw, unsupported assumption, or hallucination)?\n"
        f"Reply ONLY: CLAIM_1: SURVIVES or FAILS - one sentence reason. "
        f"Then CLAIM_2: SURVIVES or FAILS - one sentence reason.",
        max_tokens=120, enable_thinking=False, temperature=0.2,
    )

    flagged: list[str] = []
    for i, (conf, agent, claim) in enumerate(top):
        tag = f"CLAIM_{i+1}"
        # Find the evaluation line for this claim
        pat = _re_cf.search(rf'{tag}.*?(SURVIVES|FAILS)', evaluation, _re_cf.IGNORECASE)
        if pat and pat.group(1).upper() == "FAILS":
            flagged.append(f"{agent}: {claim[:100]}")

    if flagged:
        print(f"[CounterfactualProbe] {len(flagged)} claim(s) flagged as hallucination risk")
        return (
            "HALLUCINATION RISK - counterfactual probe:\n"
            + "\n".join(f"  - {f}" for f in flagged)
            + "\nThese claims did NOT survive counterfactual testing. "
            "Weight them with extra skepticism in the verdict.\n\n"
        )

    print(f"[CounterfactualProbe] All top claims survived counterfactual testing")
    return ""


# Weighted by each agent's self-reported confidence. Degenerates to plain majority when all confidence is zero.
VOTE_CONFIDENCE_WEIGHTED = True


def _vote_single_answer(
    messages: list[dict], mode: str, weighted: bool | None = None
) -> tuple[str, dict]:
    """Majority vote over each agent's FINAL: line. Confidence-weighted by default. Returns (answer, tally)."""
    import re as _re_v
    from collections import Counter, defaultdict

    if weighted is None:
        weighted = VOTE_CONFIDENCE_WEIGHTED

    # Each debate agent's LAST non-error message (their final position) + its confidence.
    # Contrarian is excluded: it's designed to oppose consensus, so on factual choice/numeric
    # tasks it systematically votes for the wrong answer and corrupts the majority.
    last_by_agent: dict[str, tuple[str, int]] = {}
    for m in messages:
        if m.get("error") or m["agent"] in ("Synthesizer", "SYSTEM", "Contrarian"):
            continue
        last_by_agent[m["agent"]] = (m["content"], m.get("confidence", 50))

    counts: Counter = Counter()
    weights: dict[str, float] = defaultdict(float)
    for content, conf in last_by_agent.values():
        fm = _re_v.search(r"FINAL:\s*(.+)", content, _re_v.IGNORECASE)
        segment = fm.group(1) if fm else content[-120:]   # fall back to the tail
        if mode == "numeric":
            nums = _re_v.findall(r"-?\d[\d,]*\.?\d*", segment)
            if not nums:
                continue
            vote = nums[-1].replace(",", "").rstrip(".")
        else:  # choice
            cm = _re_v.search(r"\b([A-H])\b", segment.upper())
            if not cm:
                continue
            vote = cm.group(1)
        counts[vote] += 1
        # clamp to [0,100] then normalise to [0,1]; missing confidence defaults to 50.
        weights[vote] += max(0.0, min(100.0, float(conf))) / 100.0

    if not counts:
        return "", {}

    # Plain-count path, or a weighted run where every confidence was zero (fall back so a
    # tie of zeros still yields the plain-majority answer instead of an arbitrary key).
    if not weighted or sum(weights.values()) == 0:
        return counts.most_common(1)[0][0], dict(counts)

    # Confidence-weighted winner; ties broken by raw vote count, then answer for determinism.
    winner = max(weights, key=lambda k: (weights[k], counts[k], k))
    return winner, {k: round(v, 2) for k, v in weights.items()}


def oracle_node(state: SimState) -> dict:
    # Resource guard: hard-stop before the (expensive) synthesis phase if the
    # host is critically low on memory - abort cleanly instead of OOM-crashing.
    import resource_guard
    resource_guard.assert_safe("oracle")

    answer_mode = state.get("answer_mode", "open")

    # SID: annotate each agent's contribution with their final-round
    # self-reported confidence so the Synthesizer weights arguments proportionally.
    conf_history = state.get("agent_confidence_history") or {}
    final_conf: dict[str, int] = {
        name: scores[-1] for name, scores in conf_history.items() if scores
    }

    def _sid_tag(agent_name: str) -> str:
        c = final_conf.get(agent_name, 50)
        if c >= 70:
            return f"[CONFIDENCE {c}/100 - weight heavily]"
        if c < 40:
            return f"[CONFIDENCE {c}/100 - treat with skepticism]"
        return f"[CONFIDENCE {c}/100]"

    # Cap per-agent content to avoid context overflow on resource-constrained hardware.
    # llama-server returns HTTP 400 (not silent truncation) when prompt > --ctx-size.
    # 7 agents × 400 chars ≈ 700 tokens; evidence capped separately below.

    # Deduplicate near-identical agent messages before synthesis.
    _debate_msgs = [m for m in state["messages"]
                    if not m.get("error") and m["agent"] != "Synthesizer"]
    try:
        _s2_contents = [m["content"][:800] for m in _debate_msgs]
        with MEMORY._encode_lock:
            _enc = MEMORY._get_encoder()
            _s2_vecs = _enc.encode(_s2_contents, convert_to_numpy=True, normalize_embeddings=True)
        _kept_idxs: list[int] = []
        _kept_vecs: list = []
        for _i, _vec in enumerate(_s2_vecs):
            if not _kept_vecs or max(float(np.dot(_vec, _kv)) for _kv in _kept_vecs) < 0.90:
                _kept_idxs.append(_i)
                _kept_vecs.append(_vec)
        _s2_filtered = [_debate_msgs[_i] for _i in _kept_idxs]
        if len(_s2_filtered) < len(_debate_msgs):
            print(f"[S²-MAD] {len(_debate_msgs)} to {len(_s2_filtered)} messages "
                  f"({len(_debate_msgs) - len(_s2_filtered)} near-duplicates removed)")
    except Exception:
        _s2_filtered = _debate_msgs

    all_opinions = "\n".join(
        f"[Round {m['round']}] {m['agent']} {_sid_tag(m['agent'])}: {m['content'][:400]}"
        for m in _s2_filtered
    )
    evidence_txt = "\n".join(
        f"  {a}: {v[:200]}" for a, v in (state.get("evidence_pool") or {}).items()
    )

    # In consensus-free mode, lead with the highest-scoring agent's position.
    if state.get("free_mad_mode"):
        scores = state.get("free_mad_scores", {})
        agent_scores = {k: v for k, v in scores.items() if not k.endswith("_prev")}
        if agent_scores:
            winner = max(agent_scores, key=lambda k: agent_scores[k])
            winner_score = agent_scores[winner]
            winner_msg = next(
                (m for m in reversed(state["messages"])
                 if m["agent"] == winner and not m.get("error")), None
            )
            if winner_msg:
                all_opinions = (
                    f"[FREE-MAD WINNER - score {winner_score:.1f}, "
                    f"chosen by score not consensus] {winner}:\n"
                    f"{winner_msg['content'][:500]}\n\n"
                    f"Full critique debate:\n{all_opinions}"
                )
                print(f"[FREE-MAD] Winner: {winner} (score={winner_score:.1f})")

    # If every agent errored there's nothing to synthesize - return a fallback verdict
    if not all_opinions:
        fallback = "VERDICT: Debate could not complete - all agent calls timed out or errored.\nCONFIDENCE: low\nCONFIDENCE_SCORE: 0"
        return {
            "messages": [{"agent": "Synthesizer", "content": fallback, "round": state["round"], "error": False}],
            "verdict": fallback,
            "report_path": "",
            "phase": "done",
        }

    # Contrarian challenges. Skipped for single-answer tasks: pushing agents off a
    # correct majority is exactly what wrecks numeric/choice accuracy.
    if answer_mode == "open":
        contrarian = ag.get_agent("Contrarian")
        challenge = ag.call_llm(
            contrarian["personality"],
            f"Topic: {state['topic']}\n\nFull debate:\n{all_opinions}\n\n"
            f"Shared evidence:\n{evidence_txt or '(none)'}\n\n"
            f"As Contrarian: identify the single biggest flaw or blind-spot in the emerging "
            f"consensus. Be specific, cite a speaker. 2-3 sentences max.",
            max_tokens=256,
            temperature=contrarian.get("temperature", 1.1),
        )
    else:
        challenge = "(skipped - single-answer mode)"

    # Synthesizer verdict
    synthesizer = ag.get_agent("Synthesizer")
    # Epistemic Context Learning: inject peer reliability hint so Synthesizer
    # weights arguments from historically accurate agents more heavily.
    from reputation import reputation_context as _rep_ctx_fn, record_verdict as _record_rep
    _participant_names = list({
        m["agent"] for m in state["messages"]
        if not m.get("error") and m["agent"] != "Synthesizer"
    })
    _rep_hint = _rep_ctx_fn(_participant_names)
    cf_warning = _counterfactual_probe(state["topic"], state["messages"], final_conf)
    audit_block = _audit_reasoning_tree(state["messages"])
    trace_block = _trace_level_synthesis(state["messages"])

    # Request a minority report when any CHALLENGE acts went unresolved.
    unresolved = state.get("unresolved_challenges", 0)
    minority_block = (
        f"\n6. MINORITY REPORT: {unresolved} CHALLENGE act(s) were raised but not fully "
        f"grounded or bridged. State the strongest unresolved objection here."
        if unresolved > 0 else ""
    )

    # Instruct Synthesizer to use inline confidence annotations.
    sid_hint = (
        "Agent confidence scores are annotated inline - weight [HIGH CONFIDENCE] arguments "
        "more heavily; treat [LOW CONFIDENCE] arguments as indicative, not authoritative.\n\n"
        if final_conf else ""
    )

    synth_prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Full debate (with SID confidence weights):\n{all_opinions}\n\n"
        f"Shared evidence:\n{evidence_txt or '(none)'}\n\n"
        f"Contrarian challenge:\n{challenge}\n\n"
        f"{trace_block}"
        f"{cf_warning}"
        f"{audit_block}"
        f"{sid_hint}"
        f"{_rep_hint}"
        f"Write a structured verdict:\n"
        f"1. VERDICT (one bold sentence)\n"
        f"2. STRONGEST ARGUMENT (who said what)\n"
        f"3. KEY TENSIONS (2-3 unresolved disagreements)\n"
        f"4. CONFIDENCE: low / medium / high + one-line reason\n"
        f"5. CONFIDENCE_SCORE: integer 0-100"
        f"{minority_block}"
        f"\n\nMANDATORY: The LAST line of your response MUST be exactly: CONFIDENCE_SCORE: <N>"
    )
    verdict_text = ag.call_llm(synthesizer["personality"], synth_prompt, max_tokens=3000,
                               temperature=synthesizer.get("temperature", 0.4),
                               enable_thinking=True)

    # Low-confidence fallback: re-synthesize with additional web evidence.
    import re
    m = re.search(r"CONFIDENCE_SCORE:\s*(\d+)", verdict_text)
    if m and int(m.group(1)) < 25:
        web = tools.web_search_tool(state["topic"], max_results=6)
        verdict_text = ag.call_llm(
            synthesizer["personality"],
            synth_prompt + f"\n\nAdditional web research:\n{web}\n\nRevise with same structure.",
            max_tokens=3000,
            temperature=synthesizer.get("temperature", 0.4),
            enable_thinking=True,
        )

    # Scale down confidence proportionally to semantic disagreement among agents.
    import re as _re_disco
    _score_m = _re_disco.search(r"CONFIDENCE_SCORE:\s*(\d+)", verdict_text)
    if _score_m:
        _raw_score = int(_score_m.group(1))
        _disco_mult = _disco_uq(state["messages"])
        _calibrated = int(_raw_score * _disco_mult)
        if _disco_mult < 0.90:
            print(f"[DiscoUQ] CONFIDENCE_SCORE calibrated: {_raw_score} to {_calibrated} "
                  f"(disagreement multiplier={_disco_mult:.2f})")
            verdict_text = (verdict_text[:_score_m.start(1)]
                            + str(_calibrated)
                            + verdict_text[_score_m.end(1):])

    # Fall back to mean agent confidence if the model omitted CONFIDENCE_SCORE.
    if not _score_m:
        _agent_confs = [m.get("confidence", 50) for m in state["messages"]
                        if not m.get("error") and m["agent"] not in ("Synthesizer", "SYSTEM")]
        if _agent_confs:
            _mean_conf = int(sum(_agent_confs) / len(_agent_confs))
            verdict_text = verdict_text + f"\nCONFIDENCE_SCORE: {_mean_conf}"
            print(f"[Conf] CONFIDENCE_SCORE missing - injecting mean agent confidence: {_mean_conf}")

    # For single-answer tasks, voted answer overrides Synthesizer prose.
    if answer_mode in ("numeric", "choice"):
        voted, tally = _vote_single_answer(state["messages"], answer_mode)
        if voted:
            _wmode = "conf-weighted" if VOTE_CONFIDENCE_WEIGHTED else "plain"
            print(f"[Vote] {answer_mode} majority ({_wmode}) = {voted}  tally={tally}")
            verdict_text = (
                f"VERDICT: {voted}\n\n"
                f"(majority vote of agents [{_wmode}], tally={tally})\n\n"
                f"{verdict_text}\n\nFINAL: {voted}"
            )

    # Update agent reputation from verdict.
    _record_rep(verdict_text, _participant_names, state.get("last_pair_idx", 0))

    # Propose a new improvement rule for the weakest agent when confidence is low.
    from constitution import maybe_update as _mac_update
    _mac_update(verdict_text, state["messages"])

    # Generate per-agent reflections in background.
    _mars_reflect_background(verdict_text, state["messages"], state["topic"])

    # Derive summary from verdict without a separate LLM call - used only for the .md report
    summary_lines = [l.strip() for l in verdict_text.split("\n") if l.strip()]
    summary = " ".join(summary_lines[:3])[:400]

    path = tools.write_report(
        topic=state["topic"],
        verdict=verdict_text,
        summary=summary,
        rounds=[m for m in state["messages"] if not m.get("error")],
    )

    return {
        "messages": [
            {"agent": "Contrarian", "content": challenge, "round": state["round"]},
            {"agent": "Synthesizer", "content": verdict_text, "round": state["round"]},
        ],
        "verdict": verdict_text,
        "report_path": path,
        "phase": "done",
    }


# Routing

def route_after_speak(state: SimState) -> str:
    if state["round"] > state["max_rounds"]:
        return "oracle"
    if state["phase"] == "debate":
        return "convergence_check"
    return "pivot_refine_gate"  # phase == "evidence"


def route_after_convergence(state: SimState) -> str:
    return "oracle" if state["convergence_score"] >= CONVERGENCE_THRESHOLD else "tool_phase"


def route_after_pivot(state: SimState) -> str:
    return "oracle" if state["phase"] == "done" else "speak_round"


# Graph

def build_graph():
    g = StateGraph(SimState)

    g.add_node("setup", setup_node)
    g.add_node("speak_round", speak_round_node)
    g.add_node("convergence_check", convergence_check_node)
    g.add_node("tool_phase", tool_phase_node)
    g.add_node("pivot_refine_gate", pivot_refine_gate_node)
    g.add_node("oracle", oracle_node)

    g.set_entry_point("setup")
    g.add_edge("setup", "speak_round")
    g.add_conditional_edges(
        "speak_round", route_after_speak,
        {"convergence_check": "convergence_check",
         "pivot_refine_gate": "pivot_refine_gate",
         "oracle": "oracle"},
    )
    g.add_conditional_edges(
        "convergence_check", route_after_convergence,
        {"oracle": "oracle", "tool_phase": "tool_phase"},
    )
    g.add_edge("tool_phase", "speak_round")
    g.add_conditional_edges(
        "pivot_refine_gate", route_after_pivot,
        {"oracle": "oracle", "speak_round": "speak_round"},
    )
    g.add_edge("oracle", END)

    return g.compile()


GRAPH = build_graph()
