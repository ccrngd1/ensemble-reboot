# Ensemble Reboot — Does Model-Family Diversity Actually Help?

## What we're building
A single consolidated experiment that supersedes the April 2026 ensemble trilogy (thinking-models, moa-bedrock-guide, persona-orchestrator). It measures whether combining models from *different labs* beats a single frontier model, best-of-N, or same-model persona panels at matched cost. It also treats reasoning effort (low → max) as an experimental axis, and directly measures pairwise co-failure between model families.

## Why (blog angle)
April's MoA result was "ensembles only win when proposers are weaker than the aggregator; equal-capability ensembles lose." That ran without the condition Wang et al. relied on, which is cross-lab diversity. Bedrock now serves OpenAI, xAI, Moonshot and Anthropic frontier models side by side. This is the first clean test of whether family diversity breaks the [[composition-ceiling]] (correlated failures) or confirms it.

## Roster (verified live 2026-10-22 via Bedrock ListFoundationModels + Converse smoke test, 10/10 OK)
Use `global.` inference profiles. **No pre-2026-H2 models in the panel.** April's roster (Opus 4.6, Sonnet 4.6, Haiku 4.5, Nova 1, Mistral 7B, Llama 3.1) is retired.

| Role | Model (profile id) | Released | Effort control (additionalModelRequestFields) |
|------|--------------------|----------|-----------------------------------------------|
| Frontier, Anthropic | global.anthropic.claude-fable-5-1 | 2026-08-31 | `output_config.effort`: low…max (default high) |
| Frontier, Anthropic | global.anthropic.claude-opus-5-5 | 2026-09-22 | `output_config.effort`: low…max (default medium) |
| Frontier, OpenAI | global.openai.gpt-6-astra | 2026-09-08 | `reasoning_config`: low…max (default medium) |
| Frontier, OpenAI | global.openai.gpt-6.1-sol | 2026-10-15 | `reasoning_config`: low…max (default medium) |
| Frontier, xAI | global.xai.grok-4.7 | 2026-09-28 | `reasoning_config`: low…xhigh (default high, no max) |
| Frontier, Moonshot | global.moonshotai.kimi-k3 | 2026-08-27 | `reasoning_config`: none…max (default max) |
| Mid, Anthropic | global.anthropic.claude-sonnet-5-5 | 2026-09-28 | `output_config.effort` |
| Cheap, Anthropic | global.anthropic.claude-haiku-5-5 | 2026-10-07 | `output_config.effort` |
| Cheap, OpenAI | global.openai.gpt-6-luna | 2026-09-22 | `reasoning_config` (has none) |
| Cheap, Amazon | global.amazon.nova-2-lite-v1:0 | 2025-12-02 | `reasoningConfig` (`maxTokens` must be unset at `high`) |

Not in the panel: Nova Pro/Premier v1 (2024-era; no Nova 2 Pro on Bedrock), DeepSeek V3.2, Mistral Large 3, Qwen3, MiniMax M2.5. These can be optional open-weight diversity arms if budget allows.
OpenClaw's configured `mantle/xai.grok-4.3` and `mantle/openai.gpt-5.5` are **superseded**. Grok 4.3 is no longer on Bedrock, and GPT-5.5 is two generations behind. Call Bedrock runtime directly.

## Architecture
```mermaid
flowchart LR
  T[Task with verifiable answer] --> A1[Single best model]
  T --> A2[Best-of-N same model]
  T --> A3[Persona panel same model]
  T --> A4[Cross-family panel]
  A4 --> S1[Strong aggregator]
  A4 --> S2[Weak aggregator]
  A1 & A2 & A3 & S1 & S2 --> G[Programmatic grader]
  A4 --> C[Pairwise co-failure matrix]
```

## Scope
In:
- 5 arms: single-best, best-of-N, persona panel, cross-family + strong aggregator, cross-family + weak aggregator.
- **Cost analysis via cost-per-correct-answer and accuracy-vs-cost frontier** (the ±15% cost-matching constraint is retired; see Key decisions).
- Effort sweep: low / medium / high / xhigh (+ max where supported) on the single-best arm and on each panel member, plus at least one "low-effort cross-family panel vs high-effort single model" comparison. **Run top-up effort sweep for models not covered in the pilot** (~$50-80 estimated).
- Task sets with headroom and programmatic grading only: **GPQA-Diamond, HLE text-only (Humanity's Last Exam), and a code set with unit tests**. Pick sets where the best single model is <80%.
- Co-failure: for every model pair, P(both wrong) vs P(A wrong)·P(B wrong).
- Judge/aggregator, if used for selection, must come from a family not on the panel (avoid self-preference).
- **maxTokens: 100,000** for all calls (eliminates truncation artifacts from the pilot).

Out:
- Open-ended LLM-judge scoring (April's main weakness).
- GSM8K / MT-Bench / AlpacaEval (saturated or judge-scored).
- Hard competition math (AIME/HMMT) — dropped; CC prefers deep reasoning over math.
- Mock mode results in any reported number.

## Acceptance criteria
```gherkin
Given the 10-model roster above
When every arm runs on all three task sets with ≥3 seeds
Then each arm reports accuracy with 95% CI, cost per correct answer, and p50/p95 latency

Given panel members' per-item correctness
When the co-failure matrix is computed
Then each pair reports observed vs independent-expected joint failure rate

Given the effort sweep
When results are plotted
Then accuracy-vs-cost curves exist per model and per arm, and we can say whether effort or diversity buys more accuracy per dollar

Given any reported number
Then it traces to a raw JSONL record with model id, effort level, tokens, cost and grader output
```

## Key decisions
- Retire thinking-models (GSM8K saturated; Opus 4.6 obsolete). Carry over its one lesson: the judge must be ≥ the candidates.
- Persona orchestrator survives only as arm A3 (prompt diversity control).
- Spend gate: estimate the full-run cost from a 10-item pilot before running. CC approves any run >$100.
- Nova 2 Lite at `high` effort: omit `inferenceConfig.maxTokens` (Bedrock 400 otherwise).

### Post-pilot decisions (CC, 2026-10-27)
1. **Task sets: drop math, add HLE.** Math (AIME 2026 + HMMT Feb 2026) is retired. Replaced with HLE text-only (Humanity's Last Exam), a 2,500-question expert-level benchmark with real headroom (frontier models ~60-65%). Use HLE-Rolling (Sept 2026 cleaned version) if available on HF, otherwise cais/hle filtered to text-only. Sample to ~250 items if >500 text-only items, stratified by subject. MC items: shuffled choices, letter grading. Short-answer items: exact-match.
2. **Pin one frontier B + cost-per-correct.** Drop the ±15% cost-matching constraint (structurally infeasible with grok-4.7/kimi-k3 token costs vs cheap B models). Instead: (a) pin one frontier B across all sets (opus-5-5 and gpt-6-astra tied at 28/30 in pilot; final pick after HLE stage-0), and (b) report cost-per-correct-answer and the accuracy-vs-cost frontier for all arms.
3. **maxTokens: 100k.** Pilot had 23 truncations at 32k (grok 10/30, fable 6, opus 4). Raise to 100,000 to eliminate truncation artifacts. Accept the cost increase for high-effort cells.
4. **Keep fable-5-1 as A4a aggregator.** Rule-4(b) soft fail on GPQA (fable 8/10 vs kimi 9/10) was n=10 noise. Re-check on full stage-0; swap only if fable is still below kimi on the full set.
5. **Run effort-sweep top-up.** The pilot only swept 3 of 10 models before the cap hit. Top up the remaining 7 (~$50-80 estimated) to get honest effort-axis data.

## Resources
- April write-ups: `../ensemble-moa-bedrock-guide/BLOG.md`, `../ensemble-thinking-models/EXECUTIVE_SUMMARY.md` (stale −3% lines; see CRITICAL_FINDING_SELFCONS.md), `../ensemble-persona-orchestrator/BLOG.md` (mock-only)
- Smoke test: `scripts/smoke_roster.py`, `results/smoke_2026-10-22.json`
- Wiki: [[composition-ceiling]], [[correctness-vs-completeness-verification]], projects/llm-ensemble
