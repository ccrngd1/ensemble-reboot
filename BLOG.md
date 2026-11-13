# Do LLMs From Different Labs Actually Fail Differently?

I spent $220 and 3,500 API calls in April testing Mixture-of-Agents on Bedrock. The headline finding: MoA ensembles only win when the proposers are clearly weaker than the aggregator. Three Novas feeding into Sonnet scored 92.4, roughly matching Opus at a quarter of the cost. But equal-capability ensembles (three Opus models feeding into Opus) performed worse than a single Opus call. More of the same model just added overhead.

That finding had a hole in it. The panel was almost entirely Claude and Nova. Same lab, same training lineage, presumably similar blind spots. The one condition that Wang et al.'s original MoA paper relied on for its gains, cross-lab model diversity, was never tested.

Bedrock fixed that problem for me. As of October 2026, I can call Anthropic (Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5), OpenAI (GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna), xAI (Grok 4.7), Moonshot (Kimi K3), and Amazon (Nova 2 Lite) through the same Converse API. Ten models from four labs plus Amazon, all on `global.` inference profiles. One API surface, one billing system, controlled conditions.

So I built the harness to find out.

## Why cross-lab diversity is the variable that matters

A June 2026 paper ("When Does Combining LMs Help?") studied 67 frontier models and found that ensemble gains are bounded by correlated failures. Models fail on the same inputs, and when they do, no amount of routing, voting, or aggregation recovers from it. The failures are structural, not random.

That raises a question the paper didn't directly answer: are failures correlated *because the models are large* (complexity breeds similar blind spots), or *because they share training lineage* (same lab, overlapping data pipelines, similar RLHF preferences)?

If it's the second one, cross-lab panels should have measurably lower co-failure rates than within-lab panels. That's a testable hypothesis, and the co-failure matrix is how you test it.

Recent evidence from outside the ensemble literature points the same direction. In November, Memdoor showed five parallel Qwen3.8-27B instances outperforming a single Fable 5.1 on LiveCodeBench Hard. Qwen3.8 costs roughly a tenth of Fable 5.1 per token, so five parallel instances still come in cheaper at higher accuracy. An arXiv paper (2611.09798) provided theoretical grounding: parallel multi-agent setups beat single long reasoning chains per unit of test-time compute.

But is that result about *model* diversity (different weights, different failure modes) or just *sampling* diversity (same model, independent rolls of the dice)? The ensemble-reboot harness is designed to separate those two things.

## Five arms, one grader

The experiment has five arms, all graded programmatically: letter matching for multiple choice, exact match for short answer, unit tests for code. No LLM judges. April's biggest weakness was open-ended judge scoring, and that's gone. Three task sets provide the questions: GPQA-Diamond (198 items), Humanity's Last Exam text-only (~250 items), and LiveCodeBench hard (61 items).

Arm A1 is the baseline: the single best model per task set, picked from a stage-0 run of all ten models. Everything else is measured against it.

A2 takes that same best model and runs it five times independently, aggregated by majority vote. Pure sampling diversity. Same weights, same training, just independent inference passes. If ensembles win here, the value is in sampling, not in model differences.

A3 applies seven persona prompts to the same model. Same weights, cosmetically different instructions. The April persona orchestrator survives as this control arm.

A4 is where cross-lab diversity enters. Three proposers from three different labs (GPT-6.1 Sol from OpenAI, Grok 4.7 from xAI, Kimi K3 from Moonshot) generate responses, and an Anthropic model aggregates them. The aggregator's lab is deliberately excluded from the panel so family self-preference can't contaminate the diversity claim. A4 runs twice: once with a strong aggregator (Fable 5.1, arm A4a) and once with a weak one (Haiku 5.5, arm A4b). If the weak aggregator still captures most of the gain, the value is in the diverse proposals. If it collapses, you need a strong synthesizer to extract the benefit of diversity.

From per-item correctness across all models, the harness computes a co-failure matrix. For every pair: observed P(both wrong) versus the independent expectation P(A wrong) × P(B wrong). Cross-lab pairs with lower observed co-failure than within-lab pairs at similar individual accuracy would be direct evidence that training lineage drives correlated failures.

## Reasoning effort is the other axis

Every model on the roster has reasoning effort controls, but they work differently per vendor. Anthropic uses `output_config.effort` (low through max). OpenAI and Moonshot use `reasoning_config`. Grok uses `reasoning_config` but caps at `xhigh` (no `max`). Nova 2 Lite uses `reasoningConfig` and needs you to omit `maxTokens` entirely at `high` effort, or Bedrock returns a 400.

The harness sweeps effort levels and asks: does a low-effort cross-family panel beat a single max-effort frontier model? If you can get better accuracy from three models thinking briefly than one expensive model thinking hard, the routing math changes.

## What the pilot already showed

A $75.46 pilot (1,049 calls, ten items per task set) revealed harness-level issues before any ensemble comparisons could be made. These are structural observations about the models and the API, not ensemble results.

Grok 4.7 is extremely slow. Median latency ranged from 205 to 416 seconds per call, with a max of 849 seconds. It blocked all eight workers during stage 0. Any production routing system needs to account for this.

Effort turns out to be a non-uniform knob. Opus 5.5's output tokens barely changed from low through xhigh (797 to 1,016 mean output tokens), then jumped to 4,642 at max. Fable and Astra scale monotonically. Per-model effort curves are required because "medium effort" means different things to different vendors.

On frontier models, max effort is disproportionately expensive. Fable 5.1 at max averaged $0.47 per item. Opus 5.5 at max on code emitted 18,916 mean output tokens with four out of ten truncations (at the original 32k maxTokens cap). After discovering that, the harness now uses 100k maxTokens to eliminate truncation artifacts.

Bedrock's error messages also deserve a mention: the "supported values" list in a 400 response doesn't always match what the model actually accepts. Only the observed 200/400 outcome is trustworthy, which is why the harness includes a level-probing script.

## What's built, what's next

Phase 1 is complete. 139 tests pass (2 skip on non-root environments that can't create network namespaces for code sandbox isolation). Datasets download and load. A dry run prints the planned call table with cost estimates labeled as assumed, not measured. All ten models are smoke-tested on Bedrock.

Full-run cost estimate: $697 for default effort across all arms, $3,751 for the full effort sweep. HLE stage-0 still has four design blockers awaiting decisions: a scoped runner, dataset mirror verification, math category scope within HLE, and exact-match grading limitations for short-answer items.

No accuracy numbers exist yet. No ensemble comparisons have been run. Results will come from live Bedrock calls in Phase 2, and I'll report them when they exist, not before.

```bash
git clone https://github.com/ccrngd1/ensemble-reboot
cd ensemble-reboot
pip install -e .
python3 -m pytest -q   # 139 passed, 2 skipped
```

If you're running multi-model architectures on Bedrock, the roster verification and effort-mapping work alone might save you some 400 errors. The co-failure question is the one I'm most interested in, though. If cross-lab diversity measurably decorrelates failures, that changes how you think about model selection for production ensembles. If it doesn't, the composition ceiling paper was right and the recovery has to come from somewhere other than model diversity.

I'll have the numbers when Phase 2 runs.
