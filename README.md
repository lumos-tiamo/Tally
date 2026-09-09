# TeamClaw

A multi-agent platform where the agent's only action is **writing Python in a
sandbox**, its context is allocated by an auditable **token budget**, and every
claim it makes is scored against **XBRL ground truth**.

Built to answer one question with data: *how much of an agent's quality comes
from its architecture rather than its model?* So it runs on free-tier models by
design — if a strong paid model produced the numbers, the question would be
unanswerable.

```
teamclaw doctor       # what is configured and reachable
teamclaw scenarios    # cost of adding a scenario
teamclaw tools        # tool-representation token costs
teamclaw dataset build && teamclaw dataset stats
teamclaw baseline     # the zero-model floor
teamclaw eval --arm full --level l1     # or --level l2 / l3
teamclaw ablate --level l1
```

---

## What is measured

Numbers below are real output from this repository, not targets. Anything not
yet measured says so.

### Ground-truth corpus — built, 177 cases

Constructed semi-automatically from SEC XBRL company facts, which is what makes
177 cases affordable where hand-labelling would cap at a few dozen.

| | |
|---|---|
| Cases | **177** (135 primary + 42 held out) |
| Companies × years | 60 × 3 |
| Quarantined | 0 |
| Build errors | 0 |
| Mean L1 field coverage | 84.7% |
| Balance-sheet identity | 177/177 pass, relative error 0.0 |

Selection is by **accounting difficulty**, not market cap: 12 bank cases and 12
insurance cases have no current/non-current split, 78 cases have no
cost-of-revenue line at all. Those absences are the entire test of *abstention* —
a model that invents a current ratio for a bank is hallucinating, and a corpus of
large-cap tech filers would never reveal it.

### L3 judgement ground truth — built, 118 year-over-year pairs

L3 is the judgement level: what do two years of figures actually show? The
obvious way to score it is an LLM judge, and a judge alone would be weak — it
cannot separate "wrote persuasively" from "noticed the right thing", and its own
reliability has to be established before its scores mean anything.

So L3 gets an **objective backbone**. A large part of what makes a diligence
finding correct is derivable: if revenue rose while operating cash flow fell,
that is a fact about the numbers. Nine such signals are computed from consecutive
years of L1 truth, each with a materiality threshold so ordinary noise does not
trigger it, and findings are scored for **recall** (did the agent notice what is
there) and **precision** (did it assert what is not). The judge then grades only
what genuinely needs judgement.

| | |
|---|---|
| Year-over-year pairs | **118** across 59 companies |
| Signals detected | 126, mean 1.07 per pair |
| **Quiet pairs (nothing material)** | **46 — 39%** |

| Signal | Severity | Fired | Detectable | Rate |
|---|---|---|---|---|
| liquidity_pressure | watch | 42 | 96 | 43.8% |
| receivables_outpacing_revenue | concern | 17 | 82 | 20.7% |
| negative_free_cash_flow | concern | 15 | 96 | 15.6% |
| revenue_up_cashflow_down | **red flag** | 14 | 118 | 11.9% |
| operating_loss | **red flag** | 11 | 88 | 12.5% |
| margin_compression | concern | 9 | 66 | 13.6% |
| equity_decline | watch | 8 | 118 | 6.8% |
| leverage_increase | watch | 6 | 84 | 7.1% |
| revenue_decline | concern | 4 | 118 | 3.4% |

The signal-dense cases are the ones that should be — PLUG (6 signals), RIVN (5),
Boeing (4) — and `detectable` varies by filer, so a bank is never penalised for
failing to report a current ratio it cannot compute.

Two design choices carry most of the weight here:

**The 39% quiet pairs are not filler.** They are the cases that test whether the
agent avoids inventing findings, and they carry no recall denominator by design —
a quiet case answered quietly has an *undefined* recall, not a recall of zero.
Reading 0.00 as failure would invert the interpretation of the best possible
outcome, so `MetricResult.applicable` says so and the pooled report counts
`correctly_silent` and `fabricated_on_quiet_case` separately.

**The prompt never names the signals.** Handing the agent the taxonomy would turn
analysis into filling in a form, and the metric would measure format compliance.
There is a test asserting the prompt contains none of the signal keys, and
scoring therefore falls back to keyword matching over prose — requiring two
distinct keywords, so a passing mention of "cash flow" is not credited as having
found the divergence.

### `workflow-c` — the autonomy comparison, run

"Autonomous agent" versus "workflow with LLM nodes" is usually argued
abstractly. The two are only comparable if built against the same tools and
scored by the same metrics, which is what having a platform is for: `workflow-c`
is a fixed six-node task graph that shares the corpus, tool modules, four
metrics and harness with every other arm, and differs only in **who decides the
next step**.

```
download → to_text → deterministic_pass → model_pass → assemble → validate
                     (caption lookup)     (only the
                                           ambiguous fields)
```

The model is called at exactly one node, and only for figures the caption lookup
could not settle confidently. `model_pass` is optional, so a case the table
resolves entirely still reaches `assemble` — which is why this arm runs with no
credentials at all.

It returns the same `CaseRunResult` as the agent runner, so the harness cannot
tell the two apart. That interchangeability is the fair-comparison guarantee, and
it means the validity guard applies here too.

**Which surfaced a distinction worth making.** This arm can legitimately consume
zero tokens — not because no model was reachable, but because none was needed.
Those two produce identical tables of zeros and mean opposite things, so
`measurement_kind` separates four cases:

| Kind | Meaning | Publishable |
|---|---|---|
| `model` | real tokens consumed | yes |
| `deterministic` | every case completed, no inference required | yes, **labelled as such** |
| `plumbing` | a test double served a call | no |
| `none` | zero tokens *and* cases errored — an unconfigured run | no |

A `deterministic` result carries an explicit note not to set it beside a model
arm's number without saying which is which.

### Zero-model floor — measured over all 135 primary cases

The document tools driven by a fixed caption table, with **no model calls at
all**. This is the floor a language model has to beat, and the cheapest
regression test for the extraction stack.

| Metric | Rate | |
|---|---|---|
| Numeric accuracy @strict (±0.5%) | **0.571** | 783 / 1372 |
| Citation verifiability | **0.985** | 1254 / 1273 |
| Abstention accuracy | **0.964** | 81 / 84 |

The shape matters more than the headline. Citations and abstentions are near
perfect because the tools only report what they actually located — so the
**43-point gap in numeric accuracy is the space a model has to earn**. And the
metric suite discriminates by difficulty as intended:

| Hardest | | Easiest | |
|---|---|---|---|
| utilities | 0.383 | semis | 0.917 |
| insurance | 0.386 | consumer | 0.789 |
| banks | 0.460 | staples | 0.708 |

### Context cost — measured

| | Tokens |
|---|---|
| One 10-K, whole, in the prompt | **61,206** |
| Tool signatures a step actually receives (5 of 17, retrieval-filtered) | **188** |
| All 17 tool signatures, no retrieval | 536 |
| All 17 tools as MCP JSON schemas | 1,456 |
| Tool stub sources (on disk, never in a prompt) | 6,739 |

A single filing exceeds a 32k window by itself, which is why the workspace holds
documents and the prompt holds digests.

**Retrieval cost is flat in registry size; both alternatives are linear.** At 17
tools the saving is real but modest, and reporting only that would understate the
mechanism — an agent wired to a few MCP servers has hundreds of tools.
`scripts/tool_retrieval_scaling.py` measures the same quantity as the registry
grows (sizes above 17 are synthetic tools shaped like real ones):

| Tools | Retrieved (k=8) | All signatures | All schemas | Saving vs schemas |
|---|---|---|---|---|
| 17 | **188** | 536 | 1,456 | 87.1% |
| 40 | **188** | 1,442 | 4,992 | 96.2% |
| 80 | **188** | 2,892 | 11,140 | 98.3% |
| 160 | **188** | 5,816 | 23,471 | 99.2% |
| 320 | **188** | 11,705 | 48,195 | 99.6% |

At 320 tools, publishing schemas costs **48,195 tokens — more than a 32k window
holds at all**. That is the argument for compiling tools into an importable
package rather than injecting their schemas.

### Arm context cost — measured, and honestly bounded

An arm differs from `full` in two separable ways: how much context it spends and
how accurate it is. The second needs a model; the first does not.
`scripts/arm_context_cost.py` drives an identical 16-step trajectory through
every arm across a window sweep.

At a 6,000-token window, where the budget actually binds:

| Arm | Mean tokens/step | Evicted | Compactions | vs full |
|---|---|---|---|---|
| `full` | 2,891 | 0 | 2 | — |
| `minus-compaction` | 3,345 | **597** | 0 | **+15.7%** |
| `minus-tool-retrieval` | 2,920 | 0 | 2 | +1.0% |
| `minus-memory` | 2,545 | 0 | 2 | −12.0% |

**Compaction prevents eviction**: without it the run spends 15.7% more context
per step *and* still loses 597 tokens to eviction. The effect disappears above a
12k window — the mechanism is inert when the budget is roomy, and the sweep is
there to show where the crossover is.

**What this measurement cannot show, stated plainly.** `minus-ledger` differs
from `full` by 0.1% in every configuration tested. The ledger's guarantee is a
*worst-case* property — hard floors and pinning stop retrieved memory from
evicting the operating contract — and it binds only when the requested content
genuinely exceeds the budget, which this trajectory never quite reaches. It is
demonstrated by a targeted test
(`test_retrieved_memory_cannot_evict_the_system_prompt`), not by a token saving,
and manufacturing a scenario to produce a flattering number here would be
dishonest. Its accuracy effect remains unmeasured until a model is wired up.

### Platform-abstraction cost — measured

Adding a scenario means writing an `AgentSpec` and tool modules. Nothing else.
`tests/test_platform_abstraction.py` asserts it by AST-checking that no scenario
subclasses or monkey-patches the runtime, and by running three scenarios
end-to-end through the same `Agent` class.

| Scenario | AgentSpec | Tools | Total | Runtime changes |
|---|---|---|---|---|
| `dd_finance` (deep) | 1,937 | 467 | 2,404 | **0** |
| `bi_analyst` | 136 | 104 | 240 | **0** |
| `deep_research` | 166 | 140 | 306 | **0** |
| `code_engineer` | 158 | 162 | 320 | **0** |

Light scenarios: **mean 288 lines, range 240–320, zero runtime changes.** The
platform runtime is 5,673 lines; the eval harness 1,608; the tests 2,078.

The four scenarios are deliberately *heterogeneous*, because a platform running
one scenario is an application with a plugin folder:

| Scenario | What it stresses | Bridged tools |
|---|---|---|
| Financial diligence | extreme context pressure, numeric precision | 2 |
| BI analyst | schema-as-context, execution safety on generated SQL | **0** |
| Deep research | sub-agent isolation, provenance as a tool invariant | 2 |
| Code engineer | sandbox filesystem, an external test oracle | **0** |

Two scenarios need no host bridge at all, which shows the bridge is an option the
platform offers rather than a dependency it imposes.

### Delivered against the design, and what is not

The design doc is `docs/superpowers/specs/2026-09-09-agent-platform-design.md`.
All 26 capabilities it specifies are implemented, all six roadmap stages are
landed, and all eight ablation arms exist. One thing it promised is **not** here:

**A-share Chinese annual reports.** Stage 5 named them as cross-language
generalisation evidence. The pieces that would need custom work are in place and
verified — the retrieval layer tokenises and ranks CJK correctly (character
unigrams plus IDF, with a Chinese stopword list), and the concept-mapping layer
is source-agnostic — but no A-share corpus is wired up. Unlike SEC, there is no
official structured API, so the ground truth would need a different acquisition
path and a different validation strategy than the XBRL identity check that
underpins the current 177 cases. Half-building it would have produced a corpus I
could not vouch for, which is worse than not having one.

### Not yet measured

**The model arms.** No provider credentials are configured in this repository, so
seven of the eight arms in `teamclaw ablate` — `full`, `minus-ledger`,
`minus-tool-retrieval`, `minus-memory`, `minus-compaction`, `minus-skills`,
`strong-naked` — are implemented and runnable but unrun. Same for the judge's κ
calibration, which needs both a judge model and a human-labelled set.

The eighth, `workflow-c`, needs no credentials to run and did (see below).

The harness will not let that gap be papered over. A run is a measurement only if
no test double served any call **and** real model tokens were consumed:

```
$ teamclaw eval --arm full --level l1
no usable model provider — every case would fail on NoProviderAvailable
and the report would be a table of zeros.
Configure at least one free-tier key in .env, or run ollama serve …
Run teamclaw baseline for the zero-model floor, which needs no provider at all.
```

Both halves of that guard exist because both failures happened here. The
scripted-provider path scores a perfect 1.000 on a case and is still refused
(`tests/test_harness_end_to_end.py`), and the *first* real invocation of
`teamclaw eval` with no credentials produced a full report of zeros marked
`valid_measurement: True` — indistinguishable from a genuine score of zero. There
is a test named after that one too.

**To produce them**, either path works and both cost nothing:

```bash
# A — entirely local. No account, no key. ~5GB model download.
brew install ollama && ollama serve &
ollama pull qwen3:8b
teamclaw doctor                       # ollama should read available=yes
teamclaw ablate --level l1 --limit 30

# B — free cloud tier. Faster and stronger; needs one signup.
#     Set any one of these in .env, then the same command:
#     TEAMCLAW_GEMINI_API_KEY / TEAMCLAW_GLM_API_KEY
#     TEAMCLAW_GROQ_API_KEY   / TEAMCLAW_CEREBRAS_API_KEY
teamclaw ablate --level l1 --limit 30
```

Path A cannot run the `judge`, by design — see the routing table. Everything
else, including all seven agent arms, runs on the local tier.

---

## How it works

```
Scenario      AgentSpec(persona, skills, toolset, budget, termination) + tools
              declarative only — no runtime code
──────────────────────────────────────────────────────────────────────────────
Orchestration agent loop · task graph · sub-agents · checkpoints · HITL
Context       ledger · compactor · layered memory · skill index
Execution     sandbox · workspace · tool registry · MCP→Python stubs · bridge
Models        purpose-tiered router · quota tracker · completion cache
Observability spans · token/cost accounting
Evaluation    harness · four metrics · judge calibration · ablation arms
```

### Code execution as the substrate

The agent's only action is running Python. Tools are not JSON schemas but modules
it imports:

```python
from tools import sec, doc, fin
meta = sec.download_filing(ticker="AAPL", fiscal_year=2024)   # host-side, audited
info = doc.to_text(meta["path"])                              # in-sandbox
hit  = doc.find_number(info["path"], "Total net sales")[ "candidates"][0]
print(hit["values"][0], hit["page_estimate"], hit["scale"])   # only this returns
```

**Only what it prints comes back.** Everything else stays in the workspace. One
decision buys three properties: the context stays small, step 40 can read what
step 3 wrote, and resuming a crashed run is a re-read rather than a replay — the
workspace *is* the checkpoint.

The sandbox runs with no network. Tools that need egress are executed host-side
and reached through a file-based RPC in the one directory both sides share, so
the host broker is a single chokepoint where every outbound call is authorised
and logged to `.rpc/audit.jsonl`.

Two backends, and a result always records which one produced it. The container
backend's properties are verified rather than asserted — `tests/test_docker_sandbox.py`
checks all ten against a live daemon and skips cleanly when one is absent:

| Property | Verified |
|---|---|
| Runs as non-root | uid 1000 |
| Network unreachable | kernel refuses the connection — not a static check |
| Root filesystem read-only | write to `/etc` raises |
| Workspace writable, visible on host | round-trips |
| Packages cannot be installed | `pip install` returns non-zero |
| Memory cap | 3 GB allocation killed (exit 137) |
| Wall clock | infinite loop killed (exit 124) |
| Host bridge from a network-free container | tool call reaches the host |
| Traceback line numbers | match the agent's own code exactly |
| Unshared mount path | detected, with an actionable reason |

The subprocess fallback is *honestly weaker* — it shares the host kernel and
filesystem namespace — and labels itself `process-rlimits-only` so any result
produced under it carries that fact.

### The context ledger

Every prompt is rebuilt from scratch, never appended to, and each slot bids for a
share of `window − max_output − safety`:

| Slot | Hard floor | Elasticity | Eviction |
|---|---|---|---|
| system / persona | 256 | 0.00 | never |
| active skills | 0 | 0.35 | drop bodies, keep the index |
| tool signatures | 200 | 0.55 | retrieval-filtered |
| workspace state | 120 | 0.45 | tree + digests, never file bodies |
| long-term memory | 0 | 0.90 | relevance × decay × confidence |
| session history | 400 | 0.75 | tail verbatim, middle summarised |

Hard floors are reserved first; surplus flows to the *least elastic* claimant.
Together those two rules are why retrieved memory can never squeeze out the
operating contract — the failure mode that makes "layered memory" claims
untrustworthy. There is a test named after it.

Every construction emits a `LedgerRecord` naming each slot's allowance, usage and
what it evicted, so any prompt can be explained after the fact:

```
[STTTTTTTTTTTTTTTTTTMMMMMMMMMHHHHHHHHHHHHHHHHHHHHHHHHHHHHH...] 6600/6800 tok
  S=system:162  T=tools:2088  M=memory:1056  H=history:3283
```

Compaction fires only at **step boundaries** above 70% utilisation — compressing
mid-reasoning breaks the chain the model is in the middle of — and it aborts if
the summary would not actually shrink what it replaces.

### Memory that can be interrogated

Five questions any reviewer asks about "layered memory", answered in code rather
than prose:

1. **When is it written?** Only through `MemoryStore.write`, at step boundaries,
   with an explicit `source` span id. There is no remember-everything path.
2. **Who decides what enters the context?** `recall` scores
   `relevance × decay × confidence` and the ledger's memory slot allocates a
   budget. Every recall can be explained: `rel=0.83 decay=0.97 conf=0.95 → 0.76`.
3. **What happens on conflict?** An explicit `ConflictPolicy`, defaulting to
   **highest-confidence, not last-write-wins** — because under last-write-wins a
   single low-confidence extraction destroys a verified fact. The loser is kept
   on disk marked superseded.
4. **What is trimmed first?** Memory: floor 0, highest elasticity of any slot.
5. **How accurate is recall?** `recall_metrics` gives the memory layer its own
   precision@k / recall@k against a labelled probe set, so it is not hiding
   inside an end-to-end score.

### Zero-cost model routing

Purpose is the routing key, not a model name, so the cost policy lives in one
place:

| Purpose | Routed to |
|---|---|
| plan / decide / code / reflect | free cloud tiers (Gemini Flash, GLM-Flash, Groq, Cerebras), then local as last resort |
| extract / classify / rewrite | local Qwen3-8B via Ollama, first |
| judge | cloud only — a model must not grade its own work |

The local tier is *last* on the reasoning purposes and *first* on extraction,
which encodes that an 8B model is a workable fallback planner and a perfectly
good extractor. It is on those lists at all because a machine with no cloud
credentials and a running Ollama is the genuine zero-cost configuration, and a
policy that omitted it there would fail every case with `NoProviderAvailable`
rather than run slowly. Judging stays cloud-only on purpose: a local-only setup
would otherwise have the actor grade its own output, and an eval that cannot
grade independently should fail loudly instead.

High-volume, low-difficulty calls go local *specifically* so they do not burn the
daily request budget that planning needs. Beyond that: quota tracked in sliding
windows with proactive avoidance (a 429 discovered by hitting it costs both
latency and a quota slot), degradation rather than retry on rate limits, and a
**disk cache keyed by request fingerprint** — which is what makes seven ablation
arms affordable, since the arms share most of their prompts.

Paid providers are unreachable unless a caller explicitly opts in; only the
`strong-naked` arm does.

### The convergence loop

A small model writing sandbox code fails constantly, so the loop that turns a
traceback back into a corrected attempt is most of whether the agent works.

Feedback is structured, not a raw traceback: exception class, the offending line
quoted from the agent's own code, a hint specific to the failure kind, and **the
misused module's real signatures re-injected**. Repetition is detected by
*mistake class* — `name 'df' is not defined` and `name 'tbl' is not defined` are
one misconception repeated, not two errors — and escalation is bounded: same
class twice → a stronger model, three times → a human, then abort.

### Evaluation

Ground truth comes from SEC XBRL, which the agent is **never given**.
`assert_no_truth_leak` fails the build if any tool could serve company facts, and
a test asserts it, because a tool that exposed the answer key would turn the task
from reading a filing into calling the oracle.

Four metrics, each for a failure the others cannot see:

| Metric | Level | Catches |
|---|---|---|
| Numeric accuracy (3 tolerance bands) | L1, L2 | wrong values; bands separate rounding from nonsense |
| **Citation verifiability** | L1 | a correct number that was guessed — the quote is not in the source |
| **Calculation consistency** | L2 | right inputs with broken arithmetic, *and* clean arithmetic over invented inputs |
| **Abstention accuracy** | L1, L2 | inventing a figure the filer never disclosed |
| **Signal recall / precision** | L3 | missing what the figures show, and asserting what they do not |

For the judgement layer the LLM judge grades a rubric on top of that backbone,
and its **Cohen's κ against human labels is reported per dimension**, with
dimensions below κ=0.6 withheld from the headline and flagged for human
spot-checks. A judge that does not agree with humans is not evidence about the
agent. Both plain and quadratic-weighted κ are reported, since a judge scoring 4
where a human scored 5 is not the same error as scoring 1.

---

## Notes from building it

Defects found by running against real filings rather than fixtures, each now a
named regression test:

- **`fy` in XBRL labels the filing, not the fact.** A FY2024 10-K carries FY2023
  and FY2022 comparatives, all tagged `fy: 2024`. Selecting on it silently mixes
  three years. Facts are matched on period end date instead.
- **A quarterly duration fact looks exactly like an annual one.** Only the span
  distinguishes them — the easiest way to build a truth set that is confidently
  wrong.
- **`filings.recent` holds ~1000 submissions.** A large bank publishing thousands
  of 8-Ks a year pushes its own 10-K out of that window within months, so reading
  only `recent` returns *no annual reports* for exactly the companies whose
  accounting is hardest. Fixing the pagination recovered 12 bank cases and 12
  REIT cases that had silently vanished.
- **`Assets = Liabilities + StockholdersEquity` is not the balance-sheet
  identity.** `StockholdersEquity` is parent-only by definition, so the check
  fails for every filer with noncontrolling interests. And adding `MinorityInterest`
  unconditionally double-counts it for filers who report the including-NCI total —
  producing a negative residual exactly equal to the NCI.
- **`\d{1,3}(?:,\d{3})*` splits `2024` into `202` and `4`.** Every year in the
  document arrives as a pair, and citation checks against any line containing one
  fail. The comma group needs `+`, not `*`.
- **Whitespace collapse is not uniform**, so a normalised offset cannot be scaled
  back to a raw offset by ratio. In HTML-converted filings the estimate lands tens
  of thousands of characters away — the difference between quoting the income
  statement and quoting the competition section. An exact index map is required.
- **A 10-K names every item twice.** Forward-greedy outline reconstruction locks
  onto the front index, where all 22 items appear in canonical order within a few
  hundred characters. Walking the item order *backwards* lands on the body.
- **`Percentage of total net sales` contains `total net sales`** and otherwise
  looks exactly like a statement row. Shape alone ties them; label position
  breaks the tie.
- **`python -I` discards `PYTHONPATH`**, so the tool package cannot be made
  importable through the environment. The bootstrap is one line, deliberately,
  because every preamble line shifts the traceback line numbers the convergence
  loop quotes back.
- **Docker Desktop shares only configured host paths, and an unshared bind mount
  does not fail — it becomes an empty directory.** The step script vanishes and
  the error surfaces as `can't find '__main__' module`, which reads like a bug in
  the agent's code. `/Users` is shared; `/var/folders` — where `mktemp` and
  pytest's `tmp_path` live on macOS — is not. There is now a mount probe, and a
  fallback that says why.
- **The shared filesystem propagates in both directions with a delay**, measured
  at up to **1.1 s**. That showed up twice. Writing the step script into the
  workspace and immediately starting the container failed about half the time, so
  the program is delivered over stdin instead and depends on no filesystem at
  all. And artefacts a container writes are not instantly visible to the host, so
  a step could finish and the agent read a workspace digest that omitted what it
  had just produced — the container backend now waits, briefly and boundedly, for
  evidence of the write.
- **A path the host deletes cannot always be recreated by the container.** The
  guest caches directory entries, so clearing a workspace host-side and then
  writing the same filename from inside the sandbox can fail with a
  `FileNotFoundError` on a *write*. Two wrong diagnoses preceded the right one —
  first the mount inode, then the propagation delay — and it was narrowing it to
  *the same filename* that identified it. Every run now gets its own workspace
  path instead of clearing and reusing one, which removes the interaction rather
  than working around it, and keeps each case's artefacts for inspection.

---

## Layout

```
src/teamclaw/
  config.py           settings and paths
  models/             providers, purpose-tiered router, quota, cache
  context/            ledger, slots, compactor, memory, skills, retrieval
  execution/          sandbox, workspace, registry, stubgen, bridge, convergence
  orchestration/      agent loop, actions, checkpoints, HITL, sub-agents, graph
  observability/      spans, token/cost accounting
  evaluation/         harness, metrics, judge calibration, ablation arms, workflow arm
  scenarios/
    dd_finance/       corpus, concept mapping, ground truth, signals, sandbox tools
    bi_analyst/       SQL over a read-only database
    deep_research/    provenance-enforcing note store
    code_engineer/    repo edits verified by the test suite
docker/Dockerfile.sandbox
scripts/deterministic_baseline.py
scripts/build_l3_signals.py
scripts/arm_context_cost.py
scripts/tool_retrieval_scaling.py
results/               measured output, committed
tests/                 200 tests — 189 offline, 11 container-gated
docs/superpowers/specs/2026-09-09-agent-platform-design.md
```

## Setup

```bash
uv venv --python 3.13 && uv pip install -e ".[dev]"
cp .env.example .env          # set TEAMCLAW_SEC_USER_AGENT at minimum
python -m pytest -q           # 189 offline; 11 more if a sandbox image exists
teamclaw doctor
```

SEC requires a self-identifying `User-Agent`; the client refuses to fetch without
one and rate-limits itself well below the published ceiling. Responses are cached
to disk, which is a correctness property rather than an optimisation: filings get
amended, and an eval re-run next week must score against the corpus it was built
from.

Optional: `docker build -t teamclaw-sandbox:latest -f docker/Dockerfile.sandbox docker/`
for container isolation, and `ollama pull qwen3:8b` for local extraction.
