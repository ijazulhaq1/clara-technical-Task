# CLARA Part A — Trigger and Scaffolding Demonstrator

This implementation directly demonstrates the Part A brief: it processes a small upstream state stream, identifies a warranted intervention with transparent rules, and selects an educator-authored scaffolding message.

## Inputs

1. `data/sample_stream.jsonl` — synthetic upstream state stream. It contains observed socio-cognitive/socio-emotional proxy features and `state_confidence`; it contains **no evaluation ground truth**.
2. `config/catalogue.json` — educator-authored scaffold catalogue. Every student-facing message originates here. `avoid_when` is treated as a hard retrieval constraint.
3. `config/policy.json` — transparent operational thresholds (hold, confidence bands, repair and escalation settings).

## Output

Running `python main.py` produces `outputs/interventions.jsonl`. Each record shows the observed event, trigger decision, confidence route, selected strategy, retrieval score, message, validation result, monitoring status and escalation/annotation state.

The main output is therefore directly inspectable as **input → trigger → scaffold**.

## Evaluation-only data

`evaluation/scenarios.py` generates synthetic ground truth labels solely for post-hoc validation. The runtime never reads those labels. `python -m evaluation.runner` produces `outputs/evaluation_results.json`, `outputs/annotations.jsonl`, and `outputs/timeline.png`. The timeline visualises the actual synthetic input signals and the runtime scaffold decisions for one seed across the six evaluation scenarios; it does not use ground-truth labels to make runtime decisions.

The evaluation uses 40 seeds × 6 scenarios = 240 scenario-runs. The primary metric scores interventions as points against warranted episodes: an episode counts as detected if at least one intervention lands inside it, and an intervention landing outside every warranted episode is a false alarm. Run-to-run episode matching is reported as a secondary diagnostic only, because it charges a policy that rations itself to one corrective per phase for its second intervention while charging nothing to a detector that fires on every window. Event-level metrics are diagnostic and should not be read as headline performance: they penalise rationing by construction. Corrective recall (0.66) is lower than any-support recall (0.99) by design, because a third of the warranted episodes are low-confidence cases where a corrective must not be sent and a check-in is the correct response. Corrective-intervention performance is reported separately from safe check-ins. These are validation-harness results, not classroom evidence.

## Design choices

- State severity and upstream classifier confidence are independent; confidence is used for routing rather than being treated as ground-truth evidence.
- Evidence is assessed in a 90-second decision window with a 30-second hop; a candidate must persist for 240 seconds before a high-confidence corrective scaffold is sent.
- A disappearing trigger clears the pending hold.
- Confidence is carried through the decision: high confidence can produce a corrective scaffold; medium/low confidence degrade to a low-risk check-in, with low-confidence cases additionally flagged for coach review. Check-ins do not consume the corrective intervention budget or corrective refractory period; they have their own two-per-phase allowance and 10-minute refractory period.
- Repair is inferred from observable changes in response quality/regulatory activity; ground truth is never used at runtime.
- A 4.5-minute participation horizon complements the 90-second decision window; TF-IDF retrieves from the educator-authored catalogue; no embedding download is required.
- The retrieved message is validated before being emitted.
- The agent cannot select, rank, decide or generate a learner solution. The current Part A language-realisation component is deterministic for reproducibility and requires no API key; an external LLM can be attached behind the same constrained interface as an optional realisation layer.


### Monitoring and support audit

Corrective scaffolds and low-risk check-ins are tracked separately. Check-ins use the neutral `C1/C2` catalogue entries and do not consume the corrective phase budget or refractory period. The evaluation reports post-hoc over-support as the rate of corrective scaffolds delivered on synthetic events already labelled as recovered; this is an evaluation metric only, not a runtime rule.


### Evaluation terminology
Episode matching is one-to-one: each ground-truth episode and predicted episode can be matched at most once. The evaluation reports three distinct support concepts: episode detection (any scaffold/check-in), corrective intervention (scaffold only), and safe support (check-in). Event-level recall remains a diagnostic. The default Part A language-realisation component is deterministic; an external LLM can be attached behind the same constrained interface. Confidence is used for routing, not as a multiplicative weight on feature values.


### Recovery timing
The 240-second persistence threshold is an operational intervention rule. CLARA does not attempt to predict future recovery: repair suppression acts on recovery evidence once it becomes observable.

The synthetic evaluation includes a degraded-confidence `unreliable_upstream` scenario to exercise the safe check-in and coach-flag paths.
