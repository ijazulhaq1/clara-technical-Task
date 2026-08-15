# Changelog


## v6
- Added an explicit `unreliable_upstream` degraded-confidence scenario so the low-confidence check-in and coach-flag paths are exercised by evaluation.
- Added a hard test preventing unlimited low-confidence check-ins and verifying phase-local reset of the check-in refractory state.
- Reset check-in refractory state when the CBL phase changes, matching the phase-local check-in allowance.
- Kept corrective budget/refractory separate from the looser check-in allowance.
- Renamed the repair helper argument from `catalogue` to `policy` for clarity.
- Updated evaluation count and README terminology to five scenarios / 200 runs.


## v4 — evaluation integrity and metric separation

- Episode matching is now one-to-one; overlapping predictions cannot all be counted as true positives for one ground-truth episode.
- Separates episode detection, corrective-intervention episodes, and safe check-in support.
- Keeps the conservative 240-second hold unchanged; no threshold tuning was used to improve recall.
- Clarifies that confidence is used for routing, not multiplicative feature weighting.
- Clarifies the deterministic language-realisation stub and optional external LLM boundary.
# CLARA Part A — v3 changes

This version preserves the original architecture and makes targeted corrections identified during code review.

- Primary trigger evaluation is now **episode-level**; event-level metrics remain diagnostic. This avoids penalising one deliberate intervention repeatedly across a sustained trigger episode.
- Medium/low-confidence candidates receive a **neutral, non-corrective check-in** rather than being silently suppressed; low-confidence cases additionally set a coach flag.
- Check-ins do **not** consume the corrective intervention budget or corrective refractory period.
- Corrective scaffolds alone consume the phase budget and start the corrective refractory period.
- Participation long-horizon evidence uses an explicit **4.5-minute (270 s) time horizon** rather than a fixed number of records.
- Added a post-hoc **over-support metric** for evaluation only.
- Updated the independent standalone parity implementation and expanded tests from 42 to **49 passing tests**.

No change was made to artificially increase recall. The 240-second hold remains in place as a deliberate conservative intervention policy.


## v7 — evaluation consistency
- Corrected scenario-run count to 40 × 6 = 240.
- Aligned hold-duration curve with warranted-episode intervention scoring.
- Preserved conservative check-in/corrective routing and independent difficulty/confidence labels.
- Reworked hold-duration sensitivity so it simulates continuous candidate persistence and uses warranted-episode intervention scoring.
