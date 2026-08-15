"""
CLARA technical challenge, Part A.
Ijaz Ul Haq.

Detects moments in a group session where an AI agent should offer support, and
picks the message to send.

    python3 clara_part_a.py

Standard library only, so there is nothing to install. Takes about a second and
gives the same numbers every time.

The idea the code is built around is that a trigger is not a state, it is a
trajectory without repair.

If you detect negative states and act on them, you get an agent that interrupts
every group having a bad five minutes, including the ones that were about to
sort themselves out. Two findings from the collaborative learning literature
push against that. Socio-cognitive and socio-emotional signals are coupled, so a
group that gets frustrated and then renegotiates its plan is regulating itself,
while a group that gets frustrated and carries on unchanged is not. The signal
is the same in both cases and the meaning is opposite. The second finding is
that when groups fail, the usual reason is that nobody noticed they needed help,
so the quiet stalled group needs its own detection path.

Both push the same way. An agent whose main risk is over-regulating needs most
of its logic devoted to staying quiet, and three of the four checks below are
there to stop an intervention rather than start one.

A larger version with tests, a fairness audit and seed sweeps is in extended/.
NOTES.md has the rationale and the limitations.
"""

import random
from collections import Counter

# --- configuration ------------------------------------------------------------
# Kept in one place because most of these are pedagogical questions rather than
# technical ones. How long to wait before helping a group, and how often it is
# acceptable to interrupt them, are decisions the teaching staff should be
# making, and they should not have to read the rest of the file to find them.

HOLD_S = 240          # wait before acting on a candidate; see (2)
REFRACTORY_S = 300    # minimum gap between corrective interventions
BUDGET_PER_PHASE = 1  # corrective interventions allowed per CBL phase
CHECK_IN_BUDGET = 2   # low-risk check-ins allowed per phase
CHECK_IN_GAP_S = 600
CONF_HIGH, CONF_MED = 0.78, 0.58   # upstream classifier reliability bands
STEP_S = 60


# =============================================================================
# (1) PROCESS A SAMPLE OF THE UPSTREAM DATA STREAM
# =============================================================================
# The brief says a pipeline has already classified the speech, so this starts
# after that step. Each record is one turn: the features, who spoke, when, and
# how confident the classifier was in its own output.
#
# There is no utterance text. In Challenge-Based Learning each group picks its
# own challenge, so anything keyed to topic vocabulary will not carry over from
# one group to the next, whereas participation structure and how the group
# moves over time will. Dropping the text also means the only student data held
# here is a pseudonymous label per turn.
#
# I used synthetic data because AMI, ICSI, MELD and IEMOCAP do not record when
# an intervention would have been warranted, and without that label there is
# nothing to check the trigger logic against.
#
# Each scenario has two labels rather than one: whether the group is in
# difficulty, and whether intervening was the right thing to do. They separate
# in the case the whole design is about, where a group in difficulty recovers on
# its own and did not need the agent. Scoring against the first label alone
# makes a simple threshold the best possible system by definition.

SCENARIOS = {
    # name: (profile function, warranted, what it tests)
    "healthy":            "nothing wrong, so any message is a false alarm",
    "self_repairing":     "in trouble, then recovers in time -> stay quiet",
    "late_repair":        "recovers after the wait -> gets interrupted, known cost",
    "unresolved":         "trouble that does not resolve -> send something",
    "quiet_stall":        "calm and even but going nowhere -> send something",
    "unreliable_upstream":   "real trouble, poor audio -> check in, do not diagnose",
}

GOOD = (0.20, 0.80, 0.82, 0.80, 0.82)   # imbalance, uptake, goal_conv, regulation, quality
BAD = (0.70, 0.43, 0.46, 0.40, 0.42)
MIDDLING = (0.54, 0.52, 0.52, 0.50, 0.57)


def _profile(scenario, i):
    """Feature means at turn i, plus whether intervening here was warranted."""
    if scenario == "healthy":
        return GOOD, False
    if scenario == "self_repairing":
        # Recovery here is gradual rather than a jump. If the group went from
        # struggling to fine in a single turn there would be no trend to read,
        # and the scenario would not test the repair logic at all.
        if i < 4:   return GOOD, False
        if i <= 6:  return BAD, False
        if i == 7:  return (.45, .62, .66, .60, .62), False
        if i == 8:  return (.28, .74, .76, .70, .74), False
        return GOOD, False
    if scenario == "late_repair":
        # Same group, recovering too late for the hold to catch it. The system
        # gets this one wrong every time. I kept it in because a wait of this
        # kind has a cost and the cost should be visible in the results.
        if i < 4:    return GOOD, False
        if i <= 13:  return BAD, False
        if i == 14:  return (.45, .62, .66, .60, .62), False
        if i == 15:  return (.28, .74, .76, .70, .74), False
        return GOOD, False
    if scenario == "quiet_stall":
        # Nothing here is extreme on its own, so a per-feature threshold sees
        # nothing. Only the combination shows the group is stuck.
        return (MIDDLING, True) if i >= 4 else ((.25, .74, .75, .72, .82), False)
    return (BAD, True) if i >= 4 else (GOOD, False)   # unresolved, unreliable_upstream


def generate_stream(scenario, seed, n=24):
    """One record per turn, as the upstream pipeline would emit them.

    The seed is worked out arithmetically instead of with hash(), because
    Python randomises string hashing between processes and the results would
    then change from one run to the next.
    """
    r = random.Random(100003 * seed + 7919 * list(SCENARIOS).index(scenario))
    # Confidence depends on the recording conditions, things like accent,
    # people talking over each other, or a microphone too far away. It has
    # nothing to do with how well the group is working, so it is generated
    # separately here. Tying the two together is a mistake I made in an earlier
    # version and it makes weak evidence look like a mild problem.
    conf_mean = 0.34 if scenario == "unreliable_upstream" else 0.88
    for i in range(n):
        vals, warranted = _profile(scenario, i)
        clip = lambda x: max(0.0, min(1.0, x))
        yield {
            "t": i * STEP_S,
            "phase": "framing" if i < 4 else ("ideation" if i < 16 else "building"),
            "speaker": r.choice("ABCD"),
            "participation_imbalance": clip(r.gauss(vals[0], .05)),
            "uptake": clip(r.gauss(vals[1], .05)),
            "goal_convergence": clip(r.gauss(vals[2], .05)),
            "regulatory_activity": clip(r.gauss(vals[3], .05)),
            "response_quality": clip(r.gauss(vals[4], .05)),
            "state_confidence": clip(r.gauss(conf_mean, .03)),
            "warranted": warranted,          # evaluation only; never read by the system
        }


# =============================================================================
# (2) IDENTIFY A MOMENT WHERE AN INTERVENTION IS WARRANTED
# =============================================================================

def detect_evidence(e):
    """Map the features onto regulation functions rather than onto emotions.

    The taxonomy CLARA will use does not exist yet, since the PhD candidate
    builds it during the project. If the rules name functions instead of
    emotions, the taxonomy can be replaced later by editing one mapping without
    touching anything else.
    """
    candidates = [
        ("participation_balance", e["participation_imbalance"], 0.58),
        ("strategy_alignment",    1 - e["uptake"],              0.48),
        ("shared_understanding",  1 - e["goal_convergence"],    0.52),
        ("monitoring",            1 - e["regulatory_activity"], 0.52),
    ]
    active = [(fn, sev) for fn, sev, thr in candidates if sev >= thr]
    if active:
        return max(active, key=lambda x: x[1])

    # This covers the group that is quietly getting nowhere. No individual
    # feature crosses a threshold, but together they show a stalled group, and
    # this is the case that most often goes unnoticed.
    composite = (0.30 * e["participation_imbalance"] + 0.25 * (1 - e["uptake"])
                 + 0.25 * (1 - e["goal_convergence"])
                 + 0.20 * (1 - e["regulatory_activity"]))
    return ("monitoring", composite) if composite >= 0.46 else None


def is_repairing(e, history):
    """Check whether the group is already sorting the problem out.

    This reads the trend in the features and never the ground truth labels. It
    asks for two things: the group has to have got back to a healthy level, and
    it has to be improving. A group whose difficulty fades into silence is not
    recovering, it has just stopped talking.

    This is what keeps the agent out of the way when a group is managing on its
    own, which is what we want to happen.
    """
    if not history:
        return False
    prev = history[-3:]
    prior_problem = any(x["response_quality"] < .60 or x["regulatory_activity"] < .55
                        or x["participation_imbalance"] > .55 for x in prev)
    improving = (e["response_quality"] - sum(x["response_quality"] for x in prev) / len(prev)) >= .12
    return (prior_problem and improving
            and e["response_quality"] >= .68 and e["regulatory_activity"] >= .62)


def decide(e, evidence, repairing, state):
    """Four checks sit between having evidence and acting on it, and three of
    them exist to stop the agent rather than let it through.

    This function only decides. Spending the budget and starting the refractory
    timer happen back in run_session once the message has passed validation,
    otherwise a blocked message would use up the group's allowance for that
    phase without anything being sent.
    """

    # First check: the group is handling it. This throws the candidate away
    # rather than holding on to it.
    if repairing:
        state["first_seen"] = None
        return "repair", None

    # Nothing there any more, so drop whatever we were holding. Without this a
    # trigger that shows up once and then disappears can still fire several
    # minutes later on evidence that has gone.
    if evidence is None:
        state["first_seen"] = None
        return "continue", None

    function, severity = evidence
    if state["first_seen"] is None:
        state["first_seen"] = e["t"]

    # Second check: wait. We do not act the moment a candidate appears, we hold
    # it and look again. When trouble starts you cannot tell a rough patch from
    # a group that is properly stuck, and the two only separate once you can see
    # whether anything has changed. Four minutes of waiting costs very little.
    # Interrupting a group that was managing on its own costs a lot more.
    if e["t"] - state["first_seen"] < HOLD_S:
        return "hold", None

    # Confidence sends the decision one of three ways instead of acting as an
    # on/off switch.
    #
    # A switch has a problem buried in it. A group the pipeline struggles to
    # hear, because of accents or crosstalk or one quiet member, would sit under
    # the threshold for the whole session and quietly get less help than
    # everyone else. The protection ends up doing the damage. So low confidence
    # weakens what we send rather than stopping us sending anything.
    band = ("high" if e["state_confidence"] >= CONF_HIGH
            else "medium" if e["state_confidence"] >= CONF_MED else "low")

    if band in ("medium", "low"):
        # Check-ins do not come out of the corrective budget, because rationing
        # them the same way would shortchange the groups we hear worst. They still need a
        # limit of their own though. Without one, a group on poor audio gets a
        # message every window and ends up contacted more than anybody else,
        # which is the same problem the other way round.
        if (state["last_check_in"] is not None
                and e["t"] - state["last_check_in"] < CHECK_IN_GAP_S):
            return "check_in_refractory", None
        if state["check_in_budget"].get(e["phase"], CHECK_IN_BUDGET) <= 0:
            return "check_in_spent", None
        return "check_in", ("any", severity)

    # Third check: leave space between corrective messages. This is a teaching
    # decision, not a rate limit.
    if state["last_corrective"] is not None and e["t"] - state["last_corrective"] < REFRACTORY_S:
        return "refractory", None
    if state["budget"].get(e["phase"], BUDGET_PER_PHASE) <= 0:
        return "budget_spent", None

    return "scaffold", (function, severity)


# =============================================================================
# (3) SELECT AN APPROPRIATE SCAFFOLDING MESSAGE
# =============================================================================
# Messages are picked from a catalogue written by teaching staff rather than
# generated. Anything a student sees can be traced back to something a person
# wrote and signed off. The avoid_when field is the useful one: it lets staff
# write down when a strategy would do more harm than good, and the system
# treats that as a rule it has to follow.

CATALOGUE = [
    {"id": "PB-1", "function": "participation_balance", "phases": ["framing", "ideation", "building"],
     "kind": "corrective", "avoid_when": "the imbalance is an agreed division of labour",
     "text": "Some perspectives have not been heard for a while. Could you go round the group and each add one thing to the current idea?"},
    {"id": "SU-1", "function": "shared_understanding", "phases": ["framing", "ideation"],
     "kind": "corrective", "avoid_when": "the group has already restated the task this phase",
     "text": "Before going further, could each of you say in one sentence what you think the group is trying to solve right now?"},
    {"id": "SU-2", "function": "shared_understanding", "phases": ["building"],
     "kind": "corrective", "avoid_when": "the group is exploring a pivot it has named",
     "text": "Which part of your original challenge statement does the piece you are building now actually address?"},
    {"id": "SA-1", "function": "strategy_alignment", "phases": ["framing", "ideation", "building"],
     "kind": "corrective", "avoid_when": "the group is already changing approach",
     "text": "It sounds like something here is stuck. What is the one obstacle you would most want removed, and what would you try next if it were?"},
    {"id": "MO-1", "function": "monitoring", "phases": ["framing", "ideation", "building"],
     "kind": "corrective", "avoid_when": "a plan was just agreed",
     "text": "What is the next concrete thing this group needs to decide, and who is going to take it forward?"},
    # Check-ins are low risk because they ask the group how things are going
    # rather than telling them something is wrong, so they are safe to send when
    # the evidence is thin.
    {"id": "CI-1", "function": "any", "phases": ["framing", "ideation", "building"],
     "kind": "check_in", "avoid_when": "a check-in was sent recently",
     "text": "Quick check-in: how is the group doing with the current step, and is there anything you would like help with?"},
]


def select_message(function, phase, kind):
    """Apply the teaching constraints first, then choose among what is left.

    The order matters. If you score the whole catalogue for similarity first,
    a strategy that reads well but is wrong for the situation can win on
    wording alone. Function, phase and kind are rules set by the teaching staff,
    so only the entries that satisfy them get considered at all.
    """
    pool = [s for s in CATALOGUE if s["kind"] == kind and phase in s["phases"]
            and s["function"] in (function, "any")]
    if not pool:
        return None
    # Prefer the most specific match (an exact function over the generic pool).
    return sorted(pool, key=lambda s: s["function"] == "any")[0]


def validate(message):
    """Last check before anything is sent to a group.

    The rules here are the ones teaching staff care about. The agent should
    support the group's thinking rather than do it for them, and it should not
    tell a group what it is feeling, since that is a guess from a speech model
    and being told you are frustrated by a system that misheard you is worse
    than being left alone. This matters more once a language model is writing
    the text, which is why the check runs on the output rather than being put
    in the prompt.
    """
    low = message.lower()
    if not message.strip().endswith("?"):
        return False, "not framed as a question"
    if any(p in low for p in ("the answer is", "you should just", "the solution is")):
        return False, "supplies the answer rather than scaffolding it"
    if any(p in low for p in ("you seem", "you appear", "you sound", "you are feeling")):
        return False, "diagnoses the group's emotional state"
    if any(p in low for p in ("one of you", "whoever", "you are not contributing")):
        return False, "singles out an individual"
    if not 10 <= len(message.split()) <= 45:
        return False, "length outside 10-45 words"
    return True, "ok"


# --- running one session ------------------------------------------------------

def run_session(events):
    state = {"first_seen": None, "last_corrective": None, "last_check_in": None,
             "budget": {}, "check_in_budget": {}}
    history, log = [], []
    phase = None
    for e in events:
        if e["phase"] != phase:      # each CBL phase gets a fresh allowance
            phase = e["phase"]
            state["budget"][phase] = BUDGET_PER_PHASE
            state["check_in_budget"][phase] = CHECK_IN_BUDGET
            state["first_seen"] = None
        route, hit = decide(e, detect_evidence(e), is_repairing(e, history), state)
        message = None
        if hit:
            kind = "check_in" if route == "check_in" else "corrective"
            strategy = select_message(hit[0], e["phase"], kind)
            if strategy:
                ok, why = validate(strategy["text"])
                if ok:
                    message = strategy["text"]
                    # Commit only on delivery.
                    if route == "check_in":
                        state["check_in_budget"][e["phase"]] = \
                            state["check_in_budget"].get(e["phase"], CHECK_IN_BUDGET) - 1
                        state["last_check_in"] = e["t"]
                    else:
                        state["budget"][e["phase"]] = \
                            state["budget"].get(e["phase"], BUDGET_PER_PHASE) - 1
                        state["last_corrective"] = e["t"]
                        state["first_seen"] = None
                else:
                    route = f"blocked:{why}"
            else:
                route = "no_strategy"
        log.append({**e, "route": route, "message": message})
        history.append(e)
    return log


def naive_baseline(events):
    """The obvious approach: put a threshold on the features and act straight
    away. No repair check, no waiting, no budget, no confidence handling.

    It is here because the comparison is the reason not to build it that way.
    """
    out = []
    for e in events:
        score = max(e["participation_imbalance"], 1 - e["uptake"],
                    1 - e["goal_convergence"], 1 - e["regulatory_activity"])
        out.append({**e, "route": "scaffold" if score >= .48 else "continue"})
    return out


# --- evaluation ---------------------------------------------------------------

def evaluate(seeds=25):
    """Two questions per episode: did a group that needed help get some, and did
    any help go to a group that did not need it?

    Each intervention is counted as a point falling inside or outside an
    episode. I tried matching runs of interventions against runs of ground truth
    first and it gave the wrong answer, because a system that sends one message
    per phase produces two separate interventions inside one long difficulty and
    the second was counted as a false alarm, while a detector firing on every
    single window formed one continuous run and was charged for nothing.
    """
    def score(runner, routes):
        tp = fp = fn = acts = 0
        for scenario in SCENARIOS:
            for seed in range(seeds):
                events = list(generate_stream(scenario, seed))
                log = runner(events)
                warranted = [x["t"] for x in events if x["warranted"]]
                fired = [x["t"] for x in log if x["route"] in routes]
                acts += len(fired)
                if warranted:
                    lo, hi = min(warranted), max(warranted)
                    inside = [t for t in fired if lo <= t <= hi]
                    tp += 1 if inside else 0
                    fn += 0 if inside else 1
                    fp += len(fired) - len(inside)
                else:
                    fp += len(fired)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, acts

    print(f"\nEvaluation over {len(SCENARIOS)} scenarios x {seeds} seeds")
    print("-" * 74)
    ours = score(run_session, {"scaffold", "check_in"})
    base = score(naive_baseline, {"scaffold"})
    print(f"{'':22s} {'precision':>10s} {'recall':>8s} {'interruptions':>15s}")
    print(f"{'this system':22s} {ours[0]:10.2f} {ours[1]:8.2f} {ours[2]:15d}")
    print(f"{'threshold baseline':22s} {base[0]:10.2f} {base[1]:8.2f} {base[2]:15d}")
    print("\nThe baseline finds every real difficulty, so recall was never the")
    print("hard part. It gets there by interrupting about ten times as often,")
    print("and most of those messages go to groups that did not need one.")

    print("\nPer scenario (routes taken across seeds)")
    print("-" * 74)
    for scenario, description in SCENARIOS.items():
        counts = Counter()
        for seed in range(seeds):
            for x in run_session(list(generate_stream(scenario, seed))):
                if x["route"] in ("scaffold", "check_in"):
                    counts[x["route"]] += 1
        got = dict(counts) or "silent"
        print(f"  {scenario:20s} {str(got):32s} {description}")


def main():
    # Run one session first so there is something concrete to look at, then the
    # numbers across all scenarios.
    events = list(generate_stream("unresolved", seed=1))
    print(f"Session: {len(events)} turns, {events[-1]['t'] // 60} minutes, "
          f"upstream confidence {sum(e['state_confidence'] for e in events) / len(events):.2f}")
    print("-" * 74)
    for x in run_session(events):
        if x["message"]:
            print(f"  t={x['t']:5d}s  {x['phase']:9s} [{x['route']}]\n            {x['message']}")
    evaluate()
    print("\nOne limitation, shown here rather than described in the notes:")
    print("'late_repair' is a group that recovers after the wait has expired, and")
    print("the system interrupts it every time. A longer wait would fix this case")
    print("and would miss real problems elsewhere. At t=540 there is no way to")
    print("know what the group does at t=900.")


if __name__ == "__main__":
    main()
