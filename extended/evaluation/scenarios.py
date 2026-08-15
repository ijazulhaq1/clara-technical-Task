import random
SPEAKERS=("A","B","C","D")
SCENARIOS=("healthy","recoverable_imbalance","late_repair","unresolved_imbalance","hidden_difficulty","unreliable_upstream")

# Two labels, kept apart deliberately.
#
# ground_truth_trigger                 = difficulty is present.
# ground_truth_intervention_warranted  = intervening was the right act.
#
# They come apart exactly where the design thesis lives. A group that hits
# trouble and then sorts itself out HAD difficulty and did not need the agent.
# Scoring against the first label makes a plain feature threshold optimal by
# construction -- it is very nearly the same boolean -- and charges every
# suppressor as an error. Scoring against the second measures the system.
#
# Confidence is generated independently of difficulty: how hard a group is to
# hear is not how badly it is doing, and using one variable for both makes the
# fairness breakdown uninterpretable.

def _clip(x): return max(0.0,min(1.0,x))
def _base(r,v,noise=.05): return _clip(r.gauss(v,noise))

def generate_stream(seed,scenario,n_events=24,step_seconds=60):
    r=random.Random(100003*seed+7919*SCENARIOS.index(scenario))
    for i in range(n_events):
        t=i*step_seconds
        phase="framing" if i<4 else ("ideation" if i<16 else "building")
        difficulty=4<=i<=11
        recovery=i>=12
        if scenario=="healthy": vals=(.20,.78,.80,.78,.82); gt=False; rep=False; hidden=False; warranted=False
        elif scenario=="recoverable_imbalance":
            # Recovery is gradual, and completes inside the 240s hold. A step
            # change would make recovery unobservable until it was already over,
            # so the scenario would test nothing about repair detection.
            if i<4:    vals=(.20,.80,.82,.80,.82)
            elif i<=6: vals=(.68,.46,.50,.44,.46)
            elif i==7: vals=(.45,.62,.66,.60,.62)
            elif i==8: vals=(.28,.74,.76,.70,.74)
            else:      vals=(.20,.80,.82,.80,.82)
            gt=(4<=i<=6); rep=i>=7; hidden=False; warranted=False
        elif scenario=="late_repair":
            # The same group, recovering too late for the hold to catch it.
            # The system WILL interrupt here, and is scored as wrong for it.
            # Kept deliberately: the honest way to present a hold is to show
            # what it costs, not to pick only the timing that flatters it.
            if i<4:      vals=(.20,.80,.82,.80,.82)
            elif i<=13:  vals=(.68,.46,.50,.44,.46)
            elif i==14:  vals=(.45,.62,.66,.60,.62)
            elif i==15:  vals=(.28,.74,.76,.70,.74)
            else:        vals=(.20,.80,.82,.80,.82)
            gt=(4<=i<=13); rep=i>=14; hidden=False; warranted=False
        elif scenario=="unresolved_imbalance": vals=((.70,.43,.46,.40,.42) if i>=4 else (.20,.80,.82,.80,.82)); gt=i>=4; rep=False; hidden=False; warranted=i>=4
        elif scenario=="unreliable_upstream": vals=((.70,.43,.46,.40,.42) if i>=4 else (.20,.80,.82,.80,.82)); gt=i>=4; rep=False; hidden=False; warranted=i>=4
        else:
            # hidden_difficulty: no feature extreme, nothing recovers. Note the
            # confidence below is HIGH: this is the absence case, not the
            # unreliable-evidence case, and conflating them made check-ins here
            # impossible to attribute to a mechanism.
            vals=((.54,.52,.52,.50,.57) if i>=4 else (.25,.74,.75,.72,.82)); gt=i>=4; rep=False; hidden=i>=4; warranted=i>=4
        confidence_target = 0.34 if scenario=="unreliable_upstream" else 0.88
        yield {"seed":seed,"scenario":scenario,"event_id":f"{scenario}:{seed}:{i:03d}","group_id":f"G{seed:03d}","speaker":r.choice(SPEAKERS),"timestamp":t,"phase":phase,"phase_elapsed":t-(0 if phase=="framing" else 240 if phase=="ideation" else 960),"participation_imbalance":_base(r,vals[0]),"uptake":_base(r,vals[1]),"goal_convergence":_base(r,vals[2]),"regulatory_activity":_base(r,vals[3]),"response_quality":_base(r,vals[4]),"state_confidence":_base(r,confidence_target,.03),"ground_truth_trigger":gt,"ground_truth_repair":rep,"ground_truth_hidden":hidden,"ground_truth_intervention_warranted":warranted}
