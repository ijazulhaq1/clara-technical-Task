from src.triggers import confidence_band

def decide_policy(evidence,state,event,policy=None,repairing=False):
    if policy is None:
        from src.config import load_policy
        policy=load_policy()
    band=confidence_band(event["state_confidence"],policy)
    if repairing:
        state.first_trigger_at=None; state.pending_function=None
        return {"triggered":False,"function":None,"severity":0.0,"confidence":event["state_confidence"],"confidence_band":band,"hidden_difficulty":False,"route":"repair","action":"continue","reason":"self_repair_detected","coach_flag":False}
    if not evidence:
        state.first_trigger_at=None; state.pending_function=None
        return {"triggered":False,"function":None,"severity":0.0,"confidence":event["state_confidence"],"confidence_band":band,"hidden_difficulty":False,"route":"continue","action":"continue","reason":"no_trigger","coach_flag":False}

    best=evidence[0]
    if state.first_trigger_at is None:
        state.first_trigger_at=event["timestamp"]
    state.pending_function=best.function
    held=event["timestamp"]-state.first_trigger_at >= policy["hold_seconds"]
    base={"triggered":False,"function":best.function,"severity":best.severity,"confidence":best.confidence,"confidence_band":band,"hidden_difficulty":best.hidden,"coach_flag":False}

    if not held:
        return {**base,"route":"hold","action":"monitor","reason":"hold_not_satisfied"}

    # Uncertainty reduces intervention strength rather than eliminating support.
    #
    # Check-ins sit outside the CORRECTIVE budget on purpose: rationing them the
    # same way would under-serve precisely the groups the upstream pipeline
    # hears worst, which is the equity failure this routing exists to prevent.
    #
    # But "not drawn from the corrective budget" is not "unlimited". A low-risk
    # message is still an interruption, and without its own allowance a group on
    # poor audio receives one every window -- ending up contacted more often
    # than any well-heard group, which is the same failure with the sign
    # flipped. So check-ins get their own, looser budget and refractory.
    if band in ("low", "medium"):
        ci_refractory = (state.last_check_in_at is not None
                         and event["timestamp"] - state.last_check_in_at
                         < policy["check_in_refractory_seconds"])
        ci_budget = state.check_in_budget_by_phase.get(
            event["phase"], policy["check_in_budget_per_phase"])
        if ci_refractory:
            return {**base,"route":"check_in_refractory","action":"monitor","reason":"check_in_refractory","coach_flag":band=="low"}
        if ci_budget <= 0:
            return {**base,"route":"check_in_budget_exhausted","action":"monitor","reason":"check_in_budget_exhausted","coach_flag":band=="low"}
        reason = "low_confidence_safe_checkin" if band == "low" else "medium_confidence_safe_checkin"
        return {**base,"triggered":True,"route":"check_in","action":"check_in","reason":reason,"coach_flag":band=="low"}

    refractory=state.last_intervention_at is not None and event["timestamp"]-state.last_intervention_at < policy["refractory_seconds"]
    budget=state.budget_by_phase.get(event["phase"],policy["phase_budget"])
    if refractory:
        return {**base,"route":"refractory","action":"monitor","reason":"refractory"}
    if budget<=0:
        return {**base,"route":"budget_exhausted","action":"monitor","reason":"phase_budget_exhausted"}
    return {**base,"triggered":True,"route":"scaffold","action":"scaffold","reason":"persistent_trigger"}
