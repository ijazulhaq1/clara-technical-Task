def observe_response(state,event,decision,policy=None):
    if policy is None:
        from src.config import load_policy
        policy=load_policy()
    quality=float(event["response_quality"])
    if state.unresolved_since is None and decision["triggered"] and quality < 0.50:
        state.unresolved_since=event["timestamp"]
        return {"status":"monitoring","escalated":False,"annotation":None}
    if state.unresolved_since is not None:
        elapsed=event["timestamp"]-state.unresolved_since
        if quality>=0.68:
            state.unresolved_since=None; state.repair_candidate_since=event["timestamp"]
            return {"status":"recovered","escalated":False,"annotation":{"type":"repair","event_id":event["event_id"],"timestamp":event["timestamp"]}}
        if quality<0.50 and elapsed>=policy["escalation_seconds"]:
            return {"status":"unresolved","escalated":True,"annotation":{"type":"escalation","event_id":event["event_id"],"timestamp":event["timestamp"],"reason":"persistent_low_response_quality"}}
        return {"status":"monitoring","escalated":False,"annotation":None}
    return {"status":"monitoring","escalated":False,"annotation":None}
