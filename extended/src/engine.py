from src.features import extract_features, repair_signal
from src.models import GroupState
from src.triggers import evidence_for, validate_event, validate_temporal_order
from src.policy import decide_policy
from src.retrieval import CatalogueRetriever
from src.llm import DeterministicLLM, validate_message
from src.monitor import observe_response

def canonical(record):
    keys=["event_id","timestamp","phase","triggered","route","confidence_band","scaffold_id"]
    return tuple(record.get(k) for k in keys)

def run_stream(events,catalogue,policy=None):
    if policy is None:
        from src.config import load_policy
        policy=load_policy()
    events=list(events)
    if not events: return []
    speakers={e["speaker"] for e in events}
    ok,errors=validate_temporal_order(events)
    if not ok: raise ValueError(errors)
    state=GroupState(events[0]["group_id"]); retriever=CatalogueRetriever(catalogue); llm=DeterministicLLM(); history=[]; results=[]; current_phase=None; last_strategy=None
    for event in events:
        valid,errors=validate_event(event,speakers)
        if not valid: raise ValueError(errors)
        if event["phase"]!=current_phase:
            current_phase=event["phase"]; state.budget_by_phase[current_phase]=policy["phase_budget"]; state.check_in_budget_by_phase[current_phase]=policy["check_in_budget_per_phase"]; state.first_trigger_at=None; state.repair_candidate_since=None; state.last_check_in_at=None; last_strategy=None
        features=extract_features(event,history)
        repairing=repair_signal(features,history,policy)
        evidence=evidence_for(features,policy)
        decision=decide_policy(evidence,state,event,policy,repairing=repairing)
        scaffold=None; message=None; msg_valid=True; msg_errors=()
        if decision["route"] in ("scaffold","check_in"):
            retrieval_function = "any" if decision["route"] == "check_in" else decision["function"]
            scaffold=retriever.retrieve(retrieval_function,event["phase"],decision["severity"],avoid_ids=[last_strategy] if last_strategy else [])
            if scaffold is None: decision={**decision,"triggered":False,"route":"blocked","action":"monitor","reason":"no_catalogue_match"}
            else:
                message=llm.generate(scaffold,event); msg_valid,msg_errors=validate_message(message,scaffold)
                if not msg_valid: decision={**decision,"triggered":False,"route":"blocked","action":"monitor","reason":"message_validation_failed"}
                else:
                    if decision["route"] == "scaffold":
                        state.budget_by_phase[event["phase"]]-=1
                        state.last_intervention_at=event["timestamp"]
                        state.intervention_count+=1
                        state.first_trigger_at=None
                        state.pending_function=None
                        last_strategy=scaffold["id"]
                    else:
                        # A delivered check-in draws on its own allowance. It
                        # does NOT clear first_trigger_at: the underlying
                        # difficulty is unresolved, and if confidence recovers
                        # the corrective should not have to re-serve the hold.
                        state.check_in_budget_by_phase[event["phase"]]-=1
                        state.last_check_in_at=event["timestamp"]
                        state.check_in_count+=1
                        last_strategy=scaffold["id"]
        monitor=observe_response(state,event,decision,policy)
        annotation_type=monitor["annotation"]["type"] if monitor["annotation"] else None
        results.append({"seed":event["seed"],"scenario":event["scenario"],"group_id":event["group_id"],"speaker":event["speaker"],"event_id":event["event_id"],"timestamp":event["timestamp"],"phase":event["phase"],"function":decision["function"],"triggered":decision["triggered"],"route":decision["route"],"action":decision["action"],"reason":decision["reason"],"state_confidence":event["state_confidence"],"confidence_band":decision["confidence_band"],"coach_flag":decision.get("coach_flag",False),"hidden_difficulty":decision["hidden_difficulty"],"severity":decision["severity"],"participation_imbalance":event["participation_imbalance"],"uptake":event["uptake"],"goal_convergence":event["goal_convergence"],"regulatory_activity":event["regulatory_activity"],"response_quality":event["response_quality"],"scaffold_id":scaffold["id"] if scaffold else None,"strategy_name":scaffold["name"] if scaffold else None,"retrieval_score":scaffold["retrieval_score"] if scaffold else None,"message":message,"message_valid":msg_valid,"message_errors":list(msg_errors),"monitor_status":monitor["status"],"escalated":monitor["escalated"],"annotation_type":annotation_type,"budget_remaining":state.budget_by_phase[event["phase"]]})
        history.append(event)
    return results
