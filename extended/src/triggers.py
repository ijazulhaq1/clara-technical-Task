from dataclasses import dataclass

@dataclass(frozen=True)
class TriggerEvidence:
    function: str
    severity: float
    evidence: tuple
    hidden: bool = False
    confidence: float = 0.0

def validate_event(event, speakers):
    required = ["event_id","seed","scenario","group_id","speaker","timestamp","phase","phase_elapsed",
                "participation_imbalance","uptake","goal_convergence","regulatory_activity","response_quality","state_confidence"]
    errors=[f"missing:{k}" for k in required if k not in event]
    if event.get("speaker") not in speakers: errors.append("invalid_speaker")
    for k in required[8:]:
        if k in event and not isinstance(event[k], (int,float)): errors.append(f"type:{k}")
        elif k in event and not 0 <= float(event[k]) <= 1: errors.append(f"bounds:{k}")
    if event.get("timestamp",0)<0 or event.get("phase_elapsed",0)<0: errors.append("negative_time")
    return not errors, tuple(errors)

def validate_temporal_order(events):
    errors=[]; last=-1
    for e in events:
        if e["timestamp"] < last: errors.append("timestamp_not_monotonic")
        last=e["timestamp"]
    return not errors, tuple(errors)

def evidence_for(features, policy=None):
    if policy is None:
        from src.config import load_policy
        policy=load_policy()
    t=policy["thresholds"]
    candidates=[
        ("inclusion", max(features.participation_imbalance, features.participation_long_horizon), ("participation_imbalance","participation_long_horizon"), t["participation_imbalance"]),
        ("shared_task_understanding", 1-features.goal_convergence, ("goal_divergence",), t["goal_divergence"]),
        ("progress_monitoring", 1-features.uptake, ("low_uptake",), t["low_uptake"]),
        ("socioemotional_monitoring", 1-features.regulatory_activity, ("regulatory_gap",), t["regulatory_gap"]),
    ]
    active=[TriggerEvidence(fn,float(s),ev,False,features.state_confidence) for fn,s,ev,th in candidates if s>=th]
    hidden_score=(.30*features.participation_imbalance+.25*(1-features.uptake)+.25*(1-features.goal_convergence)+.20*(1-features.regulatory_activity))
    if hidden_score>=t["hidden_difficulty"] and not active:
        active.append(TriggerEvidence("progress_monitoring",hidden_score,("composite_hidden_difficulty",),True,features.state_confidence))
    return sorted(active,key=lambda x:(x.severity,x.function),reverse=True)

def confidence_band(score, policy):
    if score >= policy["confidence"]["high"]: return "high"
    if score >= policy["confidence"]["medium"]: return "medium"
    return "low"
