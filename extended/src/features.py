
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureVector:
    participation_imbalance: float
    uptake: float
    goal_convergence: float
    regulatory_activity: float
    response_quality: float
    participation_long_horizon: float
    phase_elapsed: int
    state_confidence: float = 1.0

def extract_features(event, history):
    # Participation horizon is explicitly time-based: 4.5 minutes.
    # This keeps the implementation faithful to the documented decision loop
    # even when upstream events arrive at a cadence other than 60 seconds.
    cutoff = float(event["timestamp"]) - 270.0
    recent = [x for x in history if float(x["timestamp"]) >= cutoff]
    long_part = sum(x["participation_imbalance"] for x in recent) / max(1, len(recent))
    return FeatureVector(
        participation_imbalance=float(event["participation_imbalance"]),
        uptake=float(event["uptake"]),
        goal_convergence=float(event["goal_convergence"]),
        regulatory_activity=float(event["regulatory_activity"]),
        response_quality=float(event["response_quality"]),
        state_confidence=float(event["state_confidence"]),
        participation_long_horizon=float(long_part),
        phase_elapsed=int(event["phase_elapsed"]),
    )


def repair_signal(features, history, policy):
    """Observable recovery signal; never uses evaluation ground truth."""
    if not history:
        return False
    previous = history[-3:]
    prior_quality = sum(float(x["response_quality"]) for x in previous) / len(previous)
    prior_reg = sum(float(x["regulatory_activity"]) for x in previous) / len(previous)
    quality_improvement = features.response_quality - prior_quality
    regulation_improvement = features.regulatory_activity - prior_reg
    prior_problem = any(
        float(x["response_quality"]) < 0.60
        or float(x["regulatory_activity"]) < 0.55
        or float(x["participation_imbalance"]) > 0.55
        for x in previous
    )
    return (
        prior_problem
        and features.response_quality >= policy["repair"]["quality_threshold"]
        and features.regulatory_activity >= policy["repair"]["regulatory_threshold"]
        and (quality_improvement >= policy["repair"]["quality_improvement"]
             or regulation_improvement >= policy["repair"]["regulatory_improvement"])
    )
