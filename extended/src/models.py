
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Decision:
    triggered: bool
    function: Optional[str]
    severity: float
    confidence: float
    confidence_band: str
    hidden_difficulty: bool
    route: str
    scaffold_id: Optional[str]
    action: str
    reason: str

@dataclass
class GroupState:
    group_id: str
    first_trigger_at: Optional[int] = None
    last_intervention_at: Optional[int] = None
    budget_by_phase: dict = field(default_factory=dict)
    repair_candidate_since: Optional[int] = None
    unresolved_since: Optional[int] = None
    intervention_count: int = 0
    last_check_in_at: Optional[int] = None
    check_in_budget_by_phase: dict = field(default_factory=dict)
    check_in_count: int = 0
    pending_function: Optional[str] = None
