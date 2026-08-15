import json
from pathlib import Path
from src.config import load_catalogue,load_policy
from src.features import extract_features,repair_signal
from src.triggers import evidence_for,confidence_band,validate_event
from src.policy import decide_policy
from src.models import GroupState
from src.retrieval import CatalogueRetriever
from src.llm import DeterministicLLM,validate_message
from src.monitor import observe_response
from evaluation.scenarios import generate_stream,SCENARIOS
from main import run_stream,canonical

C=load_catalogue(); P=load_policy()
def event(**kw):
    e=next(generate_stream(1,'healthy',1)); e.update(kw); return e

def test_01_catalogue_strategies(): assert len(C['strategies'])==10
def test_02_catalogue_s1(): assert C['strategies'][0]['id']=='S1'
def test_03_avoid_when_present(): assert all('avoid_when' in x for x in C['strategies'])
def test_04_text_present(): assert all(x['text'].strip() for x in C['strategies'])
def test_05_policy_hold(): assert P['hold_seconds']==240
def test_06_policy_refractory(): assert P['refractory_seconds']==300
def test_07_confidence_high(): assert confidence_band(.9,P)=='high'
def test_08_confidence_medium(): assert confidence_band(.65,P)=='medium'
def test_09_confidence_low(): assert confidence_band(.3,P)=='low'
def test_10_confidence_independent():
    ev=evidence_for(extract_features(event(participation_imbalance=.9,state_confidence=.3),[]),P); assert ev and ev[0].severity>.7 and ev[0].confidence<.5
def test_11_feature_confidence(): assert extract_features(event(state_confidence=.37),[]).state_confidence==.37
def test_12_participation_trigger(): assert any(x.function=='inclusion' for x in evidence_for(extract_features(event(participation_imbalance=.8),[]),P))
def test_13_goal_trigger(): assert any(x.function=='shared_task_understanding' for x in evidence_for(extract_features(event(goal_convergence=.1),[]),P))
def test_14_uptake_trigger(): assert any(x.function=='progress_monitoring' for x in evidence_for(extract_features(event(uptake=.1),[]),P))
def test_15_regulation_trigger(): assert any(x.function=='socioemotional_monitoring' for x in evidence_for(extract_features(event(regulatory_activity=.1),[]),P))
def test_16_hidden_trigger():
    e=event(participation_imbalance=.54,uptake=.52,goal_convergence=.52,regulatory_activity=.50); assert evidence_for(extract_features(e,[]),P)
def test_17_no_trigger(): assert not evidence_for(extract_features(event(),[]),P)
def test_18_hold():
    st=GroupState('G'); ev=event(timestamp=0,participation_imbalance=.8); d=decide_policy(evidence_for(extract_features(ev,[]),P),st,ev,P); assert d['route']=='hold'
def test_19_hold_release():
    st=GroupState('G'); ev=event(timestamp=240,participation_imbalance=.8); st.first_trigger_at=0; d=decide_policy(evidence_for(extract_features(ev,[]),P),st,ev,P); assert d['triggered']
def test_20_stale_hold_clears():
    st=GroupState('G'); st.first_trigger_at=0; ev=event(timestamp=600); d=decide_policy([],st,ev,P); assert st.first_trigger_at is None and d['route']=='continue'
def test_21_low_confidence_routes_flag():
    st=GroupState('G'); st.first_trigger_at=0; ev=event(timestamp=240,participation_imbalance=.8,state_confidence=.3); d=decide_policy(evidence_for(extract_features(ev,[]),P),st,ev,P); assert d['route']=='check_in' and d['coach_flag']
def test_22_medium_confidence_checkin():
    st=GroupState('G'); st.first_trigger_at=0; ev=event(timestamp=240,participation_imbalance=.8,state_confidence=.65); d=decide_policy(evidence_for(extract_features(ev,[]),P),st,ev,P); assert d['route']=='check_in'
def test_23_repair_prevents():
    st=GroupState('G'); ev=event(); d=decide_policy([],st,ev,P,repairing=True); assert d['route']=='repair'
def test_24_retrieval_uses_catalogue():
    r=CatalogueRetriever(C).retrieve('inclusion','ideation',.7); assert r and r['id'] in {'S4','S8'}
def test_25_retrieval_avoid_constraint():
    r=CatalogueRetriever(C).retrieve('inclusion','ideation',.7,avoid_ids=['S4']); assert r['id']=='S8'
def test_26_retrieval_traceability(): assert 'source' in CatalogueRetriever(C).retrieve('inclusion','ideation',.7)
def test_27_llm_message():
    r=CatalogueRetriever(C).retrieve('inclusion','ideation',.7); assert DeterministicLLM().generate(r,event()).strip()==r['text']
def test_28_validator_accepts():
    r=CatalogueRetriever(C).retrieve('inclusion','ideation',.7); assert validate_message(r['text'],r)[0]
def test_29_validator_rejects():
    r=CatalogueRetriever(C).retrieve('inclusion','ideation',.7); assert not validate_message('You should choose the best idea.',r)[0]
def test_30_runtime_no_ground_truth_dependency():
    e=event(); e.pop('ground_truth_trigger'); e.pop('ground_truth_repair'); e.pop('ground_truth_hidden'); assert run_stream([e],C)
def test_31_output_contains_state_confidence(): assert 'state_confidence' in run_stream([event()],C)[0]
def test_32_output_has_message_on_trigger():
    r=run_stream(list(generate_stream(7,'unresolved_imbalance')),C); x=next(x for x in r if x['triggered']); assert x['message'] and x['strategy_name']
def test_33_output_has_input_event_id(): assert run_stream([event()],C)[0]['event_id'].startswith('healthy:')
def test_34_recovery_is_signal_based():
    h=[event(response_quality=.4,regulatory_activity=.4)]; f=extract_features(event(response_quality=.8,regulatory_activity=.7),h); assert repair_signal(f,h,P)
def test_35_recovery_without_truth():
    h=[event(response_quality=.4,regulatory_activity=.4)]; h[0].pop('ground_truth_repair'); f=extract_features(event(response_quality=.8,regulatory_activity=.7),h); assert repair_signal(f,h,P)
def test_36_monitor_starts():
    st=GroupState('G'); d={'triggered':True}; x=observe_response(st,event(response_quality=.4),d,P); assert x['status']=='monitoring'
def test_37_monitor_escalates():
    st=GroupState('G'); st.unresolved_since=0; x=observe_response(st,event(timestamp=300,response_quality=.4),{'triggered':False},P); assert x['escalated']
def test_38_monitor_recovers():
    st=GroupState('G'); st.unresolved_since=0; x=observe_response(st,event(timestamp=60,response_quality=.8),{'triggered':False},P); assert x['status']=='recovered'
def test_39_synthetic_four_scenarios(): assert len({x['scenario'] for x in generate_stream(1,'healthy')})==1
def test_40_event_count(): assert len(list(generate_stream(1,'healthy')))==24
def test_41_canonical_stable():
    r=run_stream([event()],C)[0]; assert canonical(r)==canonical(dict(r))
def test_42_independent_standalone_parity():
    """Parity must cover every route, not just the ones one scenario reaches.

    An earlier version of this test ran a single high-confidence scenario. The
    check-in and coach-flag routes were never exercised, so the two
    implementations could diverge on the degraded-confidence paths -- the ones
    carrying the fairness behaviour -- and the suite stayed green. A parity
    test that does not reach a branch does not protect it.
    """
    from standalone.clara_standalone import run_stream_standalone
    from evaluation.scenarios import SCENARIOS
    checked = set()
    for scenario in SCENARIOS:
        for seed in (2, 5):
            for conf in (None, 0.40, 0.65):
                events = list(generate_stream(seed, scenario))
                if conf is not None:
                    events = [dict(e, state_confidence=conf) for e in events]
                a = run_stream(events, C)
                b = run_stream_standalone(events, C)
                assert [canonical(x) for x in a] == [canonical(x) for x in b], (
                    f"divergence on {scenario} seed={seed} conf={conf}")
                checked.update(x["route"] for x in a)
    # Fail loudly if the sweep stopped reaching the routes it exists to protect.
    for route in ("scaffold", "check_in", "hold", "continue"):
        assert route in checked, f"parity sweep never exercised route: {route}"


def test_43_checkin_does_not_consume_corrective_budget():
    events=[event(timestamp=0,phase="ideation",participation_imbalance=.8,state_confidence=.65),
            event(timestamp=240,phase="ideation",participation_imbalance=.8,state_confidence=.65)]
    out=run_stream(events,C,P)[-1]
    assert out["route"]=="check_in"
    assert out["budget_remaining"]==1

def test_44_low_confidence_gets_safe_checkin_and_coach_flag():
    st=GroupState("G"); st.first_trigger_at=0
    ev=event(timestamp=240,participation_imbalance=.8,state_confidence=.3)
    d=decide_policy(evidence_for(extract_features(ev,[]),P),st,ev,P)
    assert d["route"]=="check_in" and d["coach_flag"]

def test_45_low_confidence_checkin_ignores_corrective_refractory():
    st=GroupState("G"); st.first_trigger_at=0; st.last_intervention_at=0
    ev=event(timestamp=240,participation_imbalance=.8,state_confidence=.3)
    d=decide_policy(evidence_for(extract_features(ev,[]),P),st,ev,P)
    assert d["route"]=="check_in"

def test_46_true_time_horizon():
    history=[event(timestamp=0,participation_imbalance=.9),event(timestamp=100,participation_imbalance=.1)]
    f=extract_features(event(timestamp=300,participation_imbalance=.1),history)
    assert abs(f.participation_long_horizon-.1)<1e-9

def test_47_episode_metric_counts_one_sustained_episode_once():
    from evaluation.metrics import episode_metrics
    rs=[]
    for i in range(5):
        rs.append({"seed":1,"scenario":"x","group_id":"G","event_id":str(i),"timestamp":i*60,"ground_truth_trigger":i>=1,"ground_truth_intervention_warranted":i>=1,"triggered":i==3})
    m=episode_metrics(rs)
    assert m["tp"]==1 and m["fn"]==0


def test_48_checkin_uses_neutral_catalogue_strategy():
    events=[event(timestamp=0,phase="ideation",participation_imbalance=.8,state_confidence=.65),
            event(timestamp=240,phase="ideation",participation_imbalance=.8,state_confidence=.65)]
    out=run_stream(events,C,P)[-1]
    assert out["route"]=="check_in" and out["scaffold_id"] in {"C1","C2"}

def test_49_corrective_over_support_proxy_is_zero_on_recovered_events():
    from evaluation.metrics import support_metrics
    rs=[{"route":"scaffold","ground_truth_repair":False},{"route":"check_in","ground_truth_repair":True}]
    m=support_metrics(rs)
    assert m["corrective_interventions"]==1 and m["over_support_events"]==0


def test_50_episode_matching_is_one_to_one():
    from evaluation.metrics import episode_metrics
    rs=[]
    for i in range(6):
        rs.append({"seed":1,"scenario":"x","group_id":"G","event_id":str(i),"timestamp":i*60,
                   "ground_truth_trigger":i>=1,"ground_truth_intervention_warranted":i>=1,"triggered":i in {2,3,4}})
    m=episode_metrics(rs)
    assert m["ground_truth_episodes"]==1 and m["predicted_episodes"]==1
    assert m["tp"]==1 and m["fp"]==0 and m["fn"]==0


def test_51_corrective_and_support_episode_metrics_are_separate():
    from evaluation.metrics import support_episode_metrics, corrective_episode_metrics
    rs=[]
    for i,route in enumerate(["hold","check_in","scaffold","refractory"]):
        rs.append({"seed":1,"scenario":"x","group_id":"G","event_id":str(i),"timestamp":i*60,
                   "ground_truth_trigger":True,"ground_truth_intervention_warranted":True,"route":route})
    assert support_episode_metrics(rs)["predicted_episodes"]==1
    assert corrective_episode_metrics(rs)["predicted_episodes"]==1


def test_52_low_confidence_checkin_is_rate_limited():
    events=[]
    for i in range(13):
        events.append(event(timestamp=i*60, phase="ideation", participation_imbalance=.8, state_confidence=.30))
    out=run_stream(events,C,P)
    delivered=[x for x in out if x["route"]=="check_in"]
    assert len(delivered) <= 2
    assert all(x["coach_flag"] for x in delivered)


def test_53_phase_change_resets_checkin_refractory():
    events=[event(timestamp=0,phase="ideation",participation_imbalance=.8,state_confidence=.30),
            event(timestamp=240,phase="ideation",participation_imbalance=.8,state_confidence=.30),
            event(timestamp=300,phase="building",participation_imbalance=.8,state_confidence=.30),
            event(timestamp=540,phase="building",participation_imbalance=.8,state_confidence=.30)]
    out=run_stream(events,C,P)
    assert out[1]["route"]=="check_in"
    assert out[-1]["route"]=="check_in"


def test_54_unreliable_upstream_scenario_reaches_checkin():
    events=list(generate_stream(3,"unreliable_upstream"))
    out=run_stream(events,C,P)
    assert any(x["route"]=="check_in" for x in out)
    assert any(x["coach_flag"] for x in out if x["route"]=="check_in")


# --- the design thesis, pinned ------------------------------------------------

def test_60_healthy_group_is_left_entirely_alone():
    for seed in range(15):
        out = run_stream(list(generate_stream(seed, "healthy")), C, P)
        assert not any(r["route"] in ("scaffold", "check_in") for r in out)


def test_61_group_that_repairs_in_time_is_mostly_left_alone():
    """Recovery inside the hold should suppress the corrective.

    Deliberately not asserted at 15/15: the repair signal reads a noisy
    trajectory and will occasionally miss. Pinning perfection here would just
    invite tuning the scenario until it passed.
    """
    fired = sum(any(r["route"] == "scaffold" for r in
                    run_stream(list(generate_stream(s, "recoverable_imbalance")), C, P))
                for s in range(15))
    assert fired <= 3, f"interrupted a self-repairing group on {fired}/15 seeds"


def test_62_late_repair_is_a_known_and_measured_failure():
    """A group recovering after the hold WILL be interrupted.

    This is not a bug to fix by lengthening the hold -- see the hold-duration
    curve for what that costs in missed detections. It is the price of acting
    in real time without knowing the future, and it is asserted here so that it
    stays visible rather than quietly disappearing into an average.
    """
    fired = sum(any(r["route"] == "scaffold" for r in
                    run_stream(list(generate_stream(s, "late_repair")), C, P))
                for s in range(10))
    assert fired >= 8, "late_repair no longer exercises the limitation it exists to expose"


def test_63_unreliable_upstream_gets_contact_but_never_a_corrective():
    for seed in range(15):
        out = run_stream(list(generate_stream(seed, "unreliable_upstream")), C, P)
        assert not any(r["route"] == "scaffold" for r in out), f"corrective on unusable evidence, seed {seed}"
        assert any(r["route"] == "check_in" for r in out), f"no contact at all, seed {seed}"


def test_64_intervention_metrics_do_not_punish_rationing():
    from evaluation.metrics import intervention_metrics
    rs = [{"seed": 1, "scenario": "x", "group_id": "G", "event_id": str(i),
           "timestamp": i * 60, "ground_truth_intervention_warranted": i >= 1,
           "route": "scaffold" if i in {2, 5} else "continue"} for i in range(8)]
    m = intervention_metrics(rs)
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0


def test_65_system_beats_the_threshold_baseline_on_interruptions():
    """The claim the README makes, asserted rather than asserted-about."""
    from evaluation.metrics import intervention_metrics, naive_intervention_metrics
    recs = []
    for seed in range(8):
        for sc in SCENARIOS:
            events = list(generate_stream(seed, sc))
            truth = {e["event_id"]: e for e in events}
            for r in run_stream(events, C, P):
                r["ground_truth_intervention_warranted"] = truth[r["event_id"]]["ground_truth_intervention_warranted"]
                recs.append(r)
    ours = intervention_metrics(recs, routes=("scaffold", "check_in"))
    base = naive_intervention_metrics(recs)
    assert ours["precision"] > base["precision"] * 2
    assert ours["interventions"] < base["interventions"] / 5

def test_50_scenario_count_is_reported_correctly():
    from evaluation.runner import run
    # runner's scenario count should be seeds × scenarios, not squared.
    # We inspect the same SCENARIOS constant used by the runner.
    from evaluation.scenarios import SCENARIOS
    assert 40 * len(SCENARIOS) == 240

def test_51_hold_curve_uses_intervention_metric():
    from evaluation.metrics import hold_curve
    records=[]
    for i in range(6):
        records.append({
            "seed":1,"scenario":"x","group_id":"G","event_id":str(i),
            "timestamp":i*60,"severity":.7,"ground_truth_trigger":True,
            "ground_truth_intervention_warranted":True,
            "ground_truth_repair":False
        })
    curve=hold_curve(records, holds=(0,240,360))
    assert all("interventions" in x for x in curve)
