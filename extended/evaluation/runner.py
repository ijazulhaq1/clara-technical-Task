import json
from pathlib import Path
from main import run_stream
from src.config import load_catalogue, load_policy
from evaluation.scenarios import generate_stream, SCENARIOS
from evaluation.metrics import intervention_metrics, naive_intervention_metrics, event_metrics, episode_metrics, support_episode_metrics, corrective_episode_metrics, naive_metrics, repair_metrics, support_metrics, hold_curve
from evaluation.fairness import breakdown, disparity, support_disparity
from evaluation.visualization import make_timeline

def run():
    c=load_catalogue(); p=load_policy(); records=[]
    for seed in range(40):
        for scenario in SCENARIOS:
            events=list(generate_stream(seed,scenario)); run_records=run_stream(events,c,p); truth={e["event_id"]:e for e in events}
            for r in run_records:
                e=truth[r["event_id"]]; r["ground_truth_trigger"]=bool(e["ground_truth_trigger"]); r["ground_truth_repair"]=bool(e["ground_truth_repair"]); r["ground_truth_hidden"]=bool(e["ground_truth_hidden"]); r["ground_truth_intervention_warranted"]=bool(e["ground_truth_intervention_warranted"])
            records.extend(run_records)
    fairness=breakdown(records); result={"scenario_runs":40*len(SCENARIOS),"seeds":40,"events":len(records),"primary_metrics":intervention_metrics(records),"any_support_metrics":intervention_metrics(records,routes=("scaffold","check_in")),"naive_baseline_pointwise":naive_intervention_metrics(records),"legacy_episode_metrics":episode_metrics(records),
    "support_episode_metrics":support_episode_metrics(records),
    "corrective_episode_metrics":corrective_episode_metrics(records),"event_level_diagnostic":event_metrics(records),"naive_baseline":naive_metrics(records),"repair_and_escalation":repair_metrics(records),"support_and_over_support":support_metrics(records),"hold_duration_curve":hold_curve(records),"fairness":fairness,"fairness_recall_disparity_by_speaker":disparity(fairness["speaker"]),"fairness_recall_disparity_by_confidence":disparity(fairness["confidence_band"]),"fairness_support_disparity_by_speaker":support_disparity(fairness["speaker"]),"fairness_support_disparity_by_confidence":support_disparity(fairness["confidence_band"]),"synthetic_data":True,"interpretation":"Validation-harness results only; not classroom evidence."}
    out=Path("outputs"); out.mkdir(exist_ok=True)
    (out/"evaluation_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (out/"annotations.jsonl").open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,sort_keys=True)+"\n")
    make_timeline(out / "timeline.png")
    return result
if __name__=="__main__": print(json.dumps(run(),indent=2))
