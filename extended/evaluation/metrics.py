TRUTH_KEY = "ground_truth_intervention_warranted"


from collections import defaultdict


def event_metrics(records, prediction_key="triggered", truth_key=TRUTH_KEY):
    tp=fp=fn=tn=0
    for r in records:
        y=bool(r[truth_key])
        p=bool(r.get(prediction_key, False))
        if y and p: tp+=1
        elif not y and p: fp+=1
        elif y and not p: fn+=1
        else: tn+=1
    return {
        "denominator":"event", "tp":tp,"fp":fp,"fn":fn,"tn":tn,
        "n_events":tp+fp+fn+tn,
        "precision":tp/(tp+fp) if tp+fp else 0.0,
        "recall":tp/(tp+fn) if tp+fn else 0.0
    }


def _runs(records, key):
    """Return contiguous true runs, grouped by scenario/group/seed."""
    runs=[]
    groups=defaultdict(list)
    for r in records:
        groups[(r["seed"],r["scenario"],r["group_id"])].append(r)
    for group, rs in groups.items():
        rs=sorted(rs,key=lambda x:x["timestamp"])
        current=[]
        prev_t=None
        for r in rs:
            flag=bool(r[key])
            contiguous=prev_t is not None and r["timestamp"]-prev_t <= 60
            if flag and (not current or contiguous):
                current.append(r)
            elif flag:
                if current: runs.append(current)
                current=[r]
            elif current:
                runs.append(current); current=[]
            prev_t=r["timestamp"]
        if current: runs.append(current)
    return runs


def _episode_metrics_from_runs(gt, pred):
    """One-to-one episode matching by shared event IDs.

    Each ground-truth episode can match at most one predicted episode and vice
    versa. This prevents overlapping predictions from the same episode from
    being counted as additional true positives.
    """
    gt_sets=[{r["event_id"] for r in g} for g in gt]
    pred_sets=[{r["event_id"] for r in p} for p in pred]
    edges=[]
    for pi, ps in enumerate(pred_sets):
        for gi, gs in enumerate(gt_sets):
            if ps & gs:
                overlap=len(ps & gs)
                edges.append((overlap, pi, gi))
    matched_pred=set(); matched_gt=set()
    for overlap, pi, gi in sorted(edges, reverse=True):
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi); matched_gt.add(gi)
    tp=len(matched_gt)
    fn=len(gt)-tp
    fp=len(pred)-len(matched_pred)
    return {
        "denominator":"episode", "tp":tp,"fp":fp,"fn":fn,
        "ground_truth_episodes":len(gt), "predicted_episodes":len(pred),
        "precision":tp/(tp+fp) if tp+fp else 0.0,
        "recall":tp/(tp+fn) if tp+fn else 0.0
    }


def episode_metrics(records, prediction_key="triggered", truth_key=TRUTH_KEY):
    """Evaluate whether a predicted episode overlaps each true episode once."""
    gt=_runs(records,truth_key)
    pred=_runs(records,prediction_key)
    return _episode_metrics_from_runs(gt,pred)


def support_episode_metrics(records):
    """Episode-level metrics for any support (corrective scaffold or check-in)."""
    flagged=[{**r,"support_event":r.get("route") in {"scaffold","check_in"}} for r in records]
    return episode_metrics(flagged,"support_event")


def corrective_episode_metrics(records):
    """Episode-level metrics for corrective scaffolds only."""
    flagged=[{**r,"corrective_event":r.get("route")=="scaffold"} for r in records]
    return episode_metrics(flagged,"corrective_event")


def naive_metrics(records):
    pred=[]
    for r in records:
        score=max(r["participation_imbalance"],1-r["uptake"],1-r["goal_convergence"],1-r["regulatory_activity"])
        pred.append({**r,"triggered":score>=.48})
    return {
        "event":event_metrics(pred),
        "episode":episode_metrics(pred),
        "support_episode":episode_metrics(pred)
    }


def support_metrics(records):
    corrective=[r for r in records if r["route"]=="scaffold"]
    safe=[r for r in records if r["route"]=="check_in"]
    over=sum(bool(r.get("ground_truth_repair")) for r in corrective)
    return {
        "corrective_interventions":len(corrective),
        "safe_checkins":len(safe),
        "over_support_events":over,
        "over_support_rate":over/len(corrective) if corrective else 0.0
    }


def repair_metrics(records):
    recovered=[r for r in records if r["ground_truth_repair"]]
    unresolved=[r for r in records if not r["ground_truth_repair"] and r["ground_truth_trigger"]]
    return {
        "recoverable_events":len(recovered),
        "unresolved_trigger_events":len(unresolved),
        "recovery_annotations":sum(r["annotation_type"]=="repair" for r in recovered),
        "escalations":sum(r["escalated"] for r in unresolved)
    }


def hold_curve(records, holds=(0,60,120,180,240,300,360)):
    """Sensitivity of the deterministic persistence rule to hold duration.

    This is a post-hoc diagnostic over the recorded severity stream. It creates
    hypothetical corrective acts when severity remains above the candidate
    threshold continuously for the requested hold. It uses the same
    warranted-episode point scoring as the headline intervention metric.
    """
    out=[]
    ordered=sorted(records, key=lambda r:(r["seed"],r["scenario"],r["group_id"],r["timestamp"]))
    for h in holds:
        pred=[]
        first={}
        prev={}
        for r in ordered:
            g=(r["seed"],r["scenario"],r["group_id"])
            candidate=r["severity"]>=0.48
            contiguous = (
                g in prev and candidate and
                r["timestamp"]-prev[g] <= 90
            )
            if candidate and (g not in first or not contiguous):
                first[g]=r["timestamp"]
            if not candidate:
                first.pop(g,None)
            fired=bool(candidate and g in first and
                       r["timestamp"]-first[g] >= h and
                       r["severity"]>=0.58)
            pred.append({**r, "route":"scaffold" if fired else "continue"})
            prev[g]=r["timestamp"]
        em=intervention_metrics(pred, routes=("scaffold",), truth_key=TRUTH_KEY)
        out.append({"hold_seconds":h,"precision":em["precision"],
                    "recall":em["recall"],"interventions":em["interventions"]})
    return out


def intervention_metrics(records, routes=("scaffold",), truth_key=TRUTH_KEY):
    """Score interventions as POINTS against warranted episodes.

    Not run-vs-run matching. A policy that rations itself -- one corrective per
    CBL phase -- produces two isolated interventions inside one long difficulty.
    Run-based matching reads those as two predicted episodes: one true positive,
    one false alarm. A detector firing on every window forms a single contiguous
    run and is charged nothing. That scores the noisier system higher, which
    inverts the property this design exists to have.

        TP = warranted episodes containing at least one intervention
        FN = warranted episodes containing none
        FP = interventions landing outside every warranted episode

    Two interventions inside one genuine difficulty are not a false alarm. They
    may be over-support, which support_metrics() measures separately.
    """
    episodes = _runs(records, truth_key)
    spans = []
    for ep in episodes:
        key = (ep[0]["seed"], ep[0]["scenario"], ep[0]["group_id"])
        spans.append((key, min(r["timestamp"] for r in ep), max(r["timestamp"] for r in ep)))
    acts = [r for r in records if r.get("route") in routes]
    covered, fp = set(), 0
    for a in acts:
        key = (a["seed"], a["scenario"], a["group_id"])
        hit = [i for i, (k, lo, hi) in enumerate(spans)
               if k == key and lo <= a["timestamp"] <= hi]
        if hit:
            covered.update(hit)
        else:
            fp += 1
    tp = len(covered); fn = len(spans) - tp
    return {"denominator": "warranted_episode", "tp": tp, "fp": fp, "fn": fn,
            "warranted_episodes": len(spans), "interventions": len(acts),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}


def naive_intervention_metrics(records, truth_key=TRUTH_KEY):
    """The same point-based scoring applied to the threshold baseline.

    The baseline has no budget, so every window above threshold is an act, and
    it is charged for each interruption it would actually have delivered.
    """
    pred = []
    for r in records:
        score = max(r["participation_imbalance"], 1 - r["uptake"],
                    1 - r["goal_convergence"], 1 - r["regulatory_activity"])
        pred.append({**r, "route": "scaffold" if score >= .48 else "continue"})
    return intervention_metrics(pred, truth_key=truth_key)
