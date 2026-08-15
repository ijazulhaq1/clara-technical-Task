from evaluation.metrics import event_metrics

def support_rate(records):
    if not records: return 0.0
    supported=sum(r["route"] in {"scaffold","check_in","coach_flag"} for r in records)
    return supported/len(records)

def breakdown(records):
    speakers=sorted({r["speaker"] for r in records})
    bands=("high","medium","low")
    return {
        "speaker": {s:{**event_metrics([r for r in records if r["speaker"]==s]),
                        "support_rate":support_rate([r for r in records if r["speaker"]==s])} for s in speakers},
        "confidence_band": {
            b:{**event_metrics([r for r in records if r["confidence_band"]==b]),
               "support_rate":support_rate([r for r in records if r["confidence_band"]==b])}
            for b in bands if any(r["confidence_band"]==b for r in records)
        }
    }

def disparity(table, metric="recall"):
    vals=[v[metric] for v in table.values() if v.get("tp",0)+v.get("fn",0)>0]
    return max(vals)-min(vals) if vals else 0.0

def support_disparity(table):
    vals=[v["support_rate"] for v in table.values()]
    return max(vals)-min(vals) if vals else 0.0
