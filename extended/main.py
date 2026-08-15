from src.engine import run_stream, canonical
from src.config import load_catalogue, load_policy

__all__ = ["run_stream", "canonical"]

if __name__ == "__main__":
    import json
    from pathlib import Path
    input_path=Path("data/sample_stream.jsonl")
    events=[json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records=run_stream(events, load_catalogue(), load_policy())
    out=Path("outputs"); out.mkdir(exist_ok=True)
    with (out/"interventions.jsonl").open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"Processed {len(records)} events -> {out/'interventions.jsonl'}")
