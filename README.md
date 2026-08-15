# CLARA technical challenge, Part A

Ijaz Ul Haq

Finding the moments in a group session where an AI agent should step in, and
choosing what it says.

## Running it

```bash
python3 clara_part_a.py
```

Nothing to install, no network access needed, no API key. It takes about a
second and gives the same numbers each time you run it.

The file is 463 lines, but only 218 of those are code. The rest is comments
explaining the choices, which the brief asked for, plus blank lines. The three
numbered requirements are marked in the file as `(1)`, `(2)` and `(3)`.

## What is in here

| | |
|---|---|
| `clara_part_a.py` | The answer to the brief. |
| `NOTES.md` | A page on why it works the way it does, and where it falls short. |
| `extended/` | A modular version of the same ideas with 62 tests, a fairness audit and seed sweeps. Not the same code line for line: ten catalogue strategies rather than six, and the features, triggers, policy, retrieval and monitoring split across modules. |

```bash
cd extended
pip install -r requirements.txt
python -m pytest tests/ -q      # 62 tests
python -m evaluation.runner     # 40 seeds across 6 scenarios
```

## Results

| | precision | recall | interruptions |
|---|---:|---:|---:|
| This system | 0.74 | 1.00 | 174 |
| Threshold baseline | 0.18 | 1.00 | 1,837 |

The baseline finds every real problem, so recall was never the difficult part.
It gets there by interrupting roughly ten times as often, and most of those
messages go to groups that were fine.

## The idea

A trigger is not a state, it is a trajectory without repair.

A group that gets frustrated and then renegotiates its plan is managing itself.
A group that gets frustrated and carries on unchanged is not. The signal looks
the same in both cases. Because the main risk with this kind of agent is
over-regulating a group, three of the four checks between having evidence and
sending a message exist to stop it happening.
