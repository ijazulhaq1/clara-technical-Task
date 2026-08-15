# Notes on Part A

Ijaz Ul Haq

`clara_part_a.py` answers the brief: it reads a stream of classified states (1),
works out when an intervention is warranted (2), and picks the message to send
(3). `extended/` is a modular version of the same ideas with tests, a fairness
audit and seed sweeps. The brief asked for a short script, so that is the
submission. I included the larger version because I think the evaluation is
where the real difficulty in this problem sits, not because it was asked for.

## The idea behind it

A trigger is not a state, it is a trajectory without repair.

If you detect negative states and act on them, you get an agent that interrupts
every group having a bad five minutes, including the ones about to sort
themselves out. Two things in the collaborative learning literature push against
that. Socio-cognitive and socio-emotional signals are coupled, so a group that
gets frustrated and then renegotiates is regulating itself while a group that
gets frustrated and carries on is not, and the signal is identical in both
cases. And when groups fail, the usual reason is that nobody noticed they needed
help, so a quiet stalled group needs its own detection path rather than falling
through the gaps.

Both point the same way. Most of the work in an agent like this is in staying
quiet, and three of the four checks between evidence and action are there to
stop an intervention rather than start one.

## Results

| | precision | recall | interruptions |
|---|---:|---:|---:|
| This system | 0.74 | 1.00 | 174 |
| Threshold baseline | 0.18 | 1.00 | 1,837 |

The baseline finds every real problem. It does it by interrupting about ten
times as often, mostly groups that needed nothing.

There are two labels in the synthetic data rather than one: whether the group is
in difficulty, and whether intervening was the right thing to do. They come
apart in the case this design is built around, where a group in difficulty
recovers on its own. An earlier version of this evaluation only had the first
label, which made a plain feature threshold the best possible system by
definition and reported the baseline beating everything I had built.

## Choices worth explaining

**Rules rather than a trained model.** There is no annotated corpus yet, since
the PhD candidate produces it during the project, so there is nothing to train
on. A rule that the education supervisors can read and argue with is also easier
to audit than a fitted threshold. The rules name regulation functions rather
than emotions, so the taxonomy can be swapped later by editing one mapping.

**A four minute wait before acting.** A candidate is held and looked at again
rather than acted on immediately. When trouble starts you cannot tell a rough
patch from a group that is properly stuck, and they only separate once you can
see whether anything changed.

**Three ways to route on confidence rather than an on/off switch.** A switch has
a problem hidden in it. A group the pipeline struggles to hear, because of
accents or crosstalk or one quiet member, would sit under the threshold all
session and quietly receive less help than everyone else. So low confidence
weakens what gets sent instead of stopping it. Check-ins do not come out of the
corrective budget, but they have a limit of their own, because without one a
group on poor audio ends up messaged more often than anybody else.

**Nothing is sent without being checked.** The validator rejects messages that
give away the answer, tell the group what it is feeling, single out one person,
or are not phrased as a question. The model never decides whether to intervene,
and by default it does not decide what to say either.

## What it gets wrong

`late_repair` is a group that recovers after the wait has expired. The system
interrupts it every time. A longer wait would fix that case and would miss real
problems elsewhere, and at t=540 there is no way to know what the group does at
t=900. I left the scenario in so the cost shows up in the results rather than
sitting in a caveat.

## Other limitations

The data is synthetic throughout. The scenarios are my own assumptions about
what these situations look like, and real groups will do things I have not
thought of, so the generator is the first thing to throw away once there is
classroom data. The thresholds come from looking at the noise in the feature
distributions rather than from fitting anything. The taxonomy is provisional.
The precision and recall figures say the detector matches my assumptions, not
that the assumptions are correct.

## Where this would go in CLARA

The rule layer takes the real taxonomy through the same mapping. Catalogue
selection becomes retrieval over the actual coaching material, then fine-tuning
and preference alignment against experienced coaches. The planted labels become
held-out classroom sessions through the same harness. The audit becomes the
fairness validation work, broken down by group and by speaker. In `extended/`
escalations already get written to a queue, and those cases, where the agent
acted and the group did not improve, are the most useful labelled data the
project will produce. Collecting them from the start is how the training set
gets built.
