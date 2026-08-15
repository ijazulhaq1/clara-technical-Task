from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def run_stream_standalone(events,c):
    """Small independent parity implementation of the core decision path."""
    docs=c['strategies']; vec=TfidfVectorizer(ngram_range=(1,2)); mat=vec.fit_transform([
        d['function']+' '+d['name']+' '+d['goal']+' '+d['keywords'] for d in docs])
    out=[]; first=None; last=None; budget={}; phase=None; history=[]
    ci_last=None; ci_budget={}
    for e in events:
        if e['phase']!=phase:
            phase=e['phase']; first=None; ci_last=None; budget[phase]=c['phase_budget']; ci_budget[phase]=c['check_in_budget_per_phase']
        cutoff=e['timestamp']-270
        recent=[x for x in history if x['timestamp']>=cutoff]
        ib=sum(x['participation_imbalance'] for x in recent)/max(1,len(recent))
        scores={'inclusion':max(e['participation_imbalance'],ib),
                'shared_task_understanding':1-e['goal_convergence'],
                'progress_monitoring':1-e['uptake'],
                'socioemotional_monitoring':1-e['regulatory_activity']}
        thresholds={'inclusion':c['thresholds']['participation_imbalance'],
                    'shared_task_understanding':c['thresholds']['goal_divergence'],
                    'progress_monitoring':c['thresholds']['low_uptake'],
                    'socioemotional_monitoring':c['thresholds']['regulatory_gap']}
        active=[(f,s) for f,s in scores.items() if s>=thresholds[f]]
        hidden_score=(.30*e['participation_imbalance']+.25*(1-e['uptake'])+.25*(1-e['goal_convergence'])+.20*(1-e['regulatory_activity']))
        if not active and hidden_score>=c['thresholds']['hidden_difficulty']:
            active=[('progress_monitoring',hidden_score)]
        active.sort(key=lambda z:z[1],reverse=True)
        function=active[0][0] if active else None; severity=active[0][1] if active else 0
        band='high' if e['state_confidence']>=c['confidence']['high'] else ('medium' if e['state_confidence']>=c['confidence']['medium'] else 'low')
        route='continue'; triggered=False; message=None; sid=None; coach_flag=False
        repairing=False
        if history:
            prior=history[-3:]; pq=sum(x['response_quality'] for x in prior)/len(prior); pr=sum(x['regulatory_activity'] for x in prior)/len(prior)
            prior_problem=any(x['response_quality']<.60 or x['regulatory_activity']<.55 or x['participation_imbalance']>.55 for x in prior)
            repairing=(prior_problem and e['response_quality']>=c['repair']['quality_threshold'] and e['regulatory_activity']>=c['repair']['regulatory_threshold'] and (e['response_quality']-pq>=c['repair']['quality_improvement'] or e['regulatory_activity']-pr>=c['repair']['regulatory_improvement']))
        if repairing:
            first=None; route='repair'; action='continue'
        elif not active:
            first=None; route='continue'; action='continue'
        else:
            if first is None: first=e['timestamp']
            held=e['timestamp']-first>=c['hold_seconds']
            if not held:
                route='hold'; action='monitor'
            elif band in ('low','medium'):
                # Mirrors src/policy.py: check-ins avoid the corrective budget
                # but carry their own, so a poorly-heard group is not contacted
                # every single window.
                ci_refr = ci_last is not None and e['timestamp']-ci_last < c['check_in_refractory_seconds']
                if ci_refr:
                    route='check_in_refractory'; action='monitor'; coach_flag=(band=='low')
                elif ci_budget.get(phase,c['check_in_budget_per_phase'])<=0:
                    route='check_in_budget_exhausted'; action='monitor'; coach_flag=(band=='low')
                else:
                    route='check_in'; action='check_in'; triggered=True; coach_flag=(band=='low')
                    function='any'   # check-ins retrieve from the neutral pool
            else:
                refr=last is not None and e['timestamp']-last<c['refractory_seconds']
                if refr: route='refractory'; action='monitor'
                elif budget[phase]<=0: route='budget_exhausted'; action='monitor'
                else: route='scaffold'; action='scaffold'; triggered=True
            if triggered:
                pool=[(i,d) for i,d in enumerate(docs) if (d['function']==function or d['function']=='any') and phase in d['phases'] and d['severity_range'][0]<=severity<=d['severity_range'][1]]
                if pool:
                    q=vec.transform([function+' '+phase]); sc=cosine_similarity(q,mat).ravel(); i,d=max(pool,key=lambda z:sc[z[0]])
                    sid=d['id']; message=d['text']
                    if route=='scaffold':
                        budget[phase]-=1; last=e['timestamp']; first=None
                    else:
                        ci_budget[phase]=ci_budget.get(phase,c['check_in_budget_per_phase'])-1; ci_last=e['timestamp']
        out.append({'event_id':e['event_id'],'timestamp':e['timestamp'],'phase':phase,'function':function,'triggered':triggered,'route':route,'action':action,'confidence_band':band,'scaffold_id':sid,'message_valid':True,'monitor_status':'monitoring','escalated':False,'annotation_type':None})
        history.append(e)
    return out
