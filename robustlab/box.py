"""Exact linear margin minimization over a norm ball intersected with a box."""
from fractions import Fraction
from .core import model, dataset, threat, vector
from .exact import _q, _record


def box_audit(payload, data, bounds, *, norm='linf', epsilon=0.0):
    """Return a feasible primal witness and equal dual bound, using binary rationals."""
    w,b = model(payload)
    x,y = dataset(data,len(w))
    norm,eps = threat(norm,epsilon)
    if norm not in ('l1','linf'):
        raise ValueError('box audit supports only l1 and linf, not l2')
    if not isinstance(bounds,dict) or set(bounds) != {'lower','upper'}:
        raise ValueError('bounds requires exactly lower and upper vectors')
    lower,upper = vector(bounds['lower'],'lower'),vector(bounds['upper'],'upper')
    if len(lower) != len(w) or len(upper) != len(w):
        raise ValueError('bounds dimensions must match weights')
    if any(a>b for a,b in zip(lower,upper)):
        raise ValueError('lower must not exceed upper')
    if any(not a<=v<=b for row in x for a,v,b in zip(lower,row,upper)):
        raise ValueError('each clean row must lie inside the box')
    if len(w)*len(x)>4096:
        raise ValueError('box audit is capped at 4096 row-coordinate products')
    qw,qb,qe = tuple(map(_q,w)),_q(b),_q(eps)
    qlo,qhi = tuple(map(_q,lower)),tuple(map(_q,upper))
    gains = tuple(abs(t) for t in qw)
    order = tuple(sorted(range(len(w)),key=lambda i:(-gains[i],i)))
    rows=[]
    for row,label in zip(x,y):
        qx=tuple(map(_q,row))
        direction=tuple(-label*((t>0)-(t<0)) for t in qw)
        cap=tuple((t-a if s<0 else b-t if s>0 else Fraction(0))
                  for t,a,b,s in zip(qx,qlo,qhi,direction))
        effective=tuple(min(t,qe) for t in cap) if norm=='linf' else cap
        allocated=[Fraction(0)]*len(w)
        lam=Fraction(0)
        if norm=='l1':
            remaining=qe
            lam=max(gains) if qe==0 else Fraction(0)
            for i in order:
                if gains[i]==0 or remaining==0:break
                allocated[i]=min(cap[i],remaining)
                remaining-=allocated[i]
                if allocated[i]>0 and remaining==0:lam=gains[i]
        else:
            allocated=list(effective)
        mu=tuple(max(Fraction(0),g-lam) for g in gains)
        primal=sum(g*t for g,t in zip(gains,allocated))
        dual=lam*qe+sum(c*m for c,m in zip(effective,mu))
        delta=tuple(s*t for s,t in zip(direction,allocated))
        attacked=tuple(t+d for t,d in zip(qx,delta))
        score=sum(g*t for g,t in zip(qw,qx))+qb
        clean=label*score
        worst=clean-primal
        attained=label*(sum(g*t for g,t in zip(qw,attacked))+qb)
        length=sum(abs(t) for t in delta) if norm=='l1' else max(abs(t) for t in delta)
        if (primal!=dual or attained!=worst or length>qe or
            any(not a<=t<=b for a,t,b in zip(qlo,attacked,qhi))):
            raise ArithmeticError('box witness or primal/dual identity failed')
        # The old all-real minimizer, clipped coordinate-wise, is only a candidate.
        raw=[Fraction(0)]*len(w)
        if norm=='linf':raw=[s*qe for s in direction]
        else:raw[order[0]]=direction[order[0]]*qe
        clipped=tuple(min(b,max(a,t+d)) for a,t,d,b in zip(qlo,qx,raw,qhi))
        clipped_margin=label*(sum(g*t for g,t in zip(qw,clipped))+qb)
        unbounded=clean-qe*(sum(gains) if norm=='linf' else max(gains))
        rows.append({'x':list(map(_record,qx)), 'label':label,
                     'score':_record(score),'clean_signed_margin':_record(clean),
                     'worst_signed_margin':_record(worst),'attained_signed_margin':_record(attained),
                     'strict_robust_correct':worst>0,'prediction':1 if score>=0 else -1,
                     'delta':list(map(_record,delta)),'attacked_x':list(map(_record,attacked)),
                     'perturbation_norm':_record(length),
                     'directional_capacity':list(map(_record,cap)),
                     'certificate':{'allocation':list(map(_record,allocated)),
                                    'effective_upper':list(map(_record,effective)),
                                    'budget_multiplier':_record(lam),
                                    'upper_multipliers':list(map(_record,mu)),
                                    'primal_gain':_record(primal),'dual_gain_bound':_record(dual),
                                    'gap':_record(dual-primal)},
                     'all_real_worst_signed_margin':_record(unbounded),
                     'clipped_old_attack_margin':_record(clipped_margin),
                     'clipped_old_attack_is_optimal':clipped_margin==worst})
    return {'format':'robustlab-box-v1',
            'input_semantics':'exact stored binary float values, not decimal spelling',
            'model':{'version':1,'weights':list(map(_record,qw)),'bias':_record(qb)},
            'bounds':{'lower':list(map(_record,qlo)),'upper':list(map(_record,qhi))},
            'threat':{'norm':norm,'epsilon':_record(qe),'domain':'ball intersected with the supplied closed box'},
            'rows':rows,'counts':{'total':len(rows),'strict_robust_correct':sum(r['strict_robust_correct'] for r in rows)},
            'certificate_rule':'L1: g_i <= lambda + mu_i, nonnegative multipliers; gain <= lambda*epsilon + sum(cap_i*mu_i). Linf: independent effective caps min(cap_i,epsilon), lambda=0.',
            'boundary_rule':'strict worst signed margin > 0; score tie predicts +1',
            'scope':'supplied linear model and mathematical box threat only; not a separately rounded execution pipeline or neural-network certificate'}


def box_demo():
    m={'version':1,'weights':[4,1],'bias':-.25}
    d={'x':[[.0625,.75]],'y':[1]}
    bounds={'lower':[0,0],'upper':[1,1]}
    cases=[{'epsilon':e,'norm':norm,'audit':box_audit(m,d,bounds,norm=norm,epsilon=e)}
           for norm in ('l1','linf') for e in (0,.125,.25,.5,.625,1,2)]
    bad=next(c['audit']['rows'][0] for c in cases if c['norm']=='l1' and c['epsilon']==.625)
    assert _decode(bad['worst_signed_margin'])==Fraction(-1,16)
    assert _decode(bad['clipped_old_attack_margin'])==Fraction(1,2)
    assert not bad['clipped_old_attack_is_optimal']
    assert any(c['audit']['rows'][0]['strict_robust_correct'] for c in cases)
    assert any(not c['audit']['rows'][0]['strict_robust_correct'] for c in cases)
    return {'model':m,'data':d,'bounds':bounds,'cases':cases,
            'selection':'first hand-specified dyadic two-coordinate example; no parameter search or prevalence estimate',
            'scope':'clipping a feasible attack is not generally an L1 optimum; a positive candidate margin cannot prove robustness'}


def _decode(record):
    return Fraction(int(record['numerator']),int(record['denominator']))
