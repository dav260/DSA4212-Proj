import numpy as np

def optimize(rec, cfg, seed):
    import cma
    es = cma.CMAEvolutionStrategy(np.zeros(rec.D), cfg.cma_sigma, {
        'seed': int(seed)+1, 'bounds': [cfg.lower,cfg.upper],
        'verb_disp': 0, 'verbose': -9, 'verb_log': 0,
        'maxfevals': rec.budget, 'tolx': 0, 'tolfun': 0,
        'tolfunhist': 0, 'tolstagnation': 0, 'tolflatfitness': 0,
    })
    population = int(es.popsize)
    reason = 'budget_exhausted'
    while len(rec.rows) < rec.budget:
        stops = es.stop()
        if stops:
            reason = 'cma_stop:' + ','.join(sorted(stops))
            break
        remaining = rec.budget-len(rec.rows)
        # Never tell a partial generation: record it, then terminate without updating CMA.
        points = es.ask()
        count = min(len(points), remaining)
        values = [rec.evaluate(x) for x in points[:count]]
        if count < len(points):
            reason = 'budget_exhausted_partial_generation_no_tell'
            break
        es.tell(points, values)
    return {'termination_reason':reason, 'population_size':population,
            'completed_generations':int(es.countiter)}
