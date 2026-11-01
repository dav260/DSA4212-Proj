import warnings
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

def expected_improvement(mu, sd, incumbent, xi=0.0):
    improvement = incumbent - mu - xi
    z = np.divide(improvement, sd, out=np.zeros_like(improvement), where=sd>1e-14)
    return np.where(sd>1e-14, improvement*norm.cdf(z)+sd*norm.pdf(z), 0.0)

def optimize(rec, cfg, seed):
    D = rec.D
    rng = np.random.default_rng(seed)
    sobol = qmc.Sobol(d=D, scramble=True, rng=np.random.default_rng(seed))
    X = sobol.random_base2(m=3)
    Y = [rec.evaluate(cfg.lower+(cfg.upper-cfg.lower)*x) for x in X]
    length = np.ones(D) if cfg.kernel == 'ard' else 1.0
    kernel = ConstantKernel(1.0, (1e-3, 1e3))*Matern(length_scale=length,
                      length_scale_bounds=(1e-2, 1e2), nu=2.5)
    optimized_at = []
    while len(Y) < rec.budget:
        n = len(Y)
        # Full conditioning at EVERY iteration. Hyperparameter MLE only at n=8, 18, 28, ...
        do_hyper = n == cfg.initial_points or (n-cfg.initial_points) % cfg.hyper_interval == 0
        gp = GaussianProcessRegressor(kernel=kernel, alpha=cfg.alpha, normalize_y=True,
                 optimizer='fmin_l_bfgs_b' if do_hyper else None,
                 n_restarts_optimizer=0, random_state=seed)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            gp.fit(np.asarray(X), np.asarray(Y))
        kernel = gp.kernel_  # warm-start kernel, never reuse a stale posterior
        if do_hyper: optimized_at.append(n)
        pool = rng.random((cfg.candidates,D))
        mu, sd = gp.predict(pool, return_std=True)
        ei = expected_improvement(mu, sd, min(Y), cfg.xi)
        best_index = int(np.argmax(ei))
        best_x, best_ei = pool[best_index].copy(), float(ei[best_index])
        for idx in np.argsort(ei)[-cfg.local_starts:]:
            def negative_ei(z):
                m, st = gp.predict(np.asarray(z)[None,:], return_std=True)
                return -float(expected_improvement(m, st, min(Y), cfg.xi)[0])
            res = minimize(negative_ei, pool[idx], method='L-BFGS-B',
                           bounds=[(0,1)]*D, options={'maxiter':cfg.local_maxiter})
            if res.success and -res.fun > best_ei:
                best_x, best_ei = res.x.copy(), -float(res.fun)
        # Deterministic non-duplicate fallback for degenerate EI.
        if np.min(np.linalg.norm(np.asarray(X)-best_x,axis=1)) < 1e-9:
            for z in pool[np.argsort(ei)[::-1]]:
                if np.min(np.linalg.norm(np.asarray(X)-z,axis=1)) >= 1e-9:
                    best_x=z.copy(); break
        Y.append(rec.evaluate(cfg.lower+(cfg.upper-cfg.lower)*best_x))
        X = np.vstack((X,best_x))
    return {'termination_reason':'budget_exhausted', 'hyperparameter_refits_at_n':optimized_at}
