import numpy as np

def objective(x, name, D, d):
    x = np.asarray(x, dtype=float)
    if name not in ("sphere", "ellipsoid") or not 1 <= d <= D or x.shape != (D,):
        raise ValueError("Invalid objective, dimensions or input shape")
    if not np.all(np.isfinite(x)) or np.any(x < -5 - 1e-12) or np.any(x > 5 + 1e-12):
        raise ValueError("Candidate outside bounds or non-finite")
    weights = np.ones(d) if name == "sphere" else np.geomspace(1.0, 100.0, d)
    return float(np.dot(weights, (x[:d] - 1.0)**2))
