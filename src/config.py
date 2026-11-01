from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Settings:
    budget: int = 100
    seeds: tuple = (0, 1, 2, 3, 4)
    lower: float = -5.0
    upper: float = 5.0
    initial_points: int = 8
    candidates: int = 512
    local_starts: int = 2
    local_maxiter: int = 35
    hyper_interval: int = 10
    alpha: float = 1e-6
    xi: float = 0.0
    kernel: str = "ard"
    cma_sigma: float = 2.0

EXPERIMENTS = {
    "sphere_full": ("sphere", [(d,d) for d in (2,5,10,20)]),
    "ellipsoid_full": ("ellipsoid", [(d,d) for d in (2,5,10,20)]),
    "sphere_effective": ("sphere", [(20,d) for d in (2,5,10,20)]),
    "ellipsoid_effective": ("ellipsoid", [(20,d) for d in (2,5,10,20)]),
}
