"""Lightweight tests for newly added follow-up dispatch and validators."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job_counts():
    runner = load('run_experiments')
    assert len(list(runner.tasks('budget', 'sphere_full'))) == 80
    assert len(list(runner.tasks('robustness', 'sphere_effective'))) == 20
    assert len(list(runner.tasks('extension', 'ellipsoid_full'))) == 40


def test_all_experiments_defined():
    runner = load('run_experiments')
    assert set(runner.PLAN['experiments']) == set(runner.EXPERIMENTS)


def test_no_existing_result_is_required():
    validator = load('validate_results')
    errors, n = validator.check_folder(HERE / 'no_such_directory')
    assert n == 0 and errors
