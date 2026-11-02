import json
import numpy as np
import pytest
from src.objectives import objective
from src.bo import expected_improvement
from src.config import Settings
from src.runner import run_one

@pytest.mark.parametrize('name', ['sphere','ellipsoid'])
@pytest.mark.parametrize('D,d',[(2,2),(20,2),(20,5),(20,10),(20,20)])
def test_objective(name,D,d):
    x=np.ones(D); assert objective(x,name,D,d)==0
    x[:d]=2
    assert np.isclose(objective(x,name,D,d),d if name=='sphere' else np.geomspace(1,100,d).sum())
    x[d:]=4;assert objective(x,name,D,d)==objective(np.r_[np.full(d,2.),np.ones(D-d)],name,D,d)

def test_ei():
    z=expected_improvement(np.array([0.,1.]),np.array([0.,1.]),0.)
    assert z[0]==0 and z[1]>0

def test_bo_exact_and_deterministic(tmp_path):
    cfg=Settings(budget=9,candidates=16,local_starts=1,local_maxiter=2)
    a=run_one('test','sphere',2,2,'bo',0,cfg,tmp_path/'a')
    b=run_one('test','sphere',2,2,'bo',0,cfg,tmp_path/'b')
    assert a['evaluations']==b['evaluations']==9
    fa=next((tmp_path/'a').glob('*.csv'));fb=next((tmp_path/'b').glob('*.csv'))
    import pandas as pd
    x=pd.read_csv(fa);y=pd.read_csv(fb)
    assert x[['x','value','best_so_far']].equals(y[['x','value','best_so_far']])
    assert (np.diff(x.best_so_far)<=0).all()

def test_cma_exact_if_installed(tmp_path):
    pytest.importorskip('cma')
    cfg=Settings(budget=11)
    a=run_one('test','ellipsoid',5,2,'cma',0,cfg,tmp_path)
    assert a['evaluations']==11
    assert a['population_size']>=4
