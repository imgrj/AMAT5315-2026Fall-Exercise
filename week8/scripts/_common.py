"""Shared paths, plotting and machine-readable output for standalone scripts."""
import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
threadpool_limits(4)

for name in ('data','artifacts','evidence'):
    (ROOT/name).mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':140, 'savefig.dpi':170, 'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})


def save_json(name, obj, raw=False):
    path=ROOT/('artifacts' if raw else 'evidence')/name
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n',encoding='utf-8')


def figure(name, fig):
    fig.savefig(ROOT/'evidence'/name,bbox_inches='tight')
    plt.close(fig)


def environment():
    import importlib.metadata as md
    return {k:md.version(k) for k in ['jax','jaxlib','optax','numpy','scipy','matplotlib','sentencepiece','tokenizers','safetensors']}
