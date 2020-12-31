import sys
import inspect

from .util import *
from . import function

available_functions = inspect.getmembers(function, inspect.isclass)

def get_functions(d, continuous=None, convex=None, separable=None,
                  differentiable=None, mutimodal=None, randomized_term=None):

    functions = [cls for clsname, cls in available_functions]
    functions = list(filter(lambda f: f.is_dim_compatible(d), functions))

    functions = list(filter(lambda f: (continuous is None) or (f.continuous == continuous), functions))
    functions = list(filter(lambda f: (convex is None) or (f.convex == convex), functions))
    functions = list(filter(lambda f: (separable is None) or (f.separable == separable), functions))
    functions = list(filter(lambda f: (differentiable is None) or (f.differentiable == differentiable), functions))
    functions = list(filter(lambda f: (mutimodal is None) or (f.mutimodal == mutimodal), functions))
    functions = list(filter(lambda f: (randomized_term is None) or (f.randomized_term == randomized_term), functions))
    return functions
