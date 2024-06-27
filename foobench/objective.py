""" Objective functions for the Evolutionary Strategies example,
taken from here: https://en.wikipedia.org/wiki/Test_functions_for_optimization

Test functions for optimization. (2023, December 29). In Wikipedia. https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import torch
from foobench import foos
from typing import Union
import json
from functools import partial


def apply_limits(f, x, val=0., limits=()):
    if not limits:
        return f

    if not hasattr(limits, '__len__'):
        limits = [limits] * x.shape[1]

    elif len(limits) == 1:
        limits = [limits[0]] * x.shape[1]

    limits = torch.tensor(limits, device=x.device)
    # check if limits are tuples
    if len(limits.shape) == 2:
        lower_bound = limits[:, 0]
        upper_bound = limits[:, 1]

    else:
        lower_bound = -limits
        upper_bound = limits

    exceeding = torch.logical_or(x < lower_bound[None, :], (x > upper_bound[None, :]))
    f[exceeding.any(dim=1)] = val
    return f


class Objective:
    def __init__(self, foo: Union[str, callable] = "rastrigin", dim=2, maximize=False,
                 foo_module=None, foo_kwargs=(),
                 limits: Union[int, tuple] = 4, apply_limits=False, limit_val=0.,):
        self.dim = dim
        self.foo_module = foo_module
        self._foo = None
        self.foo = foo
        self.foo_kwargs = foo_kwargs or {}
        self.maximize = maximize
        self.limits = limits
        self.apply_limits = apply_limits
        self.limit_val = limit_val

    @property
    def foo(self):
        return self._foo

    @foo.setter
    def foo(self, foo):
        if hasattr(foo, '__call__'):
            self._foo = foo
        else:
            if self.foo_module is None:
                self._foo = getattr(foos, foo)
            else:
                # locate the module
                from pydoc import locate
                module = locate(self.foo_module)
                self._foo = getattr(module, foo)

    @property
    def foo_name(self):
        return self.foo.__name__

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # check limits
        if isinstance(self.foo, type):
            foo = self.foo(**self.foo_kwargs)
        else:
            foo = partial(self.foo, **self.foo_kwargs)

        if not self.maximize:
            f = foo(x)
        else:
            f = -foo(x)

        if not apply_limits:
            return f

        return apply_limits(f, x, val=self.limit_val, limits=self.limits)

    def eval_on_range(self, n_points=100, parameter_range=None):
        parameter_range = parameter_range if parameter_range is not None else self.limits

        if hasattr(parameter_range, '__len__'):
            range_x = parameter_range[0]
            range_y = parameter_range[1]

        else:
            range_x = parameter_range
            range_y = parameter_range

        if not hasattr(range_x, '__len__'):
            range_x = torch.tensor([-range_x, range_x])

        if not hasattr(range_y, '__len__'):
            range_y = torch.tensor([-range_y, range_y])

        x = torch.linspace(*range_y, n_points)
        y = torch.linspace(*range_y, n_points)
        X, Y = torch.meshgrid(x, y)
        Z = self(torch.stack([X, Y], dim=-1).reshape(-1, 2)).reshape(*X.shape)

        return (X, range_x), (Y, range_y), Z

    def visualize(self, ax=None, n_points=100, show=False, logscale=False, parameter_range=None):
        import matplotlib.pyplot as plt
        import torch

        (X, range_x), (Y, range_y), Z = self.eval_on_range(n_points=n_points, parameter_range=parameter_range)

        if logscale:
            Z = torch.log(Z + 1)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        im = ax.imshow(Z.T, extent=(*range_x, *reversed(range_y)))
        ax.invert_yaxis()

        #ax.contour(X, Y, Z, levels=20, cmap='magma')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"{self.foo.__name__}")

        if show:
            plt.show()

        return im

    @classmethod
    def from_json(cls, json_repr):
        return cls(**json.loads(json_repr))

    def to_dict(self):
        dict_repr = self.__dict__.copy()
        # get location and name of foo
        dict_repr['foo'] = self.foo.__name__
        dict_repr['foo_module'] = self.foo_module or self.foo.__module__
        keys = list(dict_repr.keys())
        [dict_repr.pop(k) for k in keys if k.startswith('_')]

        # check if parameter_range contains tuples or np.arrays, recursively convert to list
        if hasattr(dict_repr['limits'], '__len__'):
            dict_repr['limits'] = [list(pr) if hasattr(pr, '__len__') else pr for pr in
                                   dict_repr['limits']]

        return dict_repr

    def to_json(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.to_json()

    @classmethod
    def load(cls, objective_repr):
        print(type(objective_repr), isinstance(objective_repr, cls))
        if isinstance(objective_repr, cls):
            return objective_repr

        elif isinstance(objective_repr, dict):
            return cls(**objective_repr)

        try:
            return cls.from_json(objective_repr)

        except json.JSONDecodeError:
            return cls(foo=objective_repr)
