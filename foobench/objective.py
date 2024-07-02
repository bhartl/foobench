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
        """ Evaluate the ND objective function on a meshgrid. """
        parameter_range = parameter_range if parameter_range is not None else self.limits
        if not hasattr(parameter_range, '__len__'):
            parameter_range = [parameter_range] * self.dim

        # Create a list of 1D tensors for each dimension
        ranges = []
        for pr in parameter_range:
            if not hasattr(pr, '__len__'):
                pr = torch.tensor([-pr, pr])
            ranges.append(torch.linspace(*pr, n_points))

        # Create a meshgrid with n dimensions
        mesh = torch.meshgrid(*ranges)

        # Stack and reshape the meshgrid to pass it to the objective function
        stacked_mesh = torch.stack(mesh, dim=-1).reshape(-1, self.dim)

        # Evaluate the objective function
        Z = self(stacked_mesh)

        # Reshape the output of the objective function into the shape of the meshgrid
        Z = Z.reshape(*[len(r) for r in ranges])

        return *((m, [r[0], r[-1]]) for m, r in zip(mesh, ranges)), Z

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
