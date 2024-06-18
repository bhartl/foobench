""" Objective functions for the Evolutionary Strategies example,
taken from here: https://en.wikipedia.org/wiki/Test_functions_for_optimization

Test functions for optimization. (2023, December 29). In Wikipedia. https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import torch
from foobench import foos
from typing import Union
import json


def apply_limits(f, x, val=0., *limits):
    if len(limits) == 1:
        limits = [limits[0]] * x.shape[1]

    for i in range(x.shape[1]):
        exceeding = torch.logical_or(x[:, i] > limits[i], x[:, i] < -limits[i])
        f[exceeding] = val
    return f


class Objective:
    def __init__(self, foo: Union[str, callable] = "rastrigin", dim=2, reverse=False,
                 parameter_range: Union[int, tuple] = 4, foo_module=None, foo_kwargs=()):
        self.dim = dim
        self.foo_module = foo_module
        self._foo = None
        self.foo = foo
        self.foo_kwargs = foo_kwargs or {}
        self.reverse = reverse
        self.parameter_range = parameter_range

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
                module = __import__(self.foo_module)
                for comp in self.foo_module.split('.')[1:]:
                    module = getattr(module, comp)

                self._foo = getattr(module, foo)

    def __call__(self, x):
        # check limits
        if not self.reverse:
            return self.foo(x, **self.foo_kwargs)

        return -self.foo(x, **self.foo_kwargs)

    def visualize(self, ax=None, n_points=100, show=True, logscale=False):
        import matplotlib.pyplot as plt
        import torch

        if hasattr(self.parameter_range, '__len__'):
            range_x = self.parameter_range[0]
            range_y = self.parameter_range[1]

        else:
            range_x = self.parameter_range
            range_y = self.parameter_range

        if not hasattr(range_x, '__len__'):
            range_x = torch.tensor([-range_x, range_x])

        if not hasattr(range_y, '__len__'):
            range_y = torch.tensor([-range_y, range_y])

        x = torch.linspace(*range_y, n_points)
        y = torch.linspace(*range_y, n_points)
        X, Y = torch.meshgrid(x, y)
        Z = self(torch.stack([X, Y], dim=-1).reshape(-1, 2)).reshape(*X.shape)

        if logscale:
            Z = torch.log(Z + 1)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.imshow(Z.T, extent=(*range_x, *reversed(range_y)))
        ax.invert_yaxis()

        #ax.contour(X, Y, Z, levels=20, cmap='magma')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"{self.foo.__name__}")

        if show:
            plt.show()

        return ax

    @classmethod
    def from_json(cls, json_repr):
        return cls(**json.loads(json_repr))

    def to_json(self):
        dict_repr = self.__dict__.copy()
        # get location and name of foo
        dict_repr['foo'] = self.foo.__name__
        dict_repr['foo_module'] = self.foo_module or self.foo.__module__
        keys = list(dict_repr.keys())
        [dict_repr.pop(k) for k in keys if k.startswith('_')]

        # check if parameter_range contains tuples or np.arrays, recursively convert to list
        if hasattr(dict_repr['parameter_range'], '__len__'):
            dict_repr['parameter_range'] = [list(pr) if hasattr(pr, '__len__') else pr for pr in dict_repr['parameter_range']]

        return json.dumps(dict_repr)

    def __repr__(self):
        return json.dumps(self.__dict__)

    @classmethod
    def load(cls, objective_repr):
        if isinstance(objective_repr, Objective):
            return objective_repr

        return cls.from_json(objective_repr)


if __name__ == '__main__':
    foos = [{'foo': foos.rastrigin, 'logscale': False, 'lim': 4.5},
            {'foo': foos.ackley, 'logscale': False, 'lim': 4.5},
            {'foo': foos.rosenbrock, 'logscale': True, 'lim': 3},
            {'foo': foos.sphere, 'logscale': False, 'lim': 2},
            {'foo': foos.beale, 'logscale': True, 'lim': 4.5},
            {'foo': foos.himmelblau, 'logscale': True, 'lim': 4.5},
            {'foo': foos.hoelder_table, 'logscale': False, 'lim': 10.},
            {'foo': foos.double_dip, 'logscale': False, 'lim': 2.},
            ]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(foos)//2, figsize=(15, 10))
    axes = axes.ravel()
    for i, f in enumerate(foos):
        obj = Objective(foo=f['foo'], parameter_range=f['lim'], reverse=False)
        obj = Objective.load(obj.to_json())
        print(obj.foo.__name__, obj.foo.__module__)
        obj.visualize(logscale=f['logscale'], ax=axes[i], show=False)

    plt.tight_layout()
    plt.show()
