from foobench import foos
from foobench import Objective

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
