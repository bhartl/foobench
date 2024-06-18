import torch


def rastrigin(x, A=10):
    assert len(x.shape) == 2, f"Input shape must be (BATCH-SIZE, n), got {x.shape}"
    f = A * x.shape[1] + (x ** 2 - A * torch.cos(2 * torch.pi * x)).sum(axis=1)
    return f


def ackley(x):
    assert len(x.shape) == 2 and x.shape[1] == 2, f"Input shape must be (BATCH-SIZE, 2), got {x.shape}"
    f = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2))) - \
        torch.exp(0.5 * (torch.cos(2 * torch.pi * x[:, 0]) + torch.cos(2 * torch.pi * x[:, 1]))) + 20 + torch.exp(torch.tensor(1, device=x.device))
    return f


def rosenbrock(x, a=1, b=100):
    assert len(x.shape) == 2, f"Input shape must be (BATCH-SIZE, n), got {x.shape}"
    f = ((a - x[:, :-1]) ** 2 + b * (x[:, 1:] - x[:, :-1] ** 2) ** 2).sum(axis=1)
    return f


def sphere(x):
    assert len(x.shape) == 2, f"Input shape must be (BATCH-SIZE, n), got {x.shape}"
    f = (x ** 2).sum(axis=1)
    return f


def beale(x):
    assert len(x.shape) == 2, f"Input shape must be (BATCH-SIZE, n), got {x.shape}"
    f = (1.5 - x[:, 0] + x[:, 0] * x[:, 1]) ** 2 + \
        (2.25 - x[:, 0] + x[:, 0] * x[:, 1] ** 2) ** 2 + \
        (2.625 - x[:, 0] + x[:, 0] * x[:, 1] ** 3) ** 2
    return f


def himmelblau(x):
    assert len(x.shape) == 2 and x.shape[1] == 2, f"Input shape must be (BATCH-SIZE, 2), got {x.shape}"
    f = (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
    return f


def hoelder_table(x):
    assert len(x.shape) == 2 and x.shape[1] == 2, f"Input shape must be (BATCH-SIZE, 2), got {x.shape}"
    f = -torch.abs(torch.sin(x[:, 0]) * torch.cos(x[:, 1]) * torch.exp(torch.abs(1 - torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / torch.pi)))
    return f


def cross_in_tray(x):
    assert len(x.shape) == 2 and x.shape[1] == 2, f"Input shape must be (BATCH-SIZE, 2), got {x.shape}"
    f = -0.0001 * (torch.abs(torch.sin(x[:, 0]) * torch.sin(x[:, 1]) * torch.exp(torch.abs(100 - torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / torch.pi))) + 1) ** 0.1
    return f


def double_dip(x: torch.Tensor, s=0.125, m=0.4 , v1=1., v2=1.):
    return -v1 * torch.exp(-((0.5 * (x - m) / s) ** 2).sum(dim=-1)) + -v2 * torch.exp(-((0.5 * (x + m) / s) ** 2).sum(dim=-1))


def double_dip_discrete(x: torch.Tensor, s=0.125, m=0.4 , v1=1., v2=1.):
    return -v1 * (torch.linalg.norm(x - m, dim=-1) < s) + -v2 * (torch.linalg.norm(x + m, dim=-1) < s)


def dip_series(x: torch.Tensor, s=0.125, m=0.25 , v1=0.25, v2=0.5, v3=1.0, v4=0.5, v5=0.25):
    rc = -v1 * (torch.linalg.norm(x - m * 2, dim=-1) < s)
    for i, vi in zip(range(-1, 3), [v2, v3, v4, v5]):
        rc -= vi * (torch.linalg.norm(x + m * i, dim=-1) < s)

    return rc


def discrete_peak_series_r(x: torch.Tensor, s=0.125, m=0.25 , v1=1., v2=0.5, v3=0.25, v4=0.5, v5=1.):
    rc = -v1 * (torch.linalg.norm(x - m * 2, dim=-1) < s)
    for i, vi in zip(range(-1, 3), [v2, v3, v4, v5]):
        rc -= vi * (torch.linalg.norm(x + m * i, dim=-1) < s)

    return rc


MINIMA = {'rastrigin': {'x': torch.tensor([0., 0.]), 'y': 0.},
          'ackley': {'x': torch.tensor([0., 0.]), 'y': 0.},
          'rosenbrock': {'x': torch.tensor([1., 1.]), 'y': 0.},
          'sphere': {'x': torch.tensor([0., 0.]), 'y': 0.},
          'beale': {'x': torch.tensor([3., 0.5]), 'y': 0.}
          }
