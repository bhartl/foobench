import matplotlib.pyplot as plt
import torch


def visualize_2D(objective, ax=None, n_points=100, show=False, logscale=False, parameter_range=None,
                 title=None, **imshow_kwargs):
    (X, range_x), (Y, range_y), Z = objective.eval_on_range(n_points=n_points, parameter_range=parameter_range)

    if logscale:
        Z = torch.log(Z + 1)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    im = ax.imshow(Z.T, extent=(*range_x, *reversed(range_y)), **imshow_kwargs)
    ax.invert_yaxis()

    # ax.contour(X, Y, Z, levels=20, cmap='magma')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title is None:
        title = f"{objective.foo.__name__}"
    ax.set_title(title)

    if show:
        plt.show()

    return im