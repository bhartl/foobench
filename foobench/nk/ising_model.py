from .nk_model import NKModel
from torch import triu_indices, sum, tensor
import torch.nn.functional as F


def ising_energy(x, J=1.):
    """ Evaluate E = -J * sum_{i,j} x_i * x_j, with j > i (no diagonal)

        x is clipped to [-1, 1] to avoid numerical issues.
    """
    # get the pairwise interactions -> x_i * x_j for all i, j
    x_ijk = x[..., None] * x[:, None, :]

    # sum over pairwise interactions, excluding the diagonal
    return -J * (sum(x_ijk[:, 0], dim=-1) - x_ijk.diagonal(dim1=-2, dim2=-1)[:, 0])


class SpingGlassModel(NKModel):
    def __init__(self, interaction_matrix, J, loc=(-1, 1), scale=1.0, values=None, tau=1):
        self.loc = loc
        self.scale = scale
        self.values = values if values is not None else loc
        self.tau = tau

        self._J = J
        self.J = J

        NKModel.__init__(self, foo=self.energy, interaction_matrix=interaction_matrix)

    def energy(self, x):
        return ising_energy(x, J=self.J)

    def remap(self, x):
        #eval logits of each spin, relating to spin locations
        if x.ndim == 1:
            loc = tensor(self.loc, device=x.device)[None, :]
        elif x.ndim == 2:
            loc = tensor(self.loc, device=x.device)[None, None, :]
        else:
            raise ValueError("x should be 1 or 2 dimensional")

        # Compute the logits of the Gumbel-Softmax distribution
        logits = -(x[..., None] - loc) ** 2 / (self.scale ** 2)

        # Apply the Gumbel-Softmax trick to obtain spin configurations in discrete space
        spin_ids = F.gumbel_softmax(logits, tau=self.tau, dim=-1, hard=False).argmax(-1)
        spins = tensor(self.values, device=x.device)[spin_ids.long()]
        return spins


class IsingModel(SpingGlassModel):
    def __init__(self, n, J=1., scale=1.):
        interaction_matrix = SpingGlassModel.generate_k_neighbor_interactions(n=n, k=3)
        SpingGlassModel.__init__(self, interaction_matrix=interaction_matrix,
                                 J=J, loc=(-1, 1), scale=scale)
