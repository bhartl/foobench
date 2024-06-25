from torch import mean, sum, tensor, long, where, zeros, full, arange, multinomial


class NKModel:
    def __init__(self, foo, interaction_matrix):
        """ Constructs an NK model with a given foo function and interaction matrix

        :param foo: callable function that takes a vector of size k and returns a scalar
        :param interaction_matrix: n x n matrix adjacency matrix, where n is the number of elements and k is the number of neighbors.
                                   For each row i, non-zero elements represent the k neighbors of i.
        """
        self.foo = foo

        self.interaction_matrix = interaction_matrix
        assert all(neighs_i.sum() == self.k for neighs_i in self.interaction_matrix)

        self._neighborhood = None

    @property
    def n(self):
        return self.interaction_matrix.shape[0]

    @property
    def k(self):
        return int(sum(self.interaction_matrix[0]))

    def __call__(self, x):
        x = self.remap(x)
        x_shape = x.shape                                                # B (optional) x n
        x_vectorized = x[..., self.neighborhood]                         # B (optional) x n x k + 1
        x_vectorized = x_vectorized.reshape(-1, x_vectorized.shape[-1])  # B (optional) * n x (k + 1): flat first dims
        f_i = self.foo(x_vectorized)                                     # B (optional) * n, foo for each neighborhood
        f_i = f_i.reshape(*x_shape)                                      # B (optional) x n
        # return mean(f_i, dim=-1)                                         # B (optional), foo for each batch
        return sum(f_i, dim=-1) / self.k                           # B (optional), foo for each batch

    @property
    def neighborhood(self):
        if self._neighborhood is None:
            neighbors = [self.get_neighbors(i) for i in range(self.n)]
            self._neighborhood = tensor([[i, *neighbors[i]] for i in range(self.n)],
                                        dtype=long, device=self.interaction_matrix.device)

        return self._neighborhood

    def remap(self, x):
        """ Remap the input x to the values used by the foo function """
        return x

    @property
    def adjacency_matrix(self):
        a = self.interaction_matrix.copy()
        a = a.fill_diagonal_(1)
        return a

    def get_neighbors(self, i):
        return where(self.interaction_matrix[i])[0]

    @classmethod
    def generate_random_interactions(cls, n, k):
        interactions = zeros((n, n))
        for i in range(n):
            p = full(n, 1./(n-1))
            p[i] = 0

            values = arange(0, n, dtype=long)
            samples = multinomial(p, k - 1, replacement=False)
            c = values[samples]
            # c = np.random.choice(np.arange(0, n, dtype=int), k - 1, p=p, replace=False)

            interactions[i, c] = 1
            assert sum(interactions[i]) == k - 1
            assert interactions[i, i] == 0

        return tensor(interactions)

    @classmethod
    def generate_k_neighbor_interactions(cls, n, k):
        interactions = zeros((n, n), dtype=long)
        for i in range(n):
            min_i = i - k // 2 + (k + 1) % 2
            max_i = i + k // 2 + 1

            if min_i < 0:
                interactions[i, :i] = 1
                interactions[i, min_i:] = 1
            else:
                interactions[i, min_i:i] = 1

            if max_i > n:
                interactions[i, i+1:] = 1
                interactions[i, :max_i-n] = 1
            else:
                interactions[i, i + 1: max_i] = 1

        return interactions
