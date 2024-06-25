from unittest import TestCase


class TestIsingModel(TestCase):
    def test_ising_model(self):
        import torch
        from foobench.nk import IsingModel

        n = 6
        J = 1.
        model = IsingModel(n, J, scale=0.1)

        # single evaluation
        x = torch.randn(n) * 2 - 1
        y = model(x)
        assert y.shape == ()

        # batch evaluation
        x = torch.randn(10, n) * 2 - 1
        y = model(x)
        assert y.shape == (10,)

        # ones
        x = torch.ones(n)
        y = model(x)
        assert y == -n * J

        y = model(-x)
        assert y == -n * J

        x[::2] = -1
        y = model(x)
        assert y == n * J
