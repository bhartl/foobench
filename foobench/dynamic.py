import numpy as np


class OscillatingDoubleDip:
    """ A dynamic version of the double dip function that oscillates the magnitude of the two peaks. """
    def __init__(self, phi=np.pi, omega=0.1):
        self.omega = omega
        self.phi = phi
        self.t = 0

        from foobench.foos import double_dip
        self._foo = double_dip

    def __call__(self, x, s=0.125, m=0.4, v1=1., v2=1.):
        v1 *= np.cos(self.omega * self.t)
        v2 *= np.cos(self.omega * self.t + self.phi)
        self.t += 1
        return self._foo(x, s=s, m=m, v1=v1, v2=v2)

    def to_dict(self):
        return {"omega": self.omega, "v_phi": self.phi}

    @property
    def __name__(self):
        # return f"OscillatingDoubleDip(phi={self.phi}, omega={self.omega})"
        return "OscillatingDoubleDip"
