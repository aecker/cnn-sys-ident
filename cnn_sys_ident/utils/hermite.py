import numpy as np
import math
from scipy.special import gamma
from numpy.polynomial.polynomial import polyval
from numpy import pi

def hermcgen(mu, nu):
    """Generate coefficients of 2D Hermite functions"""
    nur = np.arange(nu + 1)
    num = gamma(mu + nu + 1) * gamma(nu + 1) * ((-2) ** (nu - nur))
    denom = gamma(mu + 1 + nur) * gamma(1 + nur) * gamma(nu + 1 - nur)
    return num / denom


def hermite_2d(N, npts, xvalmax=None):
    """Generate 2D Hermite function basis

    Arguments:
    N           -- the maximum rank.
    npts        -- the number of points in x and y

    Keyword arguments:
    xvalmax     -- the maximum x and y value (default: 2.5 * sqrt(N))

    Returns:
    H           -- Basis set of size N*(N+1)/2 x npts x npts
    desc        -- List of descriptors specifying for each
                   basis function whether it is:
                        'z': rotationally symmetric
                        'r': real part of quadrature pair
                        'i': imaginary part of quadrature pair

    """
    xvalmax = xvalmax or 2.5 * np.sqrt(N)
    ranks = range(N)

    # Gaussian envelope
    xvalmax *= 1 - 1 / npts
    xvals = np.linspace(-xvalmax, xvalmax, npts, endpoint=True)[...,None]

    gxv = np.exp(-xvals ** 2 / 4)
    gaussian = np.dot(gxv, gxv.T)

    # Hermite polynomials
    mu = np.array([])
    nu = np.array([])
    desc = []
    for i, rank in enumerate(ranks):
        muadd = np.sort(np.abs(np.arange(-rank, rank + 0.1, 2)))
        mu = np.hstack([mu, muadd])
        nu = np.hstack([nu, (rank - muadd) / 2])
        if not (rank % 2):
            desc.append('z')
        desc += ['r', 'i'] * int(np.floor((rank + 1) / 2))

    theta = np.arctan2(xvals, xvals.T)
    radsq = xvals ** 2 + xvals.T ** 2
    nbases = mu.size
    H = np.zeros([nbases, npts, npts])
    for i, (mui, nui, desci) in enumerate(zip(mu, nu, desc)):
        radvals = polyval(radsq, hermcgen(mui, nui))
        basis = gaussian * (radsq ** (mui / 2)) * radvals * np.exp(1j * mui * theta)
        basis /= np.sqrt(2 ** (mui + 2 * nui) * pi * \
                         math.factorial(mui + nui) * math.factorial(nui))
        if desci == 'z':
            H[i] = basis.real / np.sqrt(2)
        elif desci == 'r':
            H[i] = basis.real
        elif desci == 'i':
            H[i] = basis.imag

    # normalize
    return H / np.sqrt(np.sum(H ** 2, axis=(1, 2), keepdims=True)), desc, mu


def rotation_matrix(desc, mu, angle):
    R = np.zeros((len(desc), len(desc)))
    for i, (d, m) in enumerate(zip(desc, mu)):
        if d == 'r':
            Rc = np.array([[np.cos(m*angle), np.sin(m*angle)],
                           [-np.sin(m*angle), np.cos(m*angle)]])
            R[i:i+2,i:i+2] = Rc
        elif d == 'z':
            R[i,i] = 1
    return R
