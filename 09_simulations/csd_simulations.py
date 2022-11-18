import numpy as np
from scipy.stats import ortho_group

def simulate_no_csd(n_domains, n_per_domain, p, noise_scale=1.):
    """Simulate data from a latent variable model with multiple
    domains, without a common component across domains.
    """
    xs, ys = None, None
    z = np.random.normal(size=(n_domains, p))
    # this beta range makes the values have a positive correlation with the
    # labels overall, despite being negatively correlated for some domains
    betas = np.random.uniform(-1, 2, size=(n_domains,))

    for i, beta_i in enumerate(betas):
        ys_i = np.random.choice([-1, 1], size=(n_per_domain, 1))
        xs_i = (
            np.tile(ys_i, (1, p)) *
            np.tile((np.array([beta_i]) @ z[[i], :]), (n_per_domain, 1))
        ) + (np.random.normal(scale=noise_scale, size=(n_per_domain, p)))
        if xs is None:
            xs = xs_i
        else:
            xs = np.concatenate((xs, xs_i))
        if ys is None:
            ys = ys_i
        else:
            ys = np.concatenate((ys, ys_i))
        
    return xs, ys


def simulate_no_csd_same_z(n_domains, n_per_domain, p, noise_scale=1.):
    """Simulate data from a latent variable model with multiple domains
    but the same LVs.

    This is similar to simulate_no_csd, but the latent vector z is shared
    across domains.
    """
    xs, ys = None, None
    z = np.random.normal(size=(1, p))
    # this beta range makes the values have a positive correlation with the
    # labels overall, despite being negatively correlated for some domains
    betas = np.random.uniform(-1, 2, size=(n_domains,))

    for i, beta_i in enumerate(betas):
        ys_i = np.random.choice([-1, 1], size=(n_per_domain, 1))
        xs_i = (
            np.tile(ys_i, (1, p)) *
            np.tile((np.array([beta_i]) @ z), (n_per_domain, 1))
        ) + (np.random.normal(scale=noise_scale, size=(n_per_domain, p)))
        if xs is None:
            xs = xs_i
        else:
            xs = np.concatenate((xs, xs_i))
        if ys is None:
            ys = ys_i
        else:
            ys = np.concatenate((ys, ys_i))
        
    return xs, ys


def simulate_no_csd_large_z(n_domains, n_per_domain, p, k, noise_scale=1.):
    """Simulate data from a latent variable model with multiple domains
    and multiple dimensions in the latent space.

    This is similar to simulate_no_csd_same_z, but the latent variables are
    a k x p matrix now instead of a single 1 x p vector, and the beta
    coefficients are also of dimension n_domains x k now instead of a single
    n_domains x 1 vector.
    """
    xs, ys = None, None
    assert k < p, 'latent dimension must be smaller than # of features'
    # generate orthogonal matrix and take the first k vectors as latent vars
    z = ortho_group.rvs(p)[:k, :]
    # this beta range makes the values have a positive correlation with the
    # labels overall, despite being negatively correlated for some domains
    betas = np.random.uniform(-1, 2, size=(n_domains, k))

    for i in range(n_domains):
        beta_i = betas[[i], :]
        ys_i = np.random.choice([-1, 1], size=(n_per_domain, 1))
        xs_i = (
            np.tile(ys_i, (1, p)) *
            np.tile((np.array(beta_i) @ z), (n_per_domain, 1))
        ) + (np.random.normal(scale=noise_scale, size=(n_per_domain, p)))
        if xs is None:
            xs = xs_i
        else:
            xs = np.concatenate((xs, xs_i))
        if ys is None:
            ys = ys_i
        else:
            ys = np.concatenate((ys, ys_i))
        
    return xs, ys


def simulate_csd(n_domains,
                 n_per_domain,
                 p,
                 k,
                 corr_noise=True,
                 noise_scale=1.,
                 corr_top=1.,
                 diag=None):
    """Simulate data from the CSD ('common-specific decomposition') model,
    with uncorrelated or correlated noise within domains.

    The main difference between this function and simulate_no_csd_large_z
    is the addition of a "common" latent vector that is consistent between
    domains. This is added to the "specific" latent variables that vary
    between domains to generate the data.

    CSD and its underlying generative model are described in more detail here:
    http://proceedings.mlr.press/v119/piratla20a/piratla20a.pdf
    """
    xs, ys = None, None
    assert k < p, 'latent dimension must be smaller than # of features'

    # generate orthogonal matrix
    z = ortho_group.rvs(p)

    # take the first vector as the common latent component
    z_c = z[:1, :]
    # take the next k vectors as the specific latent components
    z_s = z[1:k+1, :]

    # this beta range makes the values have a positive correlation with the
    # labels overall, despite being negatively correlated for some domains
    betas = np.random.uniform(-1, 2, size=(n_domains, k))

    for i in range(n_domains):
        beta_i = betas[[i], :]
        ys_i = np.random.choice([-1, 1], size=(n_per_domain, 1))

        if corr_noise:
            # create symmetric positive definite covariance matrix
            # doing this for each loop iteration creates noise that is correlated
            # within domains but uncorrelated across domains
            sigma_i = np.random.uniform(high=corr_top, size=(n_per_domain, n_per_domain))
            sigma_i = sigma_i @ sigma_i.T
            if diag is not None:
                sigma_i += (diag * np.eye(n_per_domain))
            noise = np.random.multivariate_normal(
                mean=np.zeros(n_per_domain), cov=sigma_i, size=(p,)
            ).T
        else:
            noise = np.random.normal(scale=noise_scale, size=(n_per_domain, p))

        xs_i = (
            np.tile(ys_i, (1, p)) *
            np.tile(z_c + (np.array(beta_i) @ z_s), (n_per_domain, 1))
        ) + noise
        if xs is None:
            xs = xs_i
        else:
            xs = np.concatenate((xs, xs_i))
        if ys is None:
            ys = ys_i
        else:
            ys = np.concatenate((ys, ys_i))
        
    return xs, ys

