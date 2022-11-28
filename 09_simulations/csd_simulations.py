import numpy as np
from scipy.stats import ortho_group

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

    CSD and its underlying generative model are described in more detail here:
    http://proceedings.mlr.press/v119/piratla20a/piratla20a.pdf

    Arguments:
      n_domains: number of domains
      n_per_domain: number of samples per domain (currently each domain
                    has the same number of samples)
      p: number of features
      k: number of latent dimensions
      corr_noise: whether or not to add noise that is correlated within domains
      noise_scale: variance parameter for uncorrelated noise model
                   (no effect when corr_noise is True)
      corr_top: off-diagonal covariance is sampled from [0, corr_top]
                (no effect when corr_noise is False)
      diag: used to scale diagonal of covariance matrix; i.e. to control
            tradeoff between variance and covariance (higher = more
            within-domain variance) (no effect when corr_noise is False)
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

