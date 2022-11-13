import numpy as np

def simulate_no_csd(n_domains, n_per_domain, p, noise_scale=1.):
    xs, ys = None, None
    z = np.random.normal(size=(n_domains, p))
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
    xs, ys = None, None
    z = np.random.normal(size=(1, p))
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
    xs, ys = None, None
    z = np.random.normal(size=(k, p))
    betas = np.random.uniform(-1, 2, size=(n_domains, k))

    for i in range(n_domains):
        beta_i = betas[[i], :]
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

def simulate_csd(n_domains, n_per_domain, p, noise_scale):
    raise NotImplementedError

