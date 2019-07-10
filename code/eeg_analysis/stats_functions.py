# --- jose c. garcia alanis
# --- utf-8
# --- Python 3.6.2
#
# --- MVPA of EEG data - DPX
# --- version jan 2018
#
# --- define functions for multivariate pattern analysis
# --- stats

# ==================================================================================================
# --- Statistics tests used, thanks to Jean-Rémi King ---


# signed rank test
def _my_wilcoxon(X):
    from scipy.stats import wilcoxon
    out = wilcoxon(X)
    return out[1]


# loop function
def _loop(x, function):
    out = list()
    for ii in range(x.shape[1]):
        out.append(function(x[:, ii]))
    return out


# correct p values for multiple testing
def parallel_stats(X, function=_my_wilcoxon, correction='FDR', n_jobs=-1):  # -1
    from mne.parallel import parallel_func
    import numpy as np
    from mne.stats import fdr_correction

    if correction not in [False, None, 'FDR']:
        raise ValueError('Unknown correction')
    # reshape to 2D
    X = np.array(X)
    dims = X.shape
    X.resize([dims[0], np.prod(dims[1:])])
    # prepare parallel
    n_cols = X.shape[1]
    parallel, pfunc, n_jobs = parallel_func(_loop, n_jobs)
    n_chunks = min(n_cols, n_jobs)
    chunks = np.array_split(range(n_cols), n_chunks)
    p_values = parallel(pfunc(X[:, chunk], function) for chunk in chunks)
    p_values = np.reshape(np.hstack(p_values), dims[1:])
    X.resize(dims)
    # apply correction
    if correction == 'FDR':
        dims = p_values.shape
        _, p_values = fdr_correction(p_values)
        p_values = np.reshape(p_values, dims)
    return p_values


# one sample t-test
def _stat_fun(x, sigma=0, method='relative'):
    from mne.stats import ttest_1samp_no_p
    import numpy as np
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


# threshold free cluster permutation test
def stats_tfce(X, n_permutations=2**10,threshold=dict(start=.1, step=.1), n_jobs=2):  # 2
    # threshold free cluster enhancement for GATs
    import numpy as np
    from mne.stats import spatio_temporal_cluster_1samp_test

    X = np.array(X)
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
    X,out_type='mask',
    stat_fun=_stat_fun,
    n_permutations=n_permutations,
    threshold=threshold,
    n_jobs=n_jobs)
    p_values = p_values.reshape(X.shape[1:])
    return p_values
