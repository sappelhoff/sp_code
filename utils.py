"""Utility functions for analyzing the sampling paradigm."""
import itertools
import multiprocessing
import os
import warnings
from collections import OrderedDict
from functools import partial

import mne
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import scipy.stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats.stats import _kendall_dis
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import _cov
from tqdm.auto import tqdm


# Adjust this path to where the bids directory is stored
home = os.path.expanduser("~")
if "stefanappelhoff" in home:
    BIDS_ROOT = os.path.join("/", "home", "stefanappelhoff", "Desktop", "sp_data")
elif "appelhoff" in home:
    BIDS_ROOT = os.path.join("/", "home", "appelhoff", "appelhoff", "sp_data")

# There are 5 types of tasks, in the datafiles the markers
# are always the same. We prepare a dict to distinguish them
# when we concatenate datafiles
TASK_BASED_INCREMENTS = {
    task: increment
    for task, increment in zip(
        ["ActiveFixed", "ActiveVariable", "YokedFixed", "YokedVariable", "description"],
        range(100, 999, 100),
    )
}
TASK_NAME_MAP = {
    "ActiveFixed": "AF",
    "ActiveVariable": "AV",
    "YokedFixed": "YF",
    "YokedVariable": "YV",
    "description": "DESC",
}


# Define dimensions along which we want to know about the events
EVENT_PARAMS = OrderedDict()

EVENT_PARAMS["sampling"] = ("active", "yoked")
EVENT_PARAMS["stopping"] = ("fixed", "variable")
EVENT_PARAMS["direction"] = ("left", "right")
EVENT_PARAMS["outcome"] = tuple(["out{}".format(ii) for ii in range(1, 10)])
EVENT_PARAMS["outcome_bin"] = tuple(["bin{}".format(ii) for ii in range(1, 10)])
EVENT_PARAMS["outcome_bin_orth"] = tuple(
    ["orthbin{}".format(ii) for ii in range(1, 10)]
)
EVENT_PARAMS["timing"] = ("early", "mid", "late", "mixed_timing")
EVENT_PARAMS["switch"] = ("pre_switch", "post_switch", "no_switch", "mixed_switch")
EVENT_PARAMS["order"] = ("first", "midth", "last", "mixed_order")
EVENT_PARAMS["half"] = ("first_half", "second_half", "mixed_half")

# Form an EVENT_ID out of all possible combinations
_all_codes = list(itertools.product(*EVENT_PARAMS.values()))
_all_codes = ["/".join(ii) for ii in _all_codes]
EVENT_ID = dict(zip(_all_codes, range(1, len(_all_codes) + 1)))


def get_ibin_for_sample(n_samples, sample, min_n_samples, memo):
    """Calculate which bin idx a sample falls into.

    Parameters
    ----------
    n_samples : int
        Overall number of samples in this trial.
    sample : int
        Index of the current sample to be binned. Zero indexed, sample must
        be < n_samples.
    min_n_samples : int
        Minimum number of samples in each trial. This determins the bins by
        np.linspace(0, 1, min_n_samples).
    memo : dict
        Previously computed results assuming the same `min_n_samples`
        parameter. Each key is a tuple in the form
        (`n_samples`, `sample`, `min_n_samples`) with a value `sample_ibin`.

    Returns
    -------
    sample_ibin : int
        The bin idx of the current `sample`.
    memo : dict
        Previously computed results updated with current result.

    Notes
    -----
    Bins include the right bin edge: ``bins[i-1] < x <= bins[i]``

    Examples
    --------
    >>> min_n_samples = 6
    >>> sample_ibin, memo = get_ibin_for_sample(6, 0, min_n_samples, {})
    0
    >>> sample_ibin, memo = get_ibin_for_sample(6, 5, min_n_samples, memo)
    5
    >>> sample_ibin, memo = get_ibin_for_sample(12, 11, min_n_samples, memo)
    5
    """
    # If we have memoized this result, return early
    if (n_samples, sample, min_n_samples) in memo:
        return memo[(n_samples, sample, min_n_samples)], memo

    # Else, check and start computing
    if n_samples < min_n_samples:
        raise RuntimeError('n_samples must be >= min_n_samples')
    if sample >= n_samples:
        raise RuntimeError('sample must be < n_samples')

    data = np.arange(n_samples) / (n_samples-1)
    bins = np.linspace(0, 1, min_n_samples)
    bin_idxs = np.digitize(data, bins, right=True)
    for isample, ibin in zip(np.arange(n_samples), bin_idxs):
        memo[(n_samples, isample, min_n_samples)] = ibin

    sample_ibin = memo[(n_samples, sample, min_n_samples)]
    return sample_ibin, memo


def get_df_bnt(BIDS_ROOT):
    """Get Berlin Numeracy Test data frame."""
    # mapping between paper "task" 1, 2a, 2b, 3 and this experiment "task" 1, 2, 3, 4:
    bnt_map = {
        "q1": "q1_correct",
        "q2a": "q4_correct",
        "q2b": "q2_correct",
        "q3": "q3_correct",
    }

    # Read data
    fname_bnt = os.path.join(BIDS_ROOT, "phenotype", "berlin_numeracy_test.tsv")

    # Make sure the result columns are read as booleans
    result_columns = ["q{}_correct".format(i) for i in range(1, 5)]
    dtypes = {i: bool for i in result_columns}

    df_bnt = pd.read_csv(fname_bnt, sep="\t", dtype=dtypes)

    # Get "subject" column: sub-01 -> 1
    df_bnt["subject"] = df_bnt["participant_id"].str.lstrip("sub-").to_numpy(dtype=int)

    def classify_bnt(df_bnt):
        """Classify participants into quartiles for the BNT.

        Use adaptive scoring method as described in:
        https://doi.org/10.1037/t45862-000

        """
        # triage the quartile for each subject
        quartiles = list()
        for idx, row in df_bnt.iterrows():

            # Got q1 right
            if row[bnt_map["q1"]]:
                # Can be 3rd or 4th quartile

                # Got q1 and q2b right
                if row[bnt_map["q2b"]]:
                    # is 4th quartile
                    quartiles.append(4)
                    continue

                # Got q1 right, but q2b wrong
                else:
                    # Can still be 3d or 4th quartile

                    if row[bnt_map["q3"]]:
                        # q1 correct, q2b wrong, but q3 correct
                        # is 4th quartile
                        quartiles.append(4)
                        continue

                    else:
                        # q1 correct, q2b wrong, q3 wrong
                        # is 3rd quartile
                        quartiles.append(3)
                        continue

            # Got q1 wrong
            else:
                # Can be 1st or 2nd quartile

                if not row[bnt_map["q2a"]]:
                    # Got q1 wrong, q2a wrong
                    # is first quartile
                    quartiles.append(1)
                    continue

                else:
                    # Got q1 wrong, but q2a right
                    # is 2nd quartile
                    quartiles.append(2)
                    continue

        return quartiles

    # Add evaluated columns to DF
    df_bnt["bnt_quartile"] = classify_bnt(df_bnt)
    df_bnt["bnt_n_correct"] = np.sum(df_bnt[result_columns].to_numpy(dtype=int), axis=1)

    return df_bnt


def extract_sample_frequencies(df, with_sides):
    """Extract sample frequencies from `df`'s 'action' and 'outcome' cols.

    Parameters
    ----------
    df : pandas.DataFrame
        The behavioral data.
    with_sides : bool
        If True, get the sampling frequencies for left (1-9) and right (1-9)
        in that order. If False, get sampling frequencies for 1-9 collapsed
        over sides (left, right).

    Returns
    -------
    sample_frequencies : ndarray
        The sample frequencies.

    """
    # get frequencies from 1 to 9 for each side
    sample_frequencies = df.groupby("action")["outcome"].value_counts(sort=False)

    # sanity check ordering of 'outcome' is 1-9 for each side
    outcome_order = np.array([i[1] for i in sample_frequencies.index])
    check_outcome_order = np.tile(np.arange(1, 10), 2)
    assert np.array_equal(outcome_order, check_outcome_order)

    # sanity check that ordering of 'action' is left, then right (0, then 1)
    level_vals = (
        df.groupby("action")["outcome"]
        .value_counts(sort=False)
        .index.get_level_values("action")
    )
    assert level_vals[0] == 0
    assert level_vals[-1] == 1

    # if collapsed over sides is wanted, we simply take the sum
    sample_frequencies = sample_frequencies.to_numpy()
    if not with_sides:
        sample_frequencies = sample_frequencies.reshape(2, 9).sum(0)
    return sample_frequencies


def _kendall_tau_a(x, y):
    """Compute Kendall's Tau metric, A-variant.

    Taken from scipy.stats.kendalltau and modified to be the tau-a variant.
    Same as in the mne-rsa package by @wmvanvliet. See:
    https://github.com/wmvanvliet/mne-rsa/blob/0c92eaf64c8d05676879b0da7ffc96ad6ac2d12e/mne_rsa/rsa.py#L15
    https://github.com/scipy/scipy/pull/9361/commits/4d85c35f57cd577a0322becef0d8c7ccb6fdec39

    Practically, this line changes:

    tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)

    to

    tau = con_minus_dis / tot

    See: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient#Tau-a

    Tested against R package DescTools:KendallTauA
    Tested against pyrsa (https://github.com/rsagroup/pyrsa)

    Notes
    -----
    When comparing model RDMs that predict tied ranks (e.g., category
    models) with model RDMs that make more detailed predictions
    (e.g., brain RDMs), Kendall's tau A correlation is recommended,
    because unlike Pearson and Spearman correlations, it does *not*
    prefer simple model RDMs (those that contain tied ranks). [1]_ [2]_

    References
    ----------
    .. [1] https://doi.org/10.1016/j.jmp.2016.10.007
    .. [2] https://doi.org/10.1371/journal.pcbi.1003553



    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError(
            "All inputs to `kendalltau` must be of the same size,"
            " found x-size %s and y-size %s" % (x.size, y.size)
        )
    elif not x.size or not y.size:
        return np.nan  # Return NaN if arrays are empty

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype("int64", copy=False)
        cnt = cnt[cnt > 1]
        return (
            (cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.0) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.0) * (2 * cnt + 5)).sum(),
        )

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind="mergesort")
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype("int64", copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)  # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)  # ties in y, stats

    tot = (size * (size - 1)) // 2

    if xtie == tot or ytie == tot:
        return np.nan

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = con_minus_dis / tot
    # Limit range to fix computational errors
    tau = min(1.0, max(-1.0, tau))

    return tau


def split_dict(erp_dict):
    """Split an erp_dict into AF, AV, YF, YV.

    An erp_dict that contains `sampling_styles` and `stopping_styles` is
    split up into a new, nested erp_dict: `newd`, where the first level of
    keys are all combinations of sampling and stopping styles, and the second
    level of keys are as they were in the original `erp_dict`

    Parameters
    ----------
    erp_dict : dict
        The ERP dict with mne.evoked objects. Must be a dict with keys that
        start with `active/fixed`, or similar.

    Returns
    -------
    newd : dict
        The `erp_dict` split up into a two level nested dict

    """
    sampling_styles = ("active", "yoked")
    stopping_styles = ("fixed", "variable")

    newd = dict()
    for key, val in erp_dict.items():
        keycomponents = key.split("/")
        assert keycomponents[0] in sampling_styles + stopping_styles
        assert keycomponents[1] in sampling_styles + stopping_styles
        remaining_facs_len = len(keycomponents) - 2
        for sampling_style in sampling_styles:
            for stopping_style in stopping_styles:
                if all(
                    style in keycomponents for style in [sampling_style, stopping_style]
                ):
                    style = "{}/{}".format(sampling_style, stopping_style)
                    newd[style] = newd.get(style, dict())
                    newkey = "/".join(keycomponents[-remaining_facs_len:])
                    newd[style][newkey] = val
    return newd


def bin_outcomes(vec, nbins):
    """Turn `vec` into `nbins` even-sized bins starting with 1."""
    # Get bin edges through percentiles to ensure each bin contains an
    # approximately equal number of observations
    bin_edges = np.zeros(nbins)
    for nth in range(1, nbins + 1):
        bin_edges[nth - 1] = np.percentile(vec, q=(100 / nbins) * nth)

    # bin the data, including the right edge, i.e. from minimum of data until
    # first edge (inclusive) is the first bin ... etc.
    vec_binned = np.digitize(vec, bins=bin_edges, right=True)

    # Make sure the bins are 1-indexed
    vec_binned += 1

    return vec_binned


def add_binned_outcomes_to_df(df):
    """Add outcome columns to the behavioral data.

    Normalize each outcome by the mean_outcome of the trial that outcome
    occurred in. Then digitize the resulting vector so that we have as many
    bins as we have "number" outcomes (1 to 9). Binned outcomes have a very
    high correlation with outcomes, so we also partial out outcomes from
    binned outcomes, and add these "orthogonalized" binned outcomes.

    In this process, the following columns will be added:

    - outcome_mean, the mean outcome for a trial
    - outcome_norm, the sample outcome when normalized by trial mean outcome
    - outcome_bin, the bin that the sample outcome is in, considering all trials
    - outcome_bin_orth, outcome_bin but with "outcome" partialled out

    As such, "outcome_bin_orth" represents the local information in a single
    trial, whereas "outcome" represents the global information over the
    whole experiment.

    See also
    --------
    bin_outcomes
    """
    already_in_df = []
    msg = f"Cannot run function, because several columns to be added are already in df: {already_in_df}"
    for col in ["outcome_mean", "outcome_norm", "outcome_bin", "outcome_bin_orth"]:
        if col in df:
            already_in_df.append(col)
    if len(already_in_df) > 0:
        raise ValueError(msg)

    # work on a copy
    df = df.copy()

    # Get mean outcome per subject, task, and trial
    df_mean_outcome = (
        df.groupby(["subject", "task", "trial"])
        .mean()["outcome"]
        .reset_index()
        .rename(columns={"outcome": "outcome_mean"})
    )

    # Merge mean_outcome column onto this DF
    df = df.merge(df_mean_outcome, on=["subject", "task", "trial"], validate="m:1")

    # Normalized outcomes
    df["outcome_norm"] = df["outcome"] - df["outcome_mean"]

    # Add binned outcomes
    df["outcome_bin"] = bin_outcomes(df["outcome_norm"].to_numpy(), nbins=9)

    # Partial out outcomes from binned outcomes
    X = df[["outcome", "outcome_bin"]].to_numpy()
    X_orth = spm_orth(X)

    # Make sure that `spm_orth` did not change the first column
    assert np.array_equal(X_orth[:, 0], df["outcome"].to_numpy())

    # however, the second column was orthogonalized with respect to first
    assert not np.array_equal(X_orth[:, 1], df["outcome_bin"].to_numpy())

    # bin the orthogonalized vector again
    df["outcome_bin_orth"] = bin_outcomes(X_orth[:, 1], nbins=9)

    return df


def spm_orth(X, opt="pad"):
    """Perform a recursive Gram-Schmidt orthogonalisation of basis functions.

    This function was translated from Matlab to Python using the code from
    the SPM MATLAB toolbox on ``spm_orth.m`` [1]_.

    .. warning:: For arrays of shape(1, m) the results are not equivalent
                 to what the original ``spm_orth.m`` produces.

    Parameters
    ----------
    X : numpy.ndarray, shape(n, m)
        Data to perform the orthogonalization on. Will be performed on the
        columns.
    opt : {"pad", "norm"}, optional
        If ``"norm"``, perform a euclidean normalization according to
        ``spm_en.m`` [2]_. If ``"pad"``, ensure that the output is of the
        same size as the input. Defaults to ``"pad"``.

    Returns
    -------
    X : numpy.ndarray
        The orthogonalized data.

    References
    ----------
    .. [1] https://github.com/spm/spm12/blob/master/spm_orth.m
    .. [2] https://github.com/spm/spm12/blob/master/spm_en.m
    """
    assert X.ndim == 2, "This function only operates on 2D numpy arrays."
    n, m = X.shape
    X = X[:, np.any(X, axis=0)]  # drop all "all-zero" columns
    rank_x = np.linalg.matrix_rank(X)

    x = X[:, np.newaxis, 0]
    j = [0]
    for i in range(1, X.shape[-1]):
        D = X[:, np.newaxis, i]
        D = D - np.dot(x, np.dot(np.linalg.pinv(x), D))
        if np.linalg.norm(D, 1) > np.exp(-32):
            x = np.concatenate([x, D], axis=1)
            j.append(i)

        if len(j) == rank_x:
            break

    if opt == "pad":
        # zero padding of null space (if present)
        X = np.zeros((n, m))
        X[:, np.asarray(j)] = x

    elif opt == "norm":
        # Euclidean normalization, based on "spm_en.m", see docstring.
        for i in range(X.shape[-1]):
            if np.any(X[:, i]):
                X[:, i] = X[:, i] / np.sqrt(np.sum(X[:, i] ** 2))

    else:
        # spm_orth.m does "X = x" here. We raise an error, because
        # this option is not documented in spm_orth.m
        # X = x
        raise ValueError("opt must be one of ['pad', 'norm'].")

    return X


def _check_nsubjs(erp_dict):
    """Check that number of subjs in dict is consistent.

    Parameters
    ----------
    erp_dict : dict
        Each key is an ERP contrast with a list of ERPs as
        value. The list contains MNE-Python evoked objects
        or np.nan.

    Returns
    -------
    nsubjs : int
        Number of subjects for each list of ERPs.

    Raises
    ------
    RuntimeError
        if `nsubjs` is not consistent across all lists of ERPs.

    """
    DUMMY_NSUBJS = -1
    nsubjs = DUMMY_NSUBJS
    for key in erp_dict:
        this_nsubjs = len(erp_dict[key])

        # If this is the first nsubjs we encounter, store it
        if nsubjs == DUMMY_NSUBJS:
            nsubjs = this_nsubjs

        # make sure all nsubjs are the same
        if this_nsubjs != nsubjs:
            raise RuntimeError("Encountered different amounts of subjs.")

    return nsubjs


def _check_erp_shape(erp_dict):
    """Check the shape of ERPs in dict is consistent.

    Parameters
    ----------
    erp_dict : dict
        Each key is an ERP contrast with a list of ERPs as
        value. The list contains MNE-Python evoked objects
        or np.nan.

    Returns
    -------
    erp_shape : tuple
        Shape of the ERPs.

    Raises
    ------
    RuntimeError
        if the `erp_shape` is not consistent across all ERPs.

    """
    DUMMY_SHAPE = (-1, -1)
    erp_shape = DUMMY_SHAPE
    for key in erp_dict:
        erp_list = erp_dict[key]

        # Go over all list entries that are not NaN
        erp_idxs = np.nonzero(~pd.isna(erp_list))[0]
        for idx in erp_idxs:

            this_shape = erp_list[idx].data.shape

            # If this is the first ERP we encounter, store its shape
            if erp_shape == DUMMY_SHAPE:
                erp_shape = this_shape

            # Check that all ERPs have the same shape
            if this_shape != erp_shape:
                raise RuntimeError("Encountered ERPs with different shape.")

    return erp_shape


def dict2arr(erp_dict, scaling=1e6):
    """Convert erp_dict to erp_arr.

    Parameters
    ----------
    erp_dict : dict
        Each key is an ERP contrast with a list of ERPs as
        value. The list contains MNE-Python evoked objects
        or np.nan.
    scaling : float
        Amount by which to scale the data. Defaults to 1e6, which
        will result in units "microVolt".

    Returns
    -------
    erp_arr : ndarray
        The ERP data from the `erp_dict` scaled by `scaling`. This
        array is of shape (n_keys, n_channels, n_timepoints, n_subjects).
        For subjects that did not have an ERP for this dict's key (np.nan),
        the array will be np.nan for erp_arr[..., subj].

    """
    nsubjs = _check_nsubjs(erp_dict)
    erp_shape = _check_erp_shape(erp_dict)

    erp_arr = np.nan * np.zeros((len(erp_dict), *erp_shape, nsubjs))
    for key_i, key in enumerate(erp_dict):
        erp_list = erp_dict[key]
        # Go over all list entries that are not NaN
        # leave the rest as NaN
        erp_idxs = np.nonzero(~pd.isna(erp_list))[0]
        for idx in erp_idxs:
            erp_arr[key_i, ..., idx] = erp_list[idx].data * scaling

    return erp_arr


def find_time_idxs(window, times):
    """Find time indices corresponding to a window onset and offset.

    Parameters
    ----------
    window : tuple of len 2
        The beginning and end of a time window in a unit corresponding to
        `times`.
    times : ndarray
        The time samples in some unit.

    Returns
    -------
    time_idxs : ndarray
        The time indices into `times` corresponding to `window`.

    """
    tol = np.mean(np.diff(times))
    time_idxs = list()
    for idx in window:
        if np.abs(times - idx).min() > tol:
            raise RuntimeWarning("`window` is not within `times`.")
        time_idxs.append((np.abs(times - idx)).argmin())

    assert len(time_idxs) == 2
    if time_idxs[0] == time_idxs[1]:
        return np.atleast_1d(time_idxs[0])

    assert time_idxs[0] < time_idxs[1]
    time_idxs = np.arange(*time_idxs)
    return time_idxs


def arr2df(erp_arr, group, window, key_names, ch_names, times):
    """Convert erp_arr to df.

    Parameters
    ----------
    erp_arr : ndarray
        The ERP data of shape (n_keys, n_channels, n_timepoints, n_subjects).
    group : list of str
        A list of channel names over which to average.
    window : tuple of float
        A tuple of length two, determining the start and end ot the time
        window over which to average.
    key_names : list of str
        The names of the different conditions, corresponding to the first
        dimension of `erp_arr`.
    ch_names : list of str
        The channel names, corresponding to the second dimension of
        `erp_arr`.
    times : ndarray
        The time samples, corresponding to the third dimension of `erp_arr`.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe of ERP amplitudes over a `group` of channels, and over a
        `window` of timepoints. Organized by keys and subjects.

    Examples
    --------
    >>> # for using this with the array output of `calc_timecourse`
    >>> corr_arr = np.expand_dims(corr_arr, (0, 1))
    >>> arr2df(corr_arr, ['dummy'], window, ['dummy'], ['dummy'], times)

    """
    # Find channel idxs for group to average
    ch_idxs = np.array([ch_names.index(ch) for ch in group])

    # Find time idxs for window to average over
    time_idxs = find_time_idxs(window, times)

    # Extract mean amplitudes
    nkeys, nchans, ntimes, nsubjs = erp_arr.shape
    keys = list()
    subjs = list()
    amps = list()

    for ikey, key_name in enumerate(key_names):
        key_erp_arr = erp_arr[ikey, ...]
        chn_mean = np.mean(key_erp_arr[ch_idxs, ...], axis=0)
        time_mean = np.mean(chn_mean[time_idxs], axis=0)

        assert time_mean.shape[0] == nsubjs
        keys += [key_name] * nsubjs
        subjs += list(range(nsubjs))
        amps += list(time_mean)

    # Form data frame
    df = pd.DataFrame((keys, subjs, amps)).T
    df.columns = ("key", "subj_repr", "mean_amp")

    df["mean_amp"] = np.asarray(df["mean_amp"].values, dtype=float)

    return df


def get_mahalanobis_rdm_times(
    subj,
    df,
    epochs,
    preproc_settings,
    samp,
    half,
    zscore_before_regress=True,
    add_constant=False,
    do_full_mnn=False,
    outcome_column_name="outcome",
):
    """Calculate the mahalanobis RDM per timepoint for a subject.

    Based on performing a ordinary least squares multiple regression across
    trials for each channel/time bin of the EEG data. Each trial is predicted
    by the condition of the trial, thus the coefficients resemble the
    condition wise ERPs, but subtracted with the overall ERP due to the
    constant term in the regression (or due to zscoring the dependent
    variable prior to regression, or due to both, depending on your
    settings of `zscore_before_regress` and `add_constant`).
    To form an RDM, we go over each timepoint and calculate the
    mahalanobis distance between coefficients
    (shape nconditions x nchannels), normalized by the inverse covariance
    matrix (shape nchannels x nchannels) that we obtain through the
    Ledoit & Wolf method from the regression residuals
    (shape ntrials x nchannels).

    Parameters
    ----------
    subj : int
        The subject ID.
    df : pandas.DataFrame
        The behavioral data.
    epochs : mne.Epochs
        The epochs data associated with subj.
    preproc_settings : dict
        Dict with settings for preprocessing. Accepts keys "crop", "smooth",
        "tshift", "baseline".
    samp : str
        Which kind of sampling epochs to use, can be "active" or "yoked".
    half : str
        Which half of the epochs to use, can be "both", "first_half", or
        "second_half".
    zscore_before_regress : bool
        Whether or not to zscore the channel/time binned EEG data across
        trials prior to regression. Defaults to True.
    add_constant : bool
        Whether or not to add a constant term to the design matrix for
        regression. If True, design matrix will be rank deficient, if False,
        `zscore_before_regress` must be True so that regression data is
        centered. Defaults to False.
    do_full_mnn : bool
        Whether or not to do a multivariate noise normalization based on
        both sampling types (active AND yoked). If False (default), calculate
        noise correction only for the sampling type that is specified via
        the `samp` parameter.
    outcome_column_name : str
        The kind of condition to use for constructing the RDMs. Most often,
        this is "outcome", referring to the numeric samples. But it can also
        be "outcome_bin", or "outcome_bin_orth" to use binned outcomes after
        normalizing them with their trial specific outcome mean, or to
        use them after orthogonalization with the normal "outcomes".

    Returns
    -------
    rdm_times : ndarray, shape(nconditions, nconditions, ntimes)
        The mahalanobis RDM per timepoint.
    coefs : ndarray, shape(ntrials, nchannels, ntimes)
        The regression coefficients from the OLS model on preprocessed epochs
        predicted by the outcomes of each trial. Can be used for double
        checking for similarity against overall mean-subtracted ERPs.

    Notes
    -----
    The design matrix with added constant term is rank deficient, because the
    constant term is a linear combination of all other columns (sum). The
    software packages will warn about that, however the results will be
    relatively stable anyhow. Nevertheless, it is recommended to run
    with ``zscore_before_regress=True`` and ``add_constant=False``.
    That will center the data, but avoid rendering the design matrix
    rank deficient.

    """
    # Check options
    if (not zscore_before_regress) and (not add_constant):
        raise ValueError(
            "One or both of `zscore_before_regress` and `add_constant` must be True."
        )

    good_outcome_cols = ["outcome", "outcome_bin", "outcome_bin_orth"]
    if outcome_column_name not in good_outcome_cols:
        raise ValueError(f"outcome_column_name must be one of '{good_outcome_cols}'")

    # Make sure only EEG channels have been picked
    epochs = epochs.pick_types(meg=False, eeg=True)

    # Select the half split we are interested in (or "both")
    if half != "both":
        assert half in ["first_half", "second_half"]
        epochs = epochs[f"{half}"]

    if not do_full_mnn:
        epochs = epochs[f"{samp}"]

    # Preprocess EEG data
    assert set(preproc_settings.keys()) == set(("crop", "smooth", "baseline", "tshift"))
    epochs = prep_epochs(epochs, **preproc_settings, crossvalidate=False, average=False)

    # Get outcome vector from behavioral data
    # optionally use "outcome_bin" or "outcome_bin_orth"
    # look into epochs.drop_log to drop those rows in behavioral data where
    # epochs have been rejected. Afterwards, the outcome vector must be of
    # same length as the epochs (each epoch associated with one outcome)
    kept_epos = np.asarray([False if len(i) > 0 else True for i in epochs.drop_log])
    all_outcomes = df[(df["subject"] == subj) & (df["task"] != "DESC")][
        outcome_column_name
    ].to_numpy()
    outcomes = all_outcomes[kept_epos]
    unique_outcomes = np.unique(outcomes)
    if not len(unique_outcomes) == 9:
        raise RuntimeError(
            f"No 9 outcomes ({outcome_column_name}) for {subj}/{samp}/{half}: {unique_outcomes}"
        )

    # similarly: get sample type vector (but from epochs data)
    inv_map = {v: k for k, v in epochs.event_id.items()}
    assert len(inv_map) == len(epochs.event_id)

    event_types = [inv_map[i] for i in epochs.events[:, -1]]
    assert len(event_types) == len(epochs)

    sampling_types = ("active", "yoked")
    samplings = []
    for event_type in event_types:
        sampling = event_type.split("/")[0]
        assert sampling in sampling_types
        samplings.append(sampling)

    samplings = np.array(samplings)

    # prepare design matrix
    # ---------------------
    ntrials = len(outcomes)
    assert ntrials == len(epochs)

    if do_full_mnn:
        # full mnn over each task and outcome
        conditions = list(itertools.product(sampling_types, unique_outcomes))
    else:
        # if we do the MNN over a specific task,
        # the predictors are only the outcomes
        conditions = unique_outcomes

    # each column is a predictor
    # the sum over columns for each row is always 1 ... that is, each epoch
    # is predicted by its condition
    # ntrials x nconditions (no "constant" yet, potentially added later)
    # nconditions is either 9 (outcomes),
    # or 18 (active x outcomes + yoked x outcomes)
    nconditions = len(conditions)
    design_matrix = np.zeros((ntrials, nconditions))

    for icondi, condi in enumerate(conditions):

        if do_full_mnn:
            assert isinstance(condi, tuple)
            sampling, outcome = condi
            outcome_idxs = outcomes == outcome
            sampling_idxs = samplings == sampling
            idxs = np.logical_and(outcome_idxs, sampling_idxs)
        else:
            assert isinstance(condi, np.int64)
            idxs = outcomes == condi

        design_matrix[idxs, icondi] = 1

    # sanity check that sum of each row is 1
    assert np.sum(design_matrix.sum(axis=1) == 1) == len(epochs)

    # add constant
    # design matrix X is now ntrials x (nconditions+1)
    if add_constant:
        X = sm.add_constant(design_matrix, has_constant="raise")
    else:
        X = design_matrix

    # get EEG data and scale to micro Volts
    # ntrials x nchannels x ntimes
    eeg_data_uV = epochs.get_data() * 1e6

    # fit ordinary least squares multiple regression for each channel/time bin
    _, nchs, ntimes = eeg_data_uV.shape
    assert _ == ntrials

    coefs = np.zeros((nconditions, nchs, ntimes))
    resids = np.zeros_like(eeg_data_uV)
    for ichannel in range(nchs):
        for itime in range(ntimes):
            data = eeg_data_uV[..., ichannel, itime]

            if zscore_before_regress:
                y = scipy.stats.zscore(data)
            else:
                y = data

            model = sm.OLS(endog=y, exog=X, missing="raise")
            results = model.fit()

            if add_constant:
                # drop intercept term
                coefs[..., ichannel, itime] = results.params[1:]
            else:
                coefs[..., ichannel, itime] = results.params
            resids[..., ichannel, itime] = results.resid

    # If we did a full MNN, we need to subselect the "sampling type" we
    # are interested in
    # NOTE: this turns the nconditions from 18 to 9, fixed to EITHER active
    # OR yoked
    if do_full_mnn:
        # get which part of "coefs" are for active, and which are for yoked
        coef_idxs = {}
        start = 0
        stop = len(np.unique(outcomes))
        assert stop == 9
        for sampling in sampling_types:
            coef_idxs[sampling] = tuple(range(start, stop))
            start += stop
            stop += stop

        # get either only active, or only yoked coefficients
        # NOTE: residuals stay the same for MNN: no subselecting there
        coefs = coefs[np.array(coef_idxs[samp]), ...]
        nconditions = coefs.shape[0]
        assert nconditions == 9

    # Calculate pairwise mahalanobis distance between regression coefficients
    # for each condition, normalized with covariance matrix from regression
    # residuals
    # done for each timepoint, ending up with rdm
    # nconditions x nconditions x ntimes
    rdm_times = np.zeros((nconditions, nconditions, ntimes))
    for itime in range(ntimes):
        response = coefs[..., itime]
        residuals = resids[..., itime]

        # Estimate covariance from residuals
        lw_shrinkage = LedoitWolf(assume_centered=True)
        cov = lw_shrinkage.fit(residuals)  # see cov.covariance_

        # Compute pairwise mahalanobis distances
        VI = np.linalg.inv(cov.covariance_)
        rdm = squareform(pdist(response, metric="mahalanobis", VI=VI))
        assert ~np.isnan(rdm).all()
        rdm_times[..., itime] = rdm

    return rdm_times, coefs


def split_key_col(df, new_col_names):
    """Split the key column in a dataframe into subcolumns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a `key` column containing `/`-separated strs to
        be split into new columns named by `new_col_names`.
    new_col_names : list of str
        New column names.

    Returns
    -------
    df : pandas.DataFrame
        Copy of `df` with additional columns named in `new_col_names`,
        containing values obtained from the `key` column.

    """
    # Work on copy
    df = df.copy()

    # NOTE: depends on ordered dictionaries
    new_cols = {name: list() for name in new_col_names}
    for key in df["key"].to_list():
        splitkey = key.split("/")
        if not len(splitkey) == len(new_col_names):
            raise RuntimeError(
                "`new_col_names` length does not match with "
                "length of each key split by the `/` symbol."
            )
        for name, val in zip(new_col_names, splitkey):
            new_cols[name].append(val)

    prev_idx = df.columns.get_loc("key")
    for name, vals in new_cols.items():
        df.insert(prev_idx + 1, name, vals)
        prev_idx += 1

    return df


def prep_epochs(
    epochs,
    crop,
    smooth,
    tshift,
    baseline,
    crossvalidate,
    average=True,
    smooth_before_baseline=False,
):
    """Preprocess epochs as final step before RSA.

    Returns a preprocessed data object, or two such objects if crossvalidate is True.
    The kind of object is an ERP if `average` is True, else it is an epochs object.
    Works on a copy of `epochs`.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object.
    crop : tuple | False
        If False, do nothing. If tuple, crop epochs to the specified time
        window.
    smooth : float | False
        If False, do nothing. If float, smooth the ERP(s) with a gaussian
        kernal of that length in milliseconds.
    tshift : float | False
        If False, do nothing. If float, shift the ERP relatively by that time
        in seconds.
    baseline : tuple | False
        If False, do nothing. If tuple, apply the baseline accordingly.
    crossvalidate : boolean
        If True, split epochs into two epochs objects with even and odd
        trials. If False, work on the single epochs object.
    average : boolean
        If True, average the epochs into ERPs. Defaults to True.
    smooth_before_baseline : bool
        If True, perform smoothing before baseline correction. If False,
        perform it the other way around. This argument only takes effect
        if both `smooth` and `baseline` are not False.

    Returns
    -------
    mne.Evoked | tuple((mne.Evoked, mne.Evoked)) | mne.Epochs | tuple((mne.Epochs, mne.Epochs))
        Either a single ERP or epochs object, or if `crossvalidate` is True,
        two ERPs or epochs objects. It depends on the `average` parameter whether
        ERPs or epochs are returned. If two objects are returned, they correspond
        to even and odd trials in this exact order.

    """
    # Work on a copy
    epochs = epochs.copy()

    # crop
    if not isinstance(crop, bool):
        assert isinstance(crop, (tuple, list)), f"{type(crop)}"
        assert len(crop) == 2, f"{len(crop)}"
        tmin, tmax = crop  # unpack tuple
        epochs = epochs.crop(tmin=tmin, tmax=tmax, include_tmax=True, verbose=False)

    # shift onset
    if not isinstance(tshift, bool):
        assert isinstance(tshift, (int, float)), f"{type(tshift)}"
        epochs = epochs.shift_time(tshift=tshift, relative=True)

    if smooth_before_baseline:
        # smooth
        if not isinstance(smooth, bool):
            assert isinstance(smooth, (float, int)), f"{type(smooth)}"
            epochs = smooth_eeg(epochs, len_ms=smooth)

        # baseline
        if not isinstance(baseline, bool):
            assert isinstance(baseline, (tuple)), f"{type(baseline)}"
            assert len(baseline) == 2, f"{len(baseline)}"
            epochs = epochs.apply_baseline(baseline=baseline, verbose=False)
    else:
        # baseline
        if not isinstance(baseline, bool):
            assert isinstance(baseline, (tuple)), f"{type(baseline)}"
            assert len(baseline) == 2, f"{len(baseline)}"
            epochs = epochs.apply_baseline(baseline=baseline, verbose=False)

        # smooth
        if not isinstance(smooth, bool):
            assert isinstance(smooth, (float, int)), f"{type(smooth)}"
            epochs = smooth_eeg(epochs, len_ms=smooth)

    # crossvalidate
    if crossvalidate:
        # split into even/uneven
        all_epochs = [epochs[0::2], epochs[1::2]]
    else:
        all_epochs = [epochs]

    # average
    if average:
        all_objects = [epo.average() for epo in all_epochs]
    else:
        all_objects = [epo for epo in all_epochs]

    # return either one or two ERPs depending on `crossvalidate`
    if crossvalidate:
        assert len(all_objects) == 2
        return all_objects[0], all_objects[1]

    assert len(all_objects) == 1
    return all_objects[0]


def calc_sigma(
    epochs, preproc_settings, crossvalidate, scaletomu, conditions, average_over_time
):
    """Calculate roots of covariance matrices for epochs.

    The output can be used for multivariate noise normalization (MNN) of
    condition wise data.

    Parameters
    ----------
    epochs : mne.Epochs
        The data.
    preproc_settings : dict
        The dictionary of preprocessing settings passed to `prep_epochs`.
        Must not contain a key called "crossvalidate".
        Must contain a key "average" set to False.
    crossvalidate : bool
        Whether or not to work on crossvalidated RDMs.
    scaletomu : bool
        Whether or not to scale EEG data to micro Volts. If not, leave them
        in Volts.
    conditions : list of str
        Over which conditions present in `epochs` to separately compute
        covariance matrices before averaging them.
    average_over_time : bool
        If True, average the covariance matrices over time, and return
        value(s) will be of shape (nchs, nchs). If False, keep the time
        dimension and return value(s) will be of shape (nchs, nchs, ntimes).

    Returns
    -------
    sigmas : list of ndarray, shape(nchs, nchs) | shape (nchs, nchs, ntimes)
        Ndarrays for either even and odd epochs (in that order) if
        `crossvalidate` is ``True``, or an ndarray for all epochs if
        `crossvalidate` is ``False``. Ndarrays correspond to the covariance
        matrix raised to the power of negative one half. If `average_over_time`
        was False, the ndarrays are per timepoint.

    Notes
    -----
    This is working on a copy of epochs. The epochs passed as a parameter
    are not changed.

    Examples
    --------
    >>> epochs = mne.read_epochs(...)
    >>> epochs = epochs.pick_types(meg=False, eeg=True)
    >>> crossvalidate = True
    >>> preproc_settings = {...}
    >>> conditions = [f"out{i}" for i in range(1, 10)]
    >>> average_over_time = True
    >>> sigmas_root = calc_sigma(epochs, preproc_settings, crossvalidate,
    >>>                          conditions, average_over_time)
    >>> len(sigmas_root)
    2
    >>> # Use sigmas to normalize data
    >>> X_even = ...  # the overall epochs for even epochs
    >>> sigma_even = sigmas[0]  #
    >>> X_norm_even = (X.swapaxes(1, 2) @ sigma).swapaxes(1, 2)
    >>> # repeat the same for odd ...
    """
    epochs = epochs.copy()

    # preprocess epochs
    # all_epochs is either of length 1 if crossvalidate=False
    # or of length 2 otherwise. If 2 epochs objects are in all_epochs, then
    # the first one will contain all even trials, and the second one will
    # contain all odd trials
    if preproc_settings.get("average", None) is True:
        raise ValueError("Must pass preproc_settings with average=False.")

    if crossvalidate:
        all_epochs = prep_epochs(
            epochs, **preproc_settings, crossvalidate=crossvalidate
        )
    else:
        epochs = prep_epochs(epochs, **preproc_settings, crossvalidate=crossvalidate)
        all_epochs = [epochs]

    tmpdata = all_epochs[0].get_data()
    assert tmpdata.ndim == 3, "must be trls x chs x times"
    _, nchs, ntimes = tmpdata.shape
    sigmas = []
    for epochs in all_epochs:

        # Calculate covariance for each conditions separately
        if average_over_time:
            sigma_c = np.full((len(conditions), nchs, nchs), np.nan)
        else:
            sigma_c = np.full((len(conditions), nchs, nchs, ntimes), np.nan)
        for icondi, condi in enumerate(conditions):
            # get data from full epoch
            X = epochs[condi].get_data()

            if scaletomu:
                X = X * 1e6

            # calculate covariance for each timepoint
            sigma_c_t = np.full((nchs, nchs, ntimes), np.nan)
            for t in range(ntimes):

                # computing covariance using Ledoit & Wolf method
                # standard sklearn way somehow leads to complex
                # data output once the fractional_matrix_power
                # is taken further below.
                # using the private _cov function solves that problem
                # and is employed in code by guggenmos 2018
                # standard sklearn way:
                # lw_shrinkage = LedoitWolf()
                # cov = lw_shrinkage.fit(X[..., t])
                # sigma_c_t[..., t] = cov.covariance_

                # see: https://github.com/m-guggenmos/megmvpa/blob/master/tutorial_python/python_reliability.ipynb
                sigma_c_t[..., t] = _cov(X[..., t], shrinkage="auto")

            # take mean over timepoints
            if average_over_time:
                sigma_c[icondi, ...] = np.mean(sigma_c_t, axis=-1)
            else:
                sigma_c[icondi, ...] = sigma_c_t.copy()

        # take mean over conditions
        sigma = np.mean(sigma_c, axis=0)
        if average_over_time:
            sigma = scipy.linalg.fractional_matrix_power(sigma, -0.5)
        else:
            for t in range(ntimes):
                sigma[..., t] = scipy.linalg.fractional_matrix_power(
                    sigma[..., t], -0.5
                )
        sigmas.append(sigma)

    return sigmas


def get_model_rdm_dict(sample_frequencies, double, lower_tri, mean_center):
    """Get model RDMs.

    All RDMs will be normalized to the range [0, 1]. This normalization
    happens before the optional mean centering.

    Parameters
    ----------
    sample_frequencies : ndarray, shape(9,) | shape(18,)
        Array to be used for creating sample_frequency RDM.
    double : bool
        Whether or not to get an 18x18 instead of a 9x9 RDM.
    lower_tri : bool
        If False, get the square RDM, if True, get a vector of the lower
        triangle excluding the diagonal.
    mean_center : bool
        Whether or not to mean center each RDM.

    Returns
    -------
    model_rdms : dict
        The model RDMs, or vectors of the lower tri excluding diag.

    """
    model_rdms = {
        "side": None,
        "identity": None,
        "numberline": None,
        "extremity": None,
        "sample_frequency": None,
    }

    if not double:
        # "side" is only possible in double RDMs (18x18)
        del model_rdms["side"]

    for model in model_rdms:
        to_get = model
        if model == "sample_frequency":
            to_get = sample_frequencies
        rdm = get_model_rdm(to_get, normalize=True, double=double)
        if lower_tri:
            rdm = _rdm2vec(rdm, lower_tri=True)
        if mean_center:
            rdm = rdm - rdm.mean()
        model_rdms[model] = rdm
    return model_rdms


def orth_model_rdms(model_rdms, exclude_from_orth=None):
    """Orthogonalize RDMs in model_rdms.

    Parameters
    ----------
    model_rdms : dict
        The model RDMs, or vectors of the lower tri excluding diag.
    exclude_from_orth : list of str | None
        Model names to exclude from orthogonalization. If None (default),
        do nothing.

    Returns
    -------
    model_rdms : dict
        The model RDMs with added keys prepended with ``o_`` for their
        orthogonalized versions.
    """
    # Get all RDM names
    names = list(model_rdms.keys())[::-1]
    if isinstance(exclude_from_orth, list):
        for exclude_name in exclude_from_orth:
            names.remove(exclude_name)

    # Return early if there is nothing to othogonalize
    if len(names) < 2:
        return model_rdms

    def rotate(names, i):
        """Rotate an iterable by `i` positions."""
        return names[-i:] + names[:-i]

    # Orthogonalize each RDM
    SHAPE_DUMMY = 1
    for i in range(0, len(names)):
        names_i = rotate(names, i)
        models = []
        prev_shape = SHAPE_DUMMY
        for name in names_i:
            model = model_rdms[name].copy()

            if prev_shape == SHAPE_DUMMY:
                prev_shape = model.shape
            else:
                assert prev_shape == model.shape, "Mixture of shapes seen."

            models.append(model.reshape(-1, 1))

            if not np.isclose(model.mean(), 0):
                warnings.warn(f"Nonzero mean for {name}: {model.mean()}")

        X = np.hstack(models)
        assert X.shape[-1] == len(names), "columns should be RDM types"
        X_orth = spm_orth(X)

        # The last column is the orthogonalized one in spm_orth
        # add to model_rdms dict
        name_orth = names_i[-1]
        model_rdms[f"o_{name_orth}"] = X_orth[:, -1].reshape(*prev_shape)

    return model_rdms


def get_model_rdm(kind, normalize=False, double=False):
    """Return a model representational dissimilarity matrix.

    Parameters
    ----------
    kind : str | ndarray, shape(9,) | shape(18,)
        If str, must be one of
        ['identity', 'side', 'numberline', 'extremity'].
        If ndarray, the RDM will be calculated through
        pairwise absolute differences between the items in
        the ndarray.

        .. note:: passing a ndarray for `kind` can be
                  helpful to construct an RDM based on
                  sample frequencies.

    normalize : bool
        Whether or not to normalize the output `rdm` to
        the range [0, 1] through ``rdm /= rdm.max()``.
        Note that this only works work RDMs that already
        have their minimum at 0. The function will raise
        an AssertionError otherwise.
    double : bool
        Whether or not to get an 18x18 instead of the
        default 9x9 RDM through ``np.tile(rdm, (2, 2))``.

    Returns
    -------
    rdm : ndarray
        The model RDM.

    """
    do_calc = True
    skip_double = False
    if isinstance(kind, np.ndarray):
        xx = kind.copy()
        assert xx.shape in [(9,), (18,)], "need shape (9,) or (18,)"
        if xx.shape == (18,):
            skip_double = True
    elif kind == "identity":
        rdm = np.abs(np.identity(9) - 1)
        do_calc = False
    elif kind == "side":
        if not double:
            raise ValueError('double must be True for kind=="side".')
        rdm = np.repeat(np.repeat(np.abs(np.identity(2) - 1), 9, axis=1), 9, axis=0)
        do_calc = False
        skip_double = True
    elif kind == "numberline":
        xx = np.abs(np.arange(1, 10, 1))
    elif kind == "extremity":
        xx = np.abs(np.arange(-4, 5, 1))
    else:
        raise ValueError(f'Could not handle kind: "{kind}"')

    if do_calc:
        arrs = list()
        for num in xx:
            arrs.append(np.abs(xx - num))
        rdm = np.stack(arrs, axis=0)
    if normalize:
        rdm = rdm / rdm.max()
        assert np.isclose(rdm.min(), 0)
        assert np.isclose(rdm.max(), 1)
    if double and not skip_double:
        rdm = np.tile(rdm, (2, 2))
    return rdm


def _rdm2vec(rdm, lower_tri=True):
    """Get an RDM as a vector.

    Parameters
    ----------
    rdm : ndarray, shape(n, n)
        The representational distance matrix.
    lower_tri : bool
        If True, return only the lower triangle without the
        diagonal as an output. If False, return the full
        RDM as an output.

    Returns
    -------
    vector : ndarray, shape(n, )
        Copy of either the full RDM as a vector or the
        lower triangle without diagonal as a vector.

    """
    assert rdm.ndim == 2
    assert rdm.shape[0] == rdm.shape[1]
    rdm = np.asarray(rdm.copy(), dtype=float)
    if lower_tri:
        lower_triangle_idx = np.tril_indices(rdm.shape[0], k=-1)
        vector = rdm[lower_triangle_idx].flatten()
    else:
        vector = rdm.flatten()
    return vector


def smooth_eeg(eeg, len_ms, alpha=2.5):
    """Smooth EEG data inplace using a gaussian kernel of length `len_ms`.

    The standard deviation of the kernel is picked such that the gaussian
    covers the whole window, see [1]_, [2]_, [3]_, and [4]_.

    Parameters
    ----------
    eeg : mne.evoked | mne.epochs
        The eeg data object with the data to be smoothed.
    len_ms : int
        Length of the gaussian window in milliseconds.
    alpha : float
        The width factor, inversely proportional to the width of the gaussian
        window: std = (len_of_window - 1) / (2 * alpha). Defaults to 2.5, as
        that is the same value used in Matlab's "gausswin" function.

    Returns
    -------
    eeg : mne.evoked | mne.epochs
        The smoothed eeg data object.

    Notes
    -----
    If evoked is passed, smoothing happens per channel. If epochs are passed,
    smoothing happens per trial and per channel.

    To calculate the Full Width Half Maximum (FWHM) statistic of the smoothing
    window, the following two formulae can be used:

    >>> def sigma2fwhm(sigma):
    ...     return sigma * np.sqrt(8 * np.log(2))

    >>> def fwhm2sigma(fwhm):
    ...     return fwhm / np.sqrt(8 * np.log(2))

    References
    ----------
    .. [1] https://stackoverflow.com/a/16167851/5201771
    .. [2] https://de.mathworks.com/help/signal/ref/gausswin.html
    .. [3] Wallisch et al. (2014), "Matlab for Neuroscientists",
           2nd Edition, pp. 319-320
    .. [4] https://matthew-brett.github.io/teaching/smoothing_intro.html

    """
    sfreq = eeg.info["sfreq"]
    len_samples = len_ms / (1000.0 / sfreq)

    # round length of window to next odd integer
    len_samples = int(np.ceil(len_samples))
    if len_samples % 2 == 0:
        len_samples += 1

    # Calculate std such that the gaussian covers the whole window
    std_gaussian = (len_samples - 1) / (2 * alpha)

    # Create kernel and normalize so it sums to 1
    kernel = scipy.signal.windows.gaussian(len_samples, std_gaussian)
    kernel /= kernel.sum()

    # smooth the data
    # if evoked, do per channel
    if "mne.evoked" in str(type(eeg)):
        nchs, ntimes = eeg.data.shape
        for ch in range(nchs):
            signal = eeg.data[ch, ...]
            eeg.data[ch, ...] = np.convolve(signal, kernel, mode="same")

    # if epochs, do per trial and channel
    elif "mne.epochs" in str(type(eeg)):
        ntrls, nchs, ntimes = eeg._data.shape
        for trl in range(ntrls):
            for ch in range(nchs):
                signal = eeg._data[trl, ch, ...]
                eeg._data[trl, ch, ...] = np.convolve(signal, kernel, mode="same")

    else:
        raise RuntimeError(f"Could not triage type of `eeg`: {str(type(eeg))}")

    return eeg


def _check_dict_for_nan(erp_dict):
    """Check whether values of a dict contain missings."""
    for key in erp_dict:
        if any(pd.isna(erp_dict[key])):
            warnings.warn(
                'Encountered NaN in values for key "{}". '
                "Trying to handle it, but be aware".format(key)
            )


def _parallel_load_and_crop(subj, name_templates, crop):
    """Help loading and cropping subject data in parallel."""
    # See _load_and_crop
    fname_epo = name_templates["epochs"].format(subj)
    epochs = mne.read_epochs(fname_epo, preload=True, verbose=0)
    if crop:
        epochs.crop(*crop)
    return epochs


def _load_and_crop(subjects, name_templates, crop):
    """Load and crop subject data in parallel."""
    _partial_parralel = partial(
        _parallel_load_and_crop, name_templates=name_templates, crop=crop
    )

    with multiprocessing.Pool() as pool:
        results = pool.map(_partial_parralel, subjects)

    loaded_data = dict()
    for subj, epochs in zip(subjects, results):
        loaded_data[subj] = epochs

    return loaded_data


def get_erps_by_dict(
    erp_dict,
    name_templates,
    subjects,
    subtract=False,
    min_epos=0,
    crop=False,
    tshift=0.0,
    smooth=False,
    baseline=False,
    smooth_before_baseline=False,
    double_subtraction=False,
):
    """Get a dict of subject wise ERPs per condition.

    Parameters
    ----------
    erp_dict : dict
        A dict with the keys for which ERPs should be computed. For
        each key, the corresponding value is an empty list that will be
        filled with the subject keys. Use the key "all" to get all ERPs.
    name_templates : dict
        Needs to have at least one key "epochs" containing a string template
        pointing to the epochs path with one field to be filles by a subject
        integer, e.g., 'home/sub-{0:02}_epochs-epo.fif.gz'.
    subjects : list
        Subjects to work on.
    subtract : tuple | False
        If a tuple of len==2 is supplied, the ERP will be a difference wave
        of epochs[subtract[0]].average() - epochs[subtract[1]].average()
    min_epos : int
        Minimum number of epochs for a subject to be included. If less epochs
        are found, the subject's ERP is set to np.nan. Defaults to 0, i.e.,
        no minimum number of epochs.
    crop : tuple | False
        If a tuple is supplied, the data will be cropped to that range.
        For example (0.4, 1.6).
    tshift : float
        Seconds to shift the times of the data to center the data on
        a different event, see `mne.Evoked.shift_time`.
    smooth : int | False
        If an int is supplied, the data will be smoothed with a gaussian
        kernel with the length of that int in milliseconds.
    baseline : tuple | False
        If a tuple is supplied, apply it as a baseline correction. For
        example (None, 0).
    smooth_before_baseline : bool
        If True, perform smoothing before baseline correction. If False,
        perform it the other way around. This argument only takes effect
        if both `smooth` and `baseline` are not False.
    double_subtraction : bool
        If False, this parameter has no impact on the function (default).
        If True, `subtraction` must be set to a tuple ``('left', 'right')``.
        Then, a "double subtraction" [1]_ will be performed, that is after
        the ERP of all "right" epochs has been subtracted from the ERP of
        all "left" epochs, the resulting lateralized ERP is further treated
        as follows: Using the occipital and parietal channels on the left
        ("O1", "PO3", "PO7", "PO9") and right ("O2", "PO4", "PO8", "PO10")
        side of the scalp, two new average channels are calculated. In
        a second step, these two average channels (ch_left and ch_right)
        are subtracted from one another (ch_right minus ch_left). The
        resulting ERP will only have one channel, the double subtracted
        lateralized channel made out of all occipital/parietal channels.

    Returns
    -------
    erp_dict : dict
        The dict with the values (lists) filled up with ERPs.
        This is good for plotting using mne.viz.plot_compare_evokeds().

    References
    ----------
    .. [1] Eimer, M. The lateralized readiness potential as an on-line
       measure of central response activation processes.
       Behavior Research Methods, Instruments, & Computers 30, 146–156 (1998).
       https://doi.org/10.3758/BF03209424

    """
    # load and crop data in parallel
    loaded_data = _load_and_crop(subjects, name_templates, crop)

    # Now do the rest subject by subject
    for subj in tqdm(subjects):

        epochs = loaded_data[subj]

        for key in erp_dict.keys():

            # Skip, if this task is not present for this subj (between design)
            if task_not_present_for_subject(subj, key):
                continue

            try:
                epo_selection = epochs[key] if key != "all" else epochs
                no_epos = len(epo_selection) == 0
            except KeyError:
                no_epos = True

            # if we do not have epochs for this key, append NaN and skip
            if no_epos or len(epo_selection) < min_epos:
                erp_dict[key].append(np.nan)
                continue

            if not subtract:
                erp = epo_selection.average()
                erp.comment = key
            else:
                all_evoked = [
                    epo_selection[subtract[0]].average(),
                    epo_selection[subtract[1]].average(),
                ]
                erp = mne.combine_evoked(all_evoked, weights=[1, -1])
                erp.comment = "{1}/{0} - {2}/{0}".format(key, *subtract)

            if double_subtraction:
                assert subtract == ("left", "right")
                picks_left = mne.pick_channels(
                    erp.ch_names, include=["O1", "PO3", "PO7", "PO9"]
                )
                picks_right = mne.pick_channels(
                    erp.ch_names, include=["O2", "PO4", "PO8", "PO10"]
                )
                erp = mne.channels.combine_channels(
                    inst=erp,
                    groups=dict(ch_left=picks_left, ch_right=picks_right),
                    method="mean",
                )
                erp = mne.channels.combine_channels(
                    inst=erp,
                    groups=dict(opo=[1, 0]),  # right - left
                    method=lambda data: data[0, :] - data[1, :],
                )
                erp.comment = "double subtracted occipital parietal: left-right"

            erp.shift_time(tshift, relative=True)

            if smooth_before_baseline:
                if smooth:
                    erp = smooth_eeg(erp, smooth)

                if baseline:
                    erp.apply_baseline(baseline, verbose=0)
            else:
                if baseline:
                    erp.apply_baseline(baseline, verbose=0)

                if smooth:
                    erp = smooth_eeg(erp, smooth)

            erp_dict[key].append(erp)

    # Check for NaN
    _check_dict_for_nan(erp_dict)

    return erp_dict


def task_not_present_for_subject(subject, task):
    """Return early if requested task is not present for subject."""
    # only even id participants did the optional stopping task.
    even_id = subject % 2 == 0
    if even_id:
        if "fixed" in task.lower():
            return True
    else:
        if "variable" in task.lower():
            return True
    return False


def provide_trigger_dict():
    """Provide a dictionnary mapping str names to byte values [1]_.

    References
    ----------
    .. [1] https://github.com/sappelhoff/sp_experiment/blob/master/sp_experiment/define_ttl_triggers.py  # noqa: E501

    """
    trigger_dict = OrderedDict()

    # At the beginning and end of the experiment ... take these triggers to
    # crop the meaningful EEG data. Make sure to include some time BEFORE and
    # AFTER the triggers so that filtering does not introduce artifacts into
    # important parts.
    trigger_dict["trig_begin_experiment"] = bytes([1])
    trigger_dict["trig_end_experiment"] = bytes([2])

    # Indication when a new trial is started
    trigger_dict["trig_new_trl"] = bytes([3])

    # Wenever a new sample within a trial is started (fixation stim)
    trigger_dict["trig_sample_onset"] = bytes([4])

    # Whenever a choice is being inquired during sampling
    trigger_dict["trig_left_choice"] = bytes([5])
    trigger_dict["trig_right_choice"] = bytes([6])
    trigger_dict["trig_final_choice"] = bytes([7])

    # When displaying outcomes during sampling
    trigger_dict["trig_mask_out_l"] = bytes([8])
    trigger_dict["trig_show_out_l"] = bytes([9])
    trigger_dict["trig_mask_out_r"] = bytes([10])
    trigger_dict["trig_show_out_r"] = bytes([11])

    # Indication when a final choice is started
    trigger_dict["trig_new_final_choice"] = bytes([12])

    # Whenever a final choice is started (fixation stim)
    trigger_dict["trig_final_choice_onset"] = bytes([13])

    # Inquiring actions during CHOICE
    trigger_dict["trig_left_final_choice"] = bytes([14])
    trigger_dict["trig_right_final_choice"] = bytes([15])

    # Displaying outcomes during CHOICE
    trigger_dict["trig_mask_final_out_l"] = bytes([16])
    trigger_dict["trig_show_final_out_l"] = bytes([17])
    trigger_dict["trig_mask_final_out_r"] = bytes([18])
    trigger_dict["trig_show_final_out_r"] = bytes([19])

    # trigger for ERROR, when a trial has to be reset
    # (ignore all markers prior to this marker within this trial)
    trigger_dict["trig_error"] = bytes([20])

    # If the subject sampled a maximum of steps and now wants to take yet
    # another one, we force stop and initiate a final choice
    trigger_dict["trig_forced_stop"] = bytes([21])

    # If subject tried to make a final choice before taking at least one sample
    trigger_dict["trig_premature_stop"] = bytes([22])

    # Display the block feedback
    trigger_dict["trig_block_feedback"] = bytes([23])

    return trigger_dict


def remove_error_rows(df):
    """Identify error rows and remove them.

    Identify errors via an event value and discard all rows and including the
    error row within the trial that the error happened. Also return a summary
    of errors per trial.

    Parameters
    ----------
    df : pandas.DataFrame
        The original data with a 'trial' and 'value' column

    Returns
    -------
    df : pandas.DataFrame
        The original df with the trials containing errors remove
    error_dict : dict
        Dictionary mapping from trial to number of errors encountered

    """
    trigger_dict = provide_trigger_dict()
    error_trig = ord(trigger_dict["trig_error"])
    error_idx = df.index[df["value"] == error_trig].to_numpy()
    error_trls = df["trial"][error_idx].to_numpy()
    error_dict = dict(zip(*np.unique(error_trls, return_counts=True)))

    remove_idx = list()
    for idx, trl in zip(error_idx, error_trls):
        __ = np.logical_and(df["trial"] == trl, df.index <= idx)
        remove_idx += df.index[__].to_list()

    df = df.drop(remove_idx)
    return df, error_dict


def events_to_behav_data(fpath, mask_event_i=1, outcome_event_i=2):
    """Convert the complete log events to a reasonable dataframe.

    Parameters
    ----------
    fpath : str
        String to the events.tsv file
    mask_event_i, outcome_event_i : int
        How many events after a sample is the outcome or mask presented.
        Assumed to be stable and true and also valid for final choices.
        Defaults to 1 for mask and 2 for outcome.

    Returns
    -------
    beh_df : pandas.DataFrame
        The behavioral data in a neat format focused on behaviorally relevant
        events instead of all events

    Notes
    -----
    For trials that contain an error, all information before that error is
    discarded. The `nerrors` column describes the errors encountered per trial.

    See also
    --------
    remove_error_rows : removing rows marked as erroneous

    """
    # Load data
    df = pd.read_csv(fpath, sep="\t")

    # Remove the error rows and note errors per trial
    df, error_dict = remove_error_rows(df)

    # Check: if only "final_choice" action types are in the df, we are dealing
    # with a dataset from description task.
    if "sample" not in df["action_type"].to_list():
        # drop non-trial rows
        df = df[~df["trial"].isna()]

        # extract variables of interest
        outcomes = df["outcome"].dropna().to_numpy()
        actions = df["action"].dropna().to_numpy()
        rts = df["response_time"].dropna().to_numpy()

        # prepare a new dataframe and subselect columns of interest
        beh_df = df.drop_duplicates(subset="trial", keep="first")
        beh_df = beh_df.loc[
            :,
            [
                "trial",
                "mag0_1",
                "prob0_1",
                "mag0_2",
                "prob0_2",
                "mag1_1",
                "prob1_1",
                "mag1_2",
                "prob1_2",
            ],
        ]
        # there are no "samples" in description ... set to zero
        beh_df["sample"] = 0
        beh_df["rt"] = rts
        beh_df["action"] = actions
        beh_df["outcome"] = outcomes
        beh_df["exp_ev0"] = np.nansum(
            (
                beh_df["mag0_1"] * beh_df["prob0_1"],
                beh_df["mag0_2"] * beh_df["prob0_2"],
            ),
            axis=0,
        )
        beh_df["exp_ev1"] = np.nansum(
            (
                beh_df["mag1_1"] * beh_df["prob1_1"],
                beh_df["mag1_2"] * beh_df["prob1_2"],
            ),
            axis=0,
        )
        # experienced and true EVs are the same
        beh_df["true_ev0"] = beh_df["exp_ev0"].to_numpy()
        beh_df["true_ev1"] = beh_df["exp_ev1"].to_numpy()
        # rts, actions, outcomes are the same for final and "sample" choice
        beh_df["fin_rt"] = rts
        beh_df["fin_action"] = actions
        beh_df["fin_outcome"] = outcomes
        beh_df["nerrors"] = 0

        # For the description task, we don't bother about
        # timestamps for the eyetracker right now
        beh_df["timestamp_sample"] = np.nan
        beh_df["timestamp_mask"] = np.nan
        beh_df["timestamp_outcome"] = np.nan

        mycols = [
            "trial",
            "sample",
            "rt",
            "action",
            "outcome",
            "exp_ev0",
            "exp_ev1",
            "true_ev0",
            "true_ev1",
            "fin_rt",
            "fin_action",
            "fin_outcome",
            "nerrors",
            "mag0_1",
            "prob0_1",
            "mag0_2",
            "prob0_2",
            "mag1_1",
            "prob1_1",
            "mag1_2",
            "prob1_2",
            "timestamp_sample",
            "timestamp_mask",
            "timestamp_outcome",
        ]

        beh_df = beh_df.loc[:, mycols]
        beh_df = beh_df.reset_index(drop=True)

        # specify data types
        intcols = ["trial", "action", "outcome", "fin_action", "fin_outcome"]
        beh_df[intcols] = beh_df[intcols].astype(int)

        # Insert a column with number of samples per trial
        ss = beh_df.groupby("trial").size()
        ss = ss.rename("n_samples")
        beh_df = pd.merge(beh_df, ss, on="trial", validate="m:1")

        return beh_df

    # Start preparing a neat dataframe
    # ...staring with samples only
    not_samples_idx = df.index[~(df["action_type"] == "sample")]
    beh_df = df.drop(not_samples_idx)

    # Add the timestamp for each sample for the eyetracker
    beh_df["timestamp_sample"] = beh_df["system_time_stamp"].to_numpy()

    # Add the timestamp for the mask onset for the eyetracker
    mask_idxs = beh_df.index + mask_event_i
    assert all(df["duration"][mask_idxs] == 0.8)  # sanity check
    beh_df["timestamp_mask"] = df["system_time_stamp"][mask_idxs].to_numpy()

    # add outcomes
    outcome_idxs = beh_df.index + outcome_event_i
    assert all(df["duration"][outcome_idxs] == 0.5)  # sanity check
    outcomes = df["outcome"][outcome_idxs].to_numpy()
    beh_df.loc[:, "outcome"] = outcomes

    # Add the timestamp for the outcome presentation for the eyetracker
    beh_df["timestamp_outcome"] = df["system_time_stamp"][outcome_idxs].to_numpy()

    # add index for samples
    beh_df["sample"] = 0
    trls, counts = np.unique(beh_df["trial"], return_counts=True)
    for trl, count in zip(trls, counts):
        beh_df.loc[beh_df["trial"] == trl, "sample"] = range(count)

    # Calculate "experienced" expected value
    # of option 0 (left) and option 1 (right)
    # NOTE: If for a trial only one action (0, or 1) was available, the
    # corresponding EV entry in that cell will be np.nan
    group_evs = beh_df.groupby(["trial", "action"])["outcome"].mean()
    long_evs = group_evs.reset_index(level=[0, 1])
    wide_evs = long_evs.pivot(index="trial", columns="action")
    wide_evs = wide_evs.reset_index()
    wide_evs.columns = ["trial", "exp_ev0", "exp_ev1"]

    # Calculate "true" expected values and add them to wide df
    # and add mags and probs as well
    __ = df[df["trial"].notnull()]
    __ = __.drop_duplicates(subset="trial")

    # rename by prepending an "a" so we do not overwrite existing columns
    wide_evs["amag0_1"] = __["mag0_1"].to_numpy()
    wide_evs["aprob0_1"] = __["prob0_1"].to_numpy()
    wide_evs["amag0_2"] = __["mag0_2"].to_numpy()
    wide_evs["aprob0_2"] = __["prob0_2"].to_numpy()
    wide_evs["amag1_1"] = __["mag1_1"].to_numpy()
    wide_evs["aprob1_1"] = __["prob1_1"].to_numpy()
    wide_evs["amag1_2"] = __["mag1_2"].to_numpy()
    wide_evs["aprob1_2"] = __["prob1_2"].to_numpy()

    __ = __.loc[:, "mag0_1":"prob1_2"].to_numpy()
    ev0 = __[:, 0] * __[:, 1] + __[:, 2] * __[:, 3]
    ev1 = __[:, 4] * __[:, 5] + __[:, 6] * __[:, 7]
    wide_evs["true_ev0"] = ev0
    wide_evs["true_ev1"] = ev1

    # and merge info into sample info, repeating rows
    beh_df = beh_df.merge(wide_evs, on="trial", validate="many_to_one")

    # get final choice info per trial
    # and merge info into sample info, repeating rows
    not_finchoice_idx = df.index[~(df["action_type"] == "final_choice")]
    only_finchoice_df = df.drop(not_finchoice_idx)
    outcome_idxs = only_finchoice_df.index + outcome_event_i
    outcomes = df["outcome"][outcome_idxs].to_numpy()
    only_finchoice_df.loc[:, "outcome"] = outcomes
    only_finchoice_df = only_finchoice_df.loc[
        :, ["trial", "response_time", "action", "outcome"]
    ]
    only_finchoice_df.columns = ["trial", "fin_rt", "fin_action", "fin_outcome"]
    beh_df = beh_df.merge(only_finchoice_df, on="trial", validate="many_to_one")

    # rename response time --> rt
    beh_df.rename(columns={"response_time": "rt"}, inplace=True)

    # add number of errors per trial, as computed above
    beh_df["nerrors"] = 0
    for trl, nerrors in error_dict.items():
        idxs = beh_df.index[beh_df["trial"] == trl].to_numpy()
        beh_df.loc[idxs, "nerrors"] = nerrors

    # select only relevant columns
    mycols = [
        "trial",
        "sample",
        "rt",
        "action",
        "outcome",
        "exp_ev0",
        "exp_ev1",
        "true_ev0",
        "true_ev1",
        "fin_rt",
        "fin_action",
        "fin_outcome",
        "nerrors",
        "amag0_1",
        "aprob0_1",
        "amag0_2",
        "aprob0_2",
        "amag1_1",
        "aprob1_1",
        "amag1_2",
        "aprob1_2",
        "timestamp_sample",
        "timestamp_mask",
        "timestamp_outcome",
    ]
    beh_df = beh_df.loc[:, mycols]

    # rename magnitude names
    rename_dict = {
        "amag0_1": "mag0_1",
        "aprob0_1": "prob0_1",
        "amag0_2": "mag0_2",
        "aprob0_2": "prob0_2",
        "amag1_1": "mag1_1",
        "aprob1_1": "prob1_1",
        "amag1_2": "mag1_2",
        "aprob1_2": "prob1_2",
    }
    beh_df.rename(columns=rename_dict, inplace=True)

    # specify data types
    intcols = [
        "trial",
        "action",
        "outcome",
        "fin_action",
        "fin_outcome",
        "timestamp_sample",
        "timestamp_mask",
        "timestamp_outcome",
    ]
    beh_df[intcols] = beh_df[intcols].astype(int)

    # Insert a column with number of samples per trial
    ss = beh_df.groupby("trial").size()
    ss = ss.rename("n_samples")
    beh_df = pd.merge(beh_df, ss, on="trial", validate="m:1")

    return beh_df


def get_balance_df(df):
    """Maka a dataframe that shows stimulus balancing.

    Parameters
    ----------
    df : pandas.DataFrame | list of data frames
        The data as prepared by events_to_behav_data.

    Returns
    -------
    df : pandas.DataFrame
        The balance data frame with columns 'side', 'number', 'count', and
        optionally. If a list of dfs was supplied as input, the output df
        contains multiple instances of each side and number.

    Notes
    -----
    Can be plotted nicely with one of the following commands:
    `sns.barplot(x='number', y='count', hue='side', data=balance_df)`
    or sns.boxplot, if a list of dfs was supplied as an input

    See also
    --------
    events_to_behav_data : For preprocessing of raw events
    calc_diff_balance    : For more robust group data visualization

    """
    # Convert to list if not already a list
    if not isinstance(df, list):
        df = [df]

    # Prepare data array to be filled: 9 stimuli on 2 sides = 18
    # and we have three columns: 'side', 'number', 'count'
    n_dfs = len(df)
    datas = np.empty((18, 3, n_dfs))
    for i, df_i in enumerate(df):
        # Aggregate counts of outcomes per selected option
        counts = df_i.groupby(["action", "outcome"]).count()["trial"].to_numpy()

        # Formatting for dataframe
        numbers = np.tile(np.arange(1, 10), 2)
        sides = np.hstack([np.zeros(9), np.ones(9)])
        datas[:, :, i] = np.vstack([sides, numbers, counts]).T

    # collapse 3rd dim again
    data = np.vstack([datas[..., i] for i in range(n_dfs)])

    # Make a dataframe
    balance_df = pd.DataFrame(data, columns=["side", "number", "count"], dtype=int)

    return balance_df


def calc_diff_balance(balance_df):
    """Calculate diff and mean deviation of stimulus balance.

    Parameters
    ----------
    balance_df : pandas.DataFrame
        Dataframe containing the stimulus balancing

    Returns
    -------
    collapsed : pandas.DataFrame
        Dataframe with columns 'diff' and 'mean_deviations'

    See also
    --------
    get_balance_df : get the stimulus balance of numbers and sides

    Examples
    --------
    Given a dictionary of dataframes: df_dict calculate stimulus balances

    >>> balance_dfs = [get_balance_df(i) for i in df_dict['spactive']]

    We could concatenate and visualize this data, however it would only
    display counts. We are interested in per subject uniformity of (i) the
    count of a number for left/right, and (ii) the counts across numbers

    >>> data = [calc_diff_balance(i) for i in balance_dfs]

    Now concatenate and visualize diffs and mean_deviations

    >>> data = pd.concat(data)
    >>> sns.boxplot(x='number', y='diff', color='black', data=data)
    >>> sns.boxplot(x='number', y='mean_deviations', color='black', data=data)

    """
    diff = (
        balance_df[balance_df["side"] == 0]["count"].to_numpy()
        - balance_df[balance_df["side"] == 1]["count"].to_numpy()
    )

    balance_df["diff"] = np.tile(diff, 2)

    collapsed = balance_df.groupby("number").mean()
    mean_count = balance_df["count"].mean()
    collapsed["mean_deviations"] = collapsed["count"].to_numpy() - mean_count
    collapsed = collapsed.reset_index()
    return collapsed


def find_bad_epochs(epochs, picks=None, thresh=3.29053):
    """Find bad epochs based on amplitude, deviation, and variance.

    Inspired by [1]_, based on code by Marijn van Vliet [2]_. This function is
    working on z-scores. You might want to select the thresholds according to
    how much of the data is expected to fall within the absolute bounds:
    95.0% -> 1.95996, 97.0% -> 2.17009, 99.0% -> 2.57583, 99.9% -> 3.29053.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to all clean EEG channels. Drops EEG
        channels marked as bad.
    thresh : float
        Epochs that surpass the threshold with their z-score based on
        amplitude, deviation, or variance, will be considered bad.

    Returns
    -------
    bads : list of int
        Indices of the bad epochs.

    Notes
    -----
    For this function to work, bad channels should have been identified
    beforehand. Additionally, baseline correction or highpass filtering
    is recommended to reduce signal drifts over time.

    References
    ----------
    .. [1] Nolan, H., Whelan, R., & Reilly, R. B. (2010). FASTER:
       fully automated statistical thresholding for EEG artifact
       rejection. Journal of neuroscience methods, 192(1), 152-162.
    .. [2] https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc

    """

    def _find_outliers(X, threshold=3.0, max_iter=2):
        from scipy.stats import zscore

        my_mask = np.zeros(len(X), dtype=np.bool)
        for _ in range(max_iter):
            X = np.ma.masked_array(X, my_mask)
            this_z = np.abs(zscore(X))
            local_bad = this_z > threshold
            my_mask = np.max([my_mask, local_bad], 0)
            if not np.any(local_bad):
                break

        bad_idx = np.where(my_mask)[0]
        return bad_idx

    def calc_deviation(data):
        ch_mean = np.mean(data, axis=2)
        return ch_mean - np.mean(ch_mean, axis=0)

    metrics = {
        "amplitude": lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        "deviation": lambda x: np.mean(calc_deviation(x), axis=1),
        "variance": lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude="bads")

    data = epochs.get_data()[:, picks, :]

    # find bad epochs
    bads = []
    for m in metrics.keys():
        signal = metrics[m](data)
        bad_idx = _find_outliers(signal, thresh)
        bads.append(bad_idx)

    return np.unique(np.concatenate(bads)).tolist()
