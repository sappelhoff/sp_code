"""Perform cluster permutation test of RSA data.

The RSA signal is a unidimensional signal of the correlation between an
empirical RDM and a model RDM (either the "numberline" or the "extremity"
model).

This RSA signal (for each model) is analyzed for each condition in a 2x2 mixed
design experiment with the following factors:

- within factor: sampling (active or yoked)
- between factor: stopping (fixed or variable)

For each timepoint of the RSA signal we calculate a mixed anova, checking for
a main effect of sampling or stopping, and an interaction effect.

We use a cluster permutation test to control for the multiple testing problem.
That is, we repeat the mixed anova per timepoint analysis many times, each time
on permuted data, exchanging the factor levels across subjects for the between
factor, and within subjects for the within factor. For each of these iterations
we extract a "maximum" cluster statistic, forming a distribution of those given
the null-hypothesis of our permuted data. Finally, we compare the "observed"
cluster statistics from the non-permuted data to the generated distributions.
We do this once for each effect (sampling, stopping, interaction).

Main functions
--------------
`_permute_df`
-> Permute the dataframe according to our rules.

`calc_p_timecourse`
-> Calculate a mixed model for each timepoint

`_return_clusters_for_df` and `_return_clusters`
-> given a model for each timepoint, extract the timepoints where the p-value
   is below a threshold `thresh` and group them into clusters. Done once for
   each effect.

`return_observed_clusters`
-> use `_return_clusters_for_df` to get the observed clusters.

`generate_cluster_distributions`
-> call `_return_clusters_for_df` repeatedly to generate a distribution of
   "maximum" cluster statistics. For example, for a given effect, get the
   lengths of all clusters ... and take the maximum. Repeat for each iteration
   and the outcomes of this process form the distribution.

`dispatch_distr_generators`
-> call `generate_cluster_distributions` in parallel to speed up computations.
   Each process to run in parallel receives a unique random seed.

`evaluate_significance`
-> given a distribution of maximum cluster statistics and some `clusterthresh`,
   first compute the cutoff beyond which higher cluster statistics can be
   considered significant at the `clusterthresh` level. Then evaluate the
   observed clusters against this threshold and determine which are
   significant.

Workflow
--------
1. obtain cluster statistic distributions via `dispatch_distr_generators`
2. get the observed clusters via `return_observed_clusters`
3. evaluate the significance and cutoff thresholds via `evaluate_significance`

Other Notes
-----------
The mixed anova is performed via the `mixed_anova` function from the Pingouin
package. For the sake of speeding up computations, `mixed_anova`, `rm_anova`,
and `anova` from the Pingouin package have been adapted in this module to
drop all unneeded features and calculatations. You can conveniently change
which function to use by setting the `MIXED_ANOVA_FUNC` variable to either
`pingouin.mixed_anova` or `mixed_anova`.

"""
import itertools
import multiprocessing
import operator
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin  # noqa: F401
import seaborn as sns
from scipy.stats import f


def _permute_df(df, rng):
    """Permute a 2x2 mixed design data frame.

    The `df` is changed in place.

    Parameters
    ----------
    df : pandas.DataFrame
        The data containing at least the following columns:
        ['subject', 'sampling', 'stopping', 'itime', 'similarity']. The
        'stopping' column is the between subjects factor, the 'sampling'
        column is the within subjects factor.
    rng : numpy.random.mtrand.RandomState
        The random number generator to use for the permutations: Some
        np.random.RandomState(seed) object initialized at a seed.

    Returns
    -------
    df : pd.DataFrame
        Copy of the data with the columns: 'stopping' and 'sampling' replaced
        by permutations of how they were originally (according to rules, see
        Notes section).

    Notes
    -----
    We shuffle the between subject factor levels across subjects and the
    within subject factor levels within each subject.

    """
    # Permute the within subjects factor -> exchange labels within each subj
    # ----------------------------------------------------------------------
    # for each subject, randomly (p=0.5) decide: keep levels as is, or switch
    subjs = np.unique(df["subject"])
    switch = rng.binomial(1, 0.5, len(subjs)).astype(bool)

    # find subjects where we switch factor levels
    switch_rows = df["subject"].isin(subjs[switch])

    # apply the map
    switchmap = {"active": "yoked", "yoked": "active"}
    permvals = df.loc[switch_rows, "sampling"].map(switchmap)
    df.loc[switch_rows, "sampling"] = permvals

    # Permute the between subjects factor -> exchange labels across subjs
    # -------------------------------------------------------------------
    # focus on subjects and the between factor
    between_df = df[["subject", "stopping"]].drop_duplicates()

    # permute the between factor across subjects
    between_df.loc[:, "stopping"] = rng.permutation(between_df["stopping"])

    # merge back onto data frame
    df = df.merge(between_df, on=["subject"], suffixes=("_old", ""), copy=False)
    df.drop("stopping_old", axis=1, inplace=True)

    # Done, return
    # ------------
    return df


def return_observed_clusters(df, thresh, mixed_anova_func):
    """Get the observed clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        The data containing at least the following columns:
        ['subject', 'sampling', 'stopping', 'itime', 'similarity']. The
        'stopping' column is the between subjects factor, the 'sampling'
        column is the within subjects factor.
    thresh : float
        Statistical cutoff when to consider a p value significant.
    mixed_anova_func : callable
        Function handle to the mixed_anova to use. Can be pingouin.mixed_anova
        or the adapted mixed_anova function from this module to achieve a
        moderate speedup.

    Returns
    -------
    clusters_dict : dict of clusters
        Each key in the dict is an effect and its values is a list of clusters
        (=list of indices).
    models : list of pandas.DataFrame
        List of mixed models, where each model is a mixed model at a timepoint.

    """
    clusters_dict, models = _return_clusters_for_df(
        df=df,
        between="stopping",
        within="sampling",
        thresh=thresh,
        mixed_anova_func=mixed_anova_func,
    )

    return clusters_dict, models


def _return_clusters_for_df(df, between, within, thresh, mixed_anova_func):
    """Return clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        The data containing at least the following columns:
        ['subject', 'sampling', 'stopping', 'itime', 'similarity']. The
        'stopping' column is the between subjects factor, the 'sampling'
        column is the within subjects factor.
    between : str
        Column from `df` to consider as between subjects factor.
    within : str
        Column from `df` to consider as within subjects factor.
    thresh : float
        Statistical cutoff when to consider a p value significant.
    mixed_anova_func : callable
        Function handle to the mixed_anova to use. Can be pingouin.mixed_anova
        or the adapted mixed_anova function from this module to achieve a
        moderate speedup.

    Returns
    -------
    clusters_dict : dict of clusters
        Each key in the dict is an effect and its values is a list of clusters
        (=list of indices).
    models : list of pandas.DataFrame
        List of mixed models, where each model is a mixed model at a timepoint.

    """
    # Get a model per timepoint
    models = calc_p_timecourse(
        df=df, between=between, within=within, mixed_anova_func=mixed_anova_func
    )

    # p-vals per timepoint for each effect
    parr = np.array([model["p-unc"].to_numpy() for model in models])

    # Get clusters for each effect
    effects = [i.lower() for i in models[0]["Source"].to_list()]
    clusters_dict = {}
    for i, effect in enumerate(effects):
        clusters_dict[effect] = _return_clusters(parr[:, i] < thresh)

    return clusters_dict, models


def generate_cluster_distributions(
    df, rng, n_iterations, thresh, clusterstat, mixed_anova_func
):
    """Perform a cluster permutation test on condition data.

    Parameters
    ----------
    df : pandas.DataFrame
        The data containing at least the following columns:
        ['subject', 'sampling', 'stopping', 'itime', 'similarity']. The
        'stopping' column is the between subjects factor, the 'sampling'
        column is the within subjects factor.
    rng : numpy.random.mtrand.RandomState
        The random number generator to use for the permutations: Some
        np.random.RandomState(seed) object initialized at a seed.
    n_iterations : int
        Number of iterations to run for generating the distribution.
    thresh : float
        Statistical cutoff when to consider a p value significant.
    clusterstat : str
        Which cluster statistic was used for `cluster_distr`, one of
        ['length', 'mass'].
    mixed_anova_func : callable
        Function handle to the mixed_anova to use. Can be pingouin.mixed_anova
        or the adapted mixed_anova function from this module to achieve a
        moderate speedup.

    Returns
    -------
    cluster_distr : dict
        Dictionary where each key is an effect and its values are cluster
        stats from each iteration in form of a list.

    """
    cluster_distr = {}
    for iter in range(n_iterations):
        # Permute the data
        df = _permute_df(df, rng)

        # Get clusters
        clusters_dict, models = _return_clusters_for_df(
            df,
            between="stopping",
            within="sampling",
            thresh=thresh,
            mixed_anova_func=mixed_anova_func,
        )

        # Get max cluster stat for each effect
        if clusterstat == "length":
            update_dict = _return_max_cluster_len(clusters_dict)
        elif clusterstat == "mass":
            update_dict = _return_max_cluster_mass(clusters_dict, models)
        else:
            raise ValueError('unknown `clusterstat` argument "{}"'.format(clusterstat))

        cluster_distr = _update_cluster_distributions(cluster_distr, update_dict)

    return cluster_distr


def _update_cluster_distributions(cluster_distrs, update_dict):
    """Update a dict.

    `cluster_distr` can also be an empty dict which will then be updated.

    Parameters
    ----------
    cluster_distr : dict
        Dictionary where each key is an effect and its values are cluster
        stats from each iteration in form of a list.
    update_dict : dict
        Each key in the dict is an effect and its values are ints reflecting
        the maximum cluster length.

    Returns
    -------
    cluster_distr : dict
        Dictionary where each key is an effect and its values are cluster
        stats from each iteration in form of a list.

    """
    for effect in update_dict:
        cluster_distrs[effect] = cluster_distrs.get(effect, [])
        cluster_distrs[effect].append(update_dict[effect])

    return cluster_distrs


def calc_p_timecourse(df, between, within, mixed_anova_func):
    """Calculate a mixed model for each timepoint.

    Parameters
    ----------
    df : pandas.DataFrame
        The data containing at least the following columns:
        ['subject', 'sampling', 'stopping', 'itime', 'similarity']. The
        'stopping' column is the between subjects factor, the 'sampling'
        column is the within subjects factor.
    between : str
        Column from `df` to consider as between subjects factor.
    within : str
        Column from `df` to consider as within subjects factor.
    mixed_anova_func : callable
        Function handle to the mixed_anova to use. Can be pingouin.mixed_anova
        or the adapted mixed_anova function from this module to achieve a
        moderate speedup.

    Returns
    -------
    models : list of pandas.DataFrame
        List of mixed models, where each model is a mixed model at a timepoint.

    """
    itimes = np.unique(df["itime"])

    # collapse to the mean for mixed anova
    data = df.groupby(["subject", within, between, "itime"]).mean().reset_index()

    models = []
    for itime in itimes:
        df_itime = data[data["itime"] == itime]

        model = mixed_anova_func(
            data=df_itime,
            dv="similarity",
            within=within,
            subject="subject",
            between=between,
        )
        models.append(model)

    return models


def _return_clusters(arr):
    """Return a list of clusters (=list of indices).

    Parameters
    ----------
    arr : ndarray
        a one dimensional array of zeros and ones, each index reflecting a
        timepoint. When the value is one, the timepoint is significant.

    Returns
    -------
    clusters : list of lists
        Each entry in the list is a list of indices into time. Each of these
        indices marks a significant timepoint.

    """
    # See: https://stackoverflow.com/a/31544396/5201771
    return [
        [i for i, value in it]
        for key, it in itertools.groupby(enumerate(arr), key=operator.itemgetter(1))
        if key != 0
    ]


def _return_max_cluster_len(clusters_dict):
    """Return maximum cluster length per effect.

    Parameters
    ----------
    clusters_dict : dict of clusters
        Each key in the dict is an effect and its values is a list of clusters
        (=list of indices).

    Returns
    -------
    max_cluster_len : dict
        Each key in the dict is an effect and its values are ints reflecting
        the maximum cluster length.

    See also
    --------
    _return_max_cluster_mass

    """
    max_cluster_len = {}
    for effect, clust in clusters_dict.items():
        cluster_lens = [len(cl) for cl in clust]
        if len(cluster_lens) == 0:
            max_cluster_len[effect] = 0
        else:
            max_cluster_len[effect] = max(cluster_lens)
    return max_cluster_len


def _calc_cluster_masses(clusters_dict, effect, models):
    """Help _return_max_cluster_mass."""
    clustermasses = list()
    for cluster in clusters_dict[effect]:
        cm = pd.concat([models[i] for i in cluster])
        fvals = cm[cm["Source"].str.lower() == effect]["F"].to_numpy()
        clustermasses.append(fvals.sum())

    return clustermasses


def _return_max_cluster_mass(clusters_dict, models):
    """Return maximum cluster mass per effect.

    Parameters
    ----------
    clusters_dict : dict of clusters
        Each key in the dict is an effect and its values is a list of clusters
        (=list of indices).
    models : list of pandas.DataFrame
        List of mixed models, where each model is a mixed model at a timepoint.

    Returns
    -------
    max_cluster_mass : dict
        Each key in the dict is an effect and its values are ints reflecting
        the maximum cluster mass.

    See also
    --------
    _return_max_cluster_len

    """
    # Go through each effect separately
    max_cluster_mass = {}
    for effect in clusters_dict:
        # Calculate clustermass for each cluster in this effect
        clustermasses = _calc_cluster_masses(clusters_dict, effect, models)

        if len(clustermasses) == 0:
            max_cluster_mass[effect] = 0
        else:
            max_cluster_mass[effect] = max(clustermasses)

    return max_cluster_mass


def _combine_cluster_distrs(cluster_distrs_list):
    """Combine a list of dicts into a single dict.

    Parameters
    ----------
    cluster_distr_list : List
        List of cluster_distrs.

    Returns
    -------
    cluster_distr : dict
        Dictionary where each key is an effect and its values are cluster
        stats from each iteration in form of a list.

    """
    cluster_distr = {}
    for cluster_distr_item in cluster_distrs_list:
        for effect in cluster_distr_item:
            cluster_distr[effect] = cluster_distr.get(effect, [])
            cluster_distr[effect] += cluster_distr_item[effect]

    return cluster_distr


def dispatch_distr_generators(
    df, seeds, n_iterations, thresh, clusterstat, mixed_anova_func
):
    """Compute cluster distrs in parallel.

    Parameters
    ----------
    df : pandas.DataFrame
        The data containing at least the following columns:
        ['subject', 'sampling', 'stopping', 'itime', 'similarity']. The
        'stopping' column is the between subjects factor, the 'sampling'
        column is the within subjects factor.
    seeds : list of int
        The number of seeds will determine the number of processes to run
        in parallel. Each seed will be passed to one process and determines
        the random number generator of that process.
    n_iterations : int
        The number of iterations to run per process.
    thresh : float
        Statistical cutoff when to consider a p value significant.
    clusterstat : str
        Which cluster statistic was used for `cluster_distr`, one of
        ['length', 'mass'].
    mixed_anova_func : callable
        Function handle to the mixed_anova to use. Can be pingouin.mixed_anova
        or the adapted mixed_anova function from this module to achieve a
        moderate speedup.

    Returns
    -------
    cluster_distr : dict
        Dictionary where each key is an effect and its values are cluster
        stats from each iteration in form of a list.

    """
    # Turn seeds into RNGs
    seeds = [np.random.RandomState(i) for i in seeds]

    # generate inputs
    inputs = list(
        itertools.product(
            [df], seeds, [n_iterations], [thresh], [clusterstat], [mixed_anova_func]
        )
    )

    # run in paralell
    with multiprocessing.Pool(len(seeds)) as pool:
        results = pool.starmap(generate_cluster_distributions, inputs)

    # combine results
    cluster_distr = _combine_cluster_distrs(results)
    return cluster_distr


def evaluate_significance(
    cluster_distr, clusters_obs, clusterstat, clusterthresh, models_obs=None
):
    """Evaluate significane of observed clusters.

    Parameters
    ----------
    cluster_distr : dict
        Dictionary where each key is an effect and its values are cluster
        stats from each iteration in form of a list.
    clusters_obs : dict of clusters
        Each key in the dict is an effect and its values is a list of clusters
        (=list of indices).
    clusterstat : str
        Which cluster statistic was used for `cluster_distr`, one of
        ['length', 'mass']. This statistic will be applied to the clusters in
        `clusters_obs`.
    clusterthresh : float
        Statistical cutoff when to consider a cluster statistic significant.
    models_obs : list of pandas.DataFrame | None
        List of mixed models, where each model is a mixed model at a timepoint.
        Only needed if `clusterstat` is "mass". Defaults to None.

    Returns
    -------
    clustersig_threshs : dict
        The cutoff thresholds per effect beyond which cluster statistics are
        significant.
    clusters_obs_stats : dict
        The statistics of each observed cluster per effect.
    clusters_obs_sig : dict
        The clusters per effect that are significant.

    """
    effects = ["stopping", "sampling", "interaction"]
    clusters_obs_stats = {}
    clusters_obs_sig = {}
    clustersig_threshs = {}
    for effect in effects:
        # Get array of sampled statistics
        arr = np.array(cluster_distr[effect])

        # Calculate significance "threshold" in terms of sampled statistics
        clusterthresh_idx = int(np.ceil(len(arr) * clusterthresh))
        clusterthresh_stat = np.sort(arr)[::-1][clusterthresh_idx]
        clustersig_threshs[effect] = clusterthresh_stat

        # Find significant observed clusters based on the used clusterstat
        is_significant = []
        if clusterstat == "length":
            clusterlengths = [len(cluster) for cluster in clusters_obs[effect]]
            clusters_obs_stats[effect] = clusterlengths
            for stat in clusterlengths:
                pval = (1 + np.sum(arr >= stat)) / (1 + len(arr))
                is_significant.append(pval < clusterthresh)

        elif clusterstat == "mass":
            assert models_obs is not None, "Need to pass `models_obs`."
            clustermasses = _calc_cluster_masses(clusters_obs, effect, models_obs)
            clusters_obs_stats[effect] = clustermasses
            for stat in clustermasses:
                pval = (1 + np.sum(arr >= stat)) / (1 + len(arr))
                is_significant.append(pval < clusterthresh)

        else:
            raise ValueError('unknown `clusterstat` argument "{}"'.format(clusterstat))

        sig_clusters = list(np.array(clusters_obs[effect])[is_significant])
        clusters_obs_sig[effect] = sig_clusters

    return clustersig_threshs, clusters_obs_stats, clusters_obs_sig


def rm_anova(data=None, dv=None, within=None, subject=None):
    """Repeated measures ANOVA adapted from Pingouin package."""
    # Collapse to the mean
    data = data.groupby([subject, within]).mean().reset_index()

    # Groupby
    grp_with = data.groupby(within)[dv]
    rm = list(data[within].unique())
    n_rm = len(rm)
    grp_with_count = grp_with.count()
    n_obs = int(grp_with_count.max())
    grandmean = data[dv].mean()

    # Calculate sums of squares
    ss_with = ((grp_with.mean() - grandmean) ** 2 * grp_with_count).sum()
    ss_resall = np.sum((data[dv] - grp_with.transform(np.mean)) ** 2)
    # sstotal = sstime + ss_resall =  sstime + (sssubj + sserror)
    # ss_total = ((data[dv] - grandmean)**2).sum()
    # We can further divide the residuals into a within and between component:
    grp_subj = data.groupby(subject)[dv]
    ss_resbetw = n_rm * np.sum((grp_subj.mean() - grandmean) ** 2)
    ss_reswith = ss_resall - ss_resbetw

    # Calculate degrees of freedom
    ddof1 = n_rm - 1
    ddof2 = ddof1 * (n_obs - 1)

    # Calculate MS, F and p-values
    ms_with = ss_with / ddof1
    ms_reswith = ss_reswith / ddof2

    aov = {
        "SS": [ss_with, ss_reswith],
        "DF": [ddof1, ddof2],
        "MS": [ms_with, ms_reswith],
    }

    return aov


def anova(data=None, dv=None, between=None):
    """ANOVA adapted from Pingouin package."""
    groups = list(data[between].unique())
    n_groups = len(groups)
    N = data[dv].size

    # Calculate sums of squares
    grp = data.groupby(between)[dv]
    # Between effect
    ssbetween = ((grp.mean() - data[dv].mean()) ** 2 * grp.count()).sum()
    # Within effect (= error between)
    #  = (grp.var(ddof=0) * grp.count()).sum()
    sserror = np.sum((data[dv] - grp.transform(np.mean)) ** 2)
    # In 1-way ANOVA, sstotal = ssbetween + sserror
    # sstotal = ssbetween + sserror

    # Calculate DOF, MS, F and p-values
    ddof1 = n_groups - 1
    ddof2 = N - n_groups
    msbetween = ssbetween / ddof1
    mserror = sserror / ddof2

    aov = {"SS": [ssbetween, sserror], "DF": [ddof1, ddof2], "MS": [msbetween, mserror]}

    return aov


def mixed_anova(data=None, dv=None, within=None, subject=None, between=None):
    """Mixed-design ANOVA adapted from Pingouin package."""
    # SUMS OF SQUARES
    grandmean = data[dv].mean()
    ss_total = ((data[dv] - grandmean) ** 2).sum()
    # Extract main effects of within and between factors
    aov_with = rm_anova(dv=dv, within=within, subject=subject, data=data)
    aov_betw = anova(dv=dv, between=between, data=data)
    ss_betw = aov_betw["SS"][0]
    ss_with = aov_with["SS"][0]
    # Extract residuals and interactions
    grp = data.groupby([between, within])[dv]
    # ssresall = residuals within + residuals between
    ss_resall = np.sum((data[dv] - grp.transform(np.mean)) ** 2)

    # Interaction
    ss_inter = ss_total - (ss_resall + ss_with + ss_betw)
    ss_reswith = aov_with["SS"][1] - ss_inter
    ss_resbetw = ss_total - (ss_with + ss_betw + ss_reswith + ss_inter)

    # DEGREES OF FREEDOM
    n_obs = data.shape[0] / data[within].nunique()
    df_with = aov_with["DF"][0]
    df_betw = aov_betw["DF"][0]
    df_resbetw = n_obs - data[between].nunique()
    df_reswith = df_with * df_resbetw
    df_inter = aov_with["DF"][0] * aov_betw["DF"][0]

    # MEAN SQUARES
    ms_betw = aov_betw["MS"][0]
    ms_with = aov_with["MS"][0]
    ms_resbetw = ss_resbetw / df_resbetw
    ms_reswith = ss_reswith / df_reswith
    ms_inter = ss_inter / df_inter

    # F VALUES
    f_betw = ms_betw / ms_resbetw
    f_with = ms_with / ms_reswith
    f_inter = ms_inter / ms_reswith

    # P-values
    p_betw = f(df_betw, df_resbetw).sf(f_betw)
    p_with = f(df_with, df_reswith).sf(f_with)
    p_inter = f(df_inter, df_reswith).sf(f_inter)

    aov = pd.DataFrame(
        {
            "Source": [between, within, "interaction"],
            "F": [f_betw, f_with, f_inter],
            "p-unc": [p_betw, p_with, p_inter],
        }
    )

    return aov


def plot_results(
    cluster_distr,
    clustersig_threshs,
    clusters_obs_stats,
    model_name,
    clusterstat,
    fname=None,
):
    """Plot results."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, effect in enumerate(cluster_distr):
        ax = axs.flat[i]
        sns.distplot(cluster_distr[effect], ax=ax)

        ax.set_title(effect)
        ax.set_xlabel("cluster stat")

        # significance cutoff
        ax.axvline(
            clustersig_threshs[effect], color="b", label="p={}".format(clusterthresh)
        )

        # observed clusters
        for j, obs_stat in enumerate(clusters_obs_stats[effect]):
            label = None
            if j == 0:
                label = "observed clusters"
            ax.axvline(obs_stat, color="r", linestyle="--", label=label)

        if i == 0:
            ax.legend()

    fig.suptitle(
        "{}-{} ... {} iterations".format(
            model_name, clusterstat, len(cluster_distr["stopping"])
        )
    )

    if fname:
        plt.savefig(fname)

    return fig, axs


if __name__ == "__main__":

    # Perform cluster permutation test for each model, once using "length" as
    # cluster stat, and once "mass". Save all results.

    MIXED_ANOVA_FUNC = mixed_anova
    NJOBS = min(40, multiprocessing.cpu_count() - 6)  # how many cores to run on
    thresh = 0.05
    clusterthresh = 0.05
    n_iterations = 250  # number of iterations per job (see NJOBS)

    # Read RSA data
    fname = "rsa_results.csv"
    condi_corr_df = pd.read_csv(fname)

    groupby = ["orth", "model"] if "orth" in condi_corr_df else ["model"]
    for meta, group in condi_corr_df.groupby(groupby):
        orthstr = "orth" if meta[0] else ""
        model_name = orthstr + meta[1]
        tmp = group[["subject", "sampling", "stopping", "itime", "similarity"]]

        for clusterstat in ["length", "mass"]:
            print(f"\n\nwork on {model_name}-{clusterstat}")
            start = time.time()

            # work on copy of original DF
            df = tmp.copy()

            # Generate cluster distributions
            seeds = np.arange(NJOBS)
            cluster_distr = dispatch_distr_generators(
                df, seeds, n_iterations, thresh, clusterstat, MIXED_ANOVA_FUNC
            )

            # Get observed clusters
            (clusters_obs, models_obs) = return_observed_clusters(
                tmp, thresh, MIXED_ANOVA_FUNC
            )

            # Evaluate significance
            (
                clustersig_threshs,
                clusters_obs_stats,
                clusters_obs_sig,
            ) = evaluate_significance(
                cluster_distr, clusters_obs, clusterstat, clusterthresh, models_obs
            )

            # save data and a plot of results
            fname_base = f"model-{model_name}_stat-{clusterstat}_thresh-{thresh}"

            fname_plot = f"{fname_base}_plot.png"
            plot_results(
                cluster_distr,
                clustersig_threshs,
                clusters_obs_stats,
                model_name,
                clusterstat,
                fname_plot,
            )

            fname_distr = f"{fname_base}_distr.npy"
            cluster_distr_arr = pd.DataFrame(cluster_distr).to_numpy()
            np.save(fname_distr, cluster_distr_arr)

            # also save the column order of the cluster distribution
            fname_distr_col_ord = f"{fname_base}_distr_column_order.txt"
            with open(fname_distr_col_ord, 'w') as fout:
                print('\n'.join(cluster_distr.keys()), file=fout)

            stop = time.time()
            print(f"time elapsed: {stop - start:.2f} seconds")
