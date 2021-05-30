"""A helper script to run RSA analyses using different parameters.

Works with:

- rsa_analysis_9x9

... by making use of these functions trying to read a config file.

"""
import json
import os
import subprocess

from utils import BIDS_ROOT


def run_with_config(
    which_rsa,
    run_parallel,
    crossvalidate,
    half,
    exclude_models,
    rsa_method,
    distance_metric,
    mnn_tuple,
    equalize_counts,
    preproc_settings,
    outcome_column_name,
):
    """Run rsa_analysis with config.

    Parameters
    ----------
    which_rsa : '9x9'
        Which rsa_analysis script to invoke.
    mnn_tuple : tuple of bool, len 3
        A tuple with three boolean settings: (1) Whether or not to perform
        multivariate noise normalization. (2) and (3) apply only If (1) is
        True: (2) Whether or not to calc a cov matrix for each epo selection
        separately, (3) Whether or not to do mnn for all timepoints at once.
        Defaults to (False, True, True), so no MNN.

    Notes
    -----
    The remaining parameters are explained in the rsa_analysis script.

    """
    # Make a config
    config = {
        "run_parallel": run_parallel,
        "crossvalidate": crossvalidate,
        "half": half,
        "exclude_models": exclude_models,
        "rsa_method": rsa_method,
        "distance_metric": distance_metric,
        "mnn_tuple": mnn_tuple,
        "equalize_counts": equalize_counts,
        "preproc_settings": preproc_settings,
        "outcome_column_name": outcome_column_name,
    }

    print(f"Start {which_rsa} with config: \n{config}")

    # Write the config
    codedir = os.path.join(BIDS_ROOT, "code")
    configpath = os.path.join(codedir, "rsa_analysis_config.json")
    with open(configpath, "w") as fout:
        json.dump(config, fout)

    # stringify preproc_settings
    preproc_str = ""
    for key, val in preproc_settings.items():
        preproc_str += str(key)[0] + "-" + str(val)

    # Run the RSA and write a log
    python_script = os.path.join(codedir, f"rsa_analysis_{which_rsa}.py")
    cmd = f"python {python_script}"
    process = subprocess.run(
        cmd, shell=True, cwd=codedir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    logname = (
        f"{which_rsa}_cv-{crossvalidate}_half-{half.split('_')[0]}"
        f"_exclude-{''.join([i.capitalize() for i in exclude_models])}"
        f"_distance-{distance_metric}_method-{rsa_method}"
        f"_mnn-{''.join([str(i) for i in mnn_tuple])}"
        f"_ec-{equalize_counts}"
        f"_{preproc_str}"
        f"_{outcome_column_name}"
        "_log.out"
    )
    logfile = os.path.join(
        codedir,
        logname,
    )
    with open(logfile, "wb") as fout:
        fout.write(process.stdout)

    # remove the config
    if os.path.exists(configpath):
        os.remove(configpath)
    else:
        print(f"Config file '{configpath}' not found. Cannot remove it.")

    print("\nDone!\n")


if __name__ == "__main__":

    # Standard preprocessing settings
    preproc_settings = {
        "crop": (0.6, 1.6),
        "tshift": -0.8,
        "smooth": 150,
        "baseline": (None, 0),
        "smooth_before_baseline": True,
    }
    # Run the different analyses
    # 9x9
    kwargs = dict(
        which_rsa="9x9",
        run_parallel=False,
        crossvalidate=False,
        exclude_models=["identity"],
        rsa_method="pearson",
        distance_metric="euclidean",
        mnn_tuple=(False, False, False),
        equalize_counts=False,
        preproc_settings=preproc_settings,
        outcome_column_name="outcome",
    )
    run_with_config(**kwargs, half="both")
    run_with_config(**kwargs, half="first_half")
    run_with_config(**kwargs, half="second_half")
