"""A helper script to run RSA analyses using different parameters.

Works with:

- rsa_plotting.py

... by making use of these functions trying to read a config file.

"""
import json
import os
import subprocess

from utils import BIDS_ROOT


def run_with_config(
    rsa_dims,
    rsa_kind,
    modelname,
    orth,
    nth_perm_results=None,
    clusterstat=None,
    do_2x2_with_halfs_ecFalse=False,
):
    """Run rsa_analysis with config.

    Parameters
    ----------
    rsa_dims : '9x9'
        The dimensions of the RSA to look at. this corresponds to a
        top level folder in the /derivatives directory.
    rsa_kind : str
        The name of the RSA results folder nested in the `rsa_dims`
        folder.
    modelname : str
        Which model to look at.
    orth : bool
        Whether to use orthogonalized models or not.
    nth_perm_results : int | None
        If there are multiple perm result distrs, this is the index of
        which distr of the sorted list to take. If None, this is
        ignored and the value specified in rsa_plotting.ipynb is used.
    clusterstat : str
        Which clusterstat to look at, can be "length" or "mass".
    do_2x2_with_halfs_ecFalse : bool
        If True, ignore the "ec" setting (equalized counts) from `rsa_kind`,
        and try to use the first_half, second_half data from a ec-False run
        instead. Can be useful to run an RSA with ec-True to get the significant
        window of interest, but to then perform 2x2 tests in first and
        second half with ec-False on the significant window.

    Notes
    -----
    The parameters are further explained in rsa_plotting.ipynb.

    """
    # Make a config
    config = {
        "rsa_dims": rsa_dims,
        "rsa_kind": rsa_kind,
        "modelname": modelname,
        "orth": orth,
        "do_2x2_with_halfs_ecFalse": do_2x2_with_halfs_ecFalse,
    }

    if nth_perm_results is not None:
        config.update({"nth_perm_results": nth_perm_results})

    if clusterstat is not None:
        config.update({"clusterstat": clusterstat})

    print(f"Start with config: \n{config}")

    # Write the config
    codedir = os.path.join(BIDS_ROOT, "code")
    configpath = os.path.join(codedir, "rsa_plotting_config.json")
    with open(configpath, "w") as fout:
        json.dump(config, fout)

    # Run the script and write a log
    python_script = os.path.join(codedir, "rsa_plotting.py")
    cmd = f"python {python_script}"
    process = subprocess.run(
        cmd, shell=True, cwd=codedir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    logfile = os.path.join(codedir, f"{rsa_dims}_{rsa_kind}_{modelname}_{orth}_log.out")
    with open(logfile, "wb") as fout:
        fout.write(process.stdout)

    # remove the config
    os.remove(configpath)

    print("Done!\n")


if __name__ == "__main__":

    # # Run the different plotting settings
    for orth in (True, False):

        for modelname in ["numberline", "extremity", "sample_frequency"]:
            run_with_config(
                rsa_dims="9x9",
                rsa_kind="cv-False_flip-False_rsa-pearson_dist-euclidean_half-both_exclude-Identity_mnn-FalseFalseFalse_ec-False_c-(0.6, 1.6)t--0.8s-150b-(None, 0)s-True_outcome",
                modelname=modelname,
                orth=orth,
                do_2x2_with_halfs_ecFalse=False,
            )
