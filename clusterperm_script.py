"""A helper script to pass files to and from clusterperm.py.

Copy RSA results files to the directory where ``clusterperm.py`` is located,
then run ``clusterperm.py``,
then copy the results to another directory.
Repeat for each kind of RSA result to be tested.

"""
import datetime
import os
import shutil
import subprocess

from utils import BIDS_ROOT


def run_clusterperm(rsa_results_path):
    """Run clusterperm.py for a given results.csv file."""
    # Where are we currently?
    wd = os.getcwd()

    # Make a temporary directory with clusterperm.py in it
    time_now = datetime.datetime.now().isoformat()
    tmpdir = os.path.join(BIDS_ROOT, "code", f"clusterperm_results_{time_now}")
    os.makedirs(tmpdir)
    clusterperm_py = os.path.join(BIDS_ROOT, "code", "clusterperm.py")
    shutil.copyfile(clusterperm_py, os.path.join(tmpdir, "clusterperm.py"))

    # Copy the results file to the new tmp clusterperm.py directory
    dest = os.path.join(tmpdir, "rsa_results.csv")
    shutil.copyfile(rsa_results_path, dest)

    # call clusterperm as subprocess
    # capture stdout and stderr in one stream and write that to file
    # after completing the process
    cmd = r"python clusterperm.py"
    process = subprocess.run(
        cmd, shell=True, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    logfile = os.path.join(tmpdir, "log.out")
    with open(logfile, "wb") as fout:
        fout.write(process.stdout)

    # copy over all results to the RSA results dir
    results_dir, _ = os.path.split(rsa_results_path)
    _, dirname = os.path.split(tmpdir)
    shutil.copytree(tmpdir, os.path.join(results_dir, dirname))

    # clean up the tmpdir
    shutil.rmtree(tmpdir)

    # change back to previous working directory
    os.chdir(wd)


if __name__ == "__main__":

    # RSA results to run clusterperm on
    derivatives = os.path.join(BIDS_ROOT, "derivatives")
    rsa_results_paths = [
        os.path.join(
            derivatives,
            "rsa_9x9",
            "cv-False_flip-False_rsa-pearson_dist-euclidean_half-both_exclude-Identity_mnn-FalseFalseFalse_ec-False_c-(0.6, 1.6)t--0.8s-150b-(None, 0)s-True_outcome",
            "rsa_results.csv",
        ),
    ]

    for rsa_results_path in rsa_results_paths:
        print(f"\nstart work on: {rsa_results_path}")
        run_clusterperm(rsa_results_path)
        print("... finished")
