[![DOI](https://zenodo.org/badge/369565611.svg)](https://zenodo.org/badge/latestdoi/369565611)

# Code for the `mpib_sp_eeg` dataset

The code in this repository was used to analyze the data from the `mpib_sp_eeg` dataset
(see "Download the data" below).

Below, we describe how to setup the environment necessary to reproduce the results.

Finally, there is a section describing each file contained in this repository.

## Further information

### Preprint

A preprint is available on BioRxiv.

- BioRxiv: https://doi.org/10.1101/2021.06.03.446960

### Experiment code

The code that was run to collect data from the human study participants is
available on GitHub, and is archived on Zenodo.

- GitHub: https://github.com/sappelhoff/sp_experiment/
- Zenodo: https://doi.org/10.5281/zenodo.3354368

### Data

The data is available on GIN (see "Download the data") below.

- GIN: https://gin.g-node.org/sappelhoff/mpib_sp_eeg

### Analysis code

The analysis code in this repository is also archived on Zenodo.

- Zenodo: https://doi.org/10.5281/zenodo.5929222

## Setup

### Install a Python environment

The code was written and tested on Linux (Ubuntu 18.04) using Python 3.8 installed via the
conda package manager (version 4.10.1).

It is RECOMMENDED to run this code on a machine with sufficient RAM, and preferably around 50 cores.
Otherwise, the analyses may take a long time to run.

To prepare a similar environment, please follow these steps:

1. Make sure you have an up to date version of `conda` available from your command line
   (we recommend downloading miniconda: https://docs.conda.io/en/latest/miniconda.html)
2. Create a new conda environment using the `environment.yml` file that you find in this
   repository. For that, from the command line, run: `conda env create -f ./environment.yml`,
   where the `./` in `./environment.yml` should be replaced with the path to the `environment.yml`
   file from this repository.
3. That will create a new conda environment called `sp`, which you can activate using
   `conda activate sp` (assuming that you did not have an environment with that name before).

All documentation below assumes that you have setup and activated this Python environment.

### Download the data

The data is available here: https://gin.g-node.org/sappelhoff/mpib_sp_eeg/

Please follow the download instructions listed there. Note that if you followed the steps
outlined above, you will already have a working version of datalad that is necessary to
download the data.

### Copy the code

Place all code from this repository in the `/code` directory within the `mpib_sp_eeg`
dataset (incuding `.gitignore`, but except for `environment.yml`, which is already there).

Then, navigate to `mpib_sp_eeg` dataset and run:

1. `datalad unlock .` (this might take a while)
2. `cp -r ./derivatives/annotation_derivatives/sub* ./derivatives`

in order to create one derivatives folder per subject, with the annotation derivatives already
stored within that folder (see `README` in `/annotation_derivatives` for more information).

### Configure your path

To run the analysis code on your system, you need to configure the path where the `mpib_sp_eeg`
dataset is stored. For that, go to `code/utils.py` and find and extend the lines that are
shown in the example below:

```Python
# Find these lines in utils.py:
# Adjust this path to where the bids directory is stored
home = os.path.expanduser("~")
if "stefanappelhoff" in home:
    BIDS_ROOT = os.path.join("/", "home", "stefanappelhoff", "Desktop", "sp_data")
elif "appelhoff" in home:
    BIDS_ROOT = os.path.join("/", "home", "appelhoff", "appelhoff", "sp_data")
elif "example" in home:
    BIDS_ROOT = os.path.join("/", "home", "example", "mpib_sp_eeg")
```

In the code block above, we added "example" with their hypothetical path to the data.
Please adjust for your own needs.

### Convert `.ipynb` notebooks to `.py` scripts using `.tpl` template

Running the code is sometimes more convenient in notebook format, and at other times in
script format. Using `nbconvert`, we can do both. To convert the notebooks to script
format, run: `bash ipynb2py.sh`

That makes use of the `simplepython.tpl` conversion template, which makes sure that no
possibly breaking "magic" syntax is included in the Python scripts.

### Other configurations

The following configurations are optional:

- For using Jupyter Notebook extensions, run: `jupyter contrib nbextension install --user`
- For getting reasonable git diffs on notebooks, run: `nbdime config-git --enable --global`

### Produce initial derivative files

Run the following commands in succession:

1. `python 02_load_and_concatenate.py`
2. `python 03_run_ica.py`
3. `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace analysis_behavior.ipynb`
4. `jupyter nbconvert --to html --ExecutePreprocessor.timeout=None --execute analysis_behavior.ipynb`
5. `python 04_epoching.py`

or all at once as a background process using:

```shell
nohup sh -c "python 02_load_and_concatenate.py && \
python 03_run_ica.py && \
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace analysis_behavior.ipynb && \
jupyter nbconvert --to html --ExecutePreprocessor.timeout=None --execute analysis_behavior.ipynb && \
python 04_epoching.py" &
```

and in that case you can check the `nohup.out` file that is produced by that command for any logs.

### Produce all remaining results

Run the following commands in succession, and inspect the resulting `analysis_behavior.html`
(from before) and `analysis_erp.html` files for the results:

- `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace analysis_erp.ipynb`
- `jupyter nbconvert --to html --ExecutePreprocessor.timeout=None --execute analysis_erp.ipynb`

Then run the following command:

- `nohup bash run_rsa_cluster_plot.sh &`

The results of that command are saved in the `derivatives/rsa_9x9` directory.
They are admittedly a bit scattered, so there is a short guideline to find the statistics that are reported
in the paper at the end of this text section.

Then, run:

- `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace analysis_neurometrics.ipynb`
- `jupyter nbconvert --to html --ExecutePreprocessor.timeout=None --execute analysis_neurometrics.ipynb`

and inspect `analysis_neurometrics.html` for the results.

To produce the figures, run:

- `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace publication_plots.ipynb`
- `jupyter nbconvert --to html --ExecutePreprocessor.timeout=None --execute publication_plots.ipynb`

and inspect the `publication_plots/` directory for the plots, and `publication_plots.html` for
some statistical results on the neurometrics analyses.

Note: To produce a complete figure 1 instead of separate files `fig1a.pdf` and `fig1bcd.pdf`, you need
to run `publication_plots/fig1.sh`; but make sure that all required software is installed before
running the script (open the script in a text editor to see the requirements).

To find the results for the RSA (after running `run_rsa_cluster_plot.sh`), use these hints:

- There are three folders in `derivatives/rsa_9x9`. Open the one that contains `half-both` in its name.
- Within that folder, there is a folder starting with `clusterperm_results_` and ending with a timestamp.
  We will refer to that folder as the `clusterperm_results_*` folder below.
- The statistics reported in the paper on RSA can be found as follows:
- start and stop of the overall (collapsed over conditions) numberline cluster, as well as its p-value:
  `perm_and_2x2_outputs/orthnumberline_length/average_onumberline_clusters.json`
- p-values of numberline clusters by conditions:
  `clusterperm_results_*/model_orthnumberline_stat-length_thresh-0.05_pvals.txt`,
  and the associated
  `clusterperm_results_*/model_orthnumberline_stat-length_thresh-0.05_plot.png`
- time window of interaction cluster: `publication_plots.html` (in the outputs)
- t-tests on the significant interaction cluster:
  `perm_and_2x2_outputs/orthnumberline_length/posthocs_interaction-0-both.html`, and
  `perm_and_2x2_outputs/orthnumberline_length/posthocs_interaction-0-first_half.html`, and
  `perm_and_2x2_outputs/orthnumberline_length/posthocs_interaction-0-second_half.html`
- start and stop of the overall (collapsed over conditions) extremity cluster, as well as its p-value:
  `perm_and_2x2_outputs/orthextremity_length/average_oextremity_clusters.json`
- p-values of extremity clusters by conditions:
  `clusterperm_results_*/model_orthextremity_stat-length_thresh-0.05_pvals.txt`,
  and the associated
  `clusterperm_results_*/model-orthextremity_stat-length_thresh-0.05_plot.png`

## Explanations for the different files

### "other" files

- `.gitignore`: So that temporary files, caches, or log files are not committed to the version control history.
- `environment.yml`: For use with the `conda` package manager to set up a Python environment for running the code.
- `simplepython.tpl`: A template file for `nbconvert` that is used to convert Jupyter Notebooks to Python scripts.
- `ipynb2py.sh`: A convenience shell script (`bash`) that converts all notebooks of interest to Python scripts.

### Utility functions and configurations

- `utils.py`: Contains miscellaneous functions and configurations that are imported throughout the other scripts/notebooks

### Preprocessing code

- `01_raw_inspection.ipynb`: Used for visually screening each raw file, and marking bad temporal segments and bad channels.
  This script was used to produce the `derivatives/annotation_derivatives` in the `mpib_sp_eeg` dataset (`*_annotations.txt`
  and `*_badchannels.txt` files per subject).

- `02_load_and_concatenate.ipynb`: Used to (1) load all three raw EEG files per subject in the original BrainVision format,
  (2) modify the event triggers to remain identifiable after concatenation of all files, (3) concatenate the files and
  save in MNE-Python native `.fif` format: One raw `.fif` per subject. This needs the outputs from `01_raw_inspection.ipynb`.

- `03_run_ica.ipynb`: Used to run ICA on the data, and perform filtering, interpolation, and re-referencing to obtain clean data.
  This script can be run in "interactive" mode (from the notebook) to screen ICA components and potentially mark them for
  rejection. This produces the `*__concat_eeg-excluded-ica-comps.txt` in `derivatives/annotation_derivatives` in
  the `mpib_sp_eeg` dataset. If those derivatives are already present, the script can also be run non-interactively to just
  preprocess the data based on previous screening.

- `04_epoching.py`: Used to epoch the data and tag each epoch along a number of conditions that it belongs to.

### Analysis code

- `analysis_behavior.ipynb`: Analysis of behavioral data.
- `analysis_erp.ipynb`: Analysis of univariate EEG/ERP data.
- `rsa_analysis_9x9.ipynb`: Run RSA and plot results based on different parameters.
- `rsa_analysis_script.py`: Allows running `rsa_analysis_9x9.py` (converted from `rsa_analysis_9x9.ipynb`) with several parameter settings
- `analysis_neurometrics.ipynb`: Analysis of multivariate EEG data (RSA): neurometrics.
- `clusterperm.py`: Run cluster based permutation analyses on RSA analysis results (produced by `rsa_analysis_9x9.ipynb`)
- `clusterperm_script.py`: Allows running `clusterperm.py` for several RSA analysis results one after another (good to run over night).
- `rsa_plotting.ipynb`: Prepare plots and statistics from RSA analysis results and their cluster based permutation analysis results.
- `rsa_plotting_script.py`: Allows running `rsa_plotting.py` (converted from `rsa_plotting.ipynb`) for several RSA analysis results one after another.
- `run_rsa_cluster_plot.sh`: Convenient bash script --> simply set up `rsa_analysis_script.py`, `clusterperm_script.py`, and `rsa_plotting_script.py`
  and then run `nohup bash run_rsa_cluster_plot.sh &` over night to let it produce all results.

### Plotting

- `publication_plots/README`: Basic information about the `publication_plots/` directory.
- `publication_plots.ipynb`: To produce the plots used in the paper.
- `publication_plots/fig1a.odg`: A "LibreOffice Draw" file for panel a in Figure 1.
- `publication_plots/fig1a.pdf`: A pdf export from the `publication_plots/fig1a.odg` file.
- `publication_plots/fig1-tikz.tex`: A XeTex file to stitch together panel a, and the remaining panels for Figure 1.
- `publication_plots/fig1-tikz_to_png.sh`: A shell script to convert Figure 1 PDF to PNG.

