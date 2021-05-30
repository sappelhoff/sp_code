#!/bin/sh

# instead of running rsa_analysis_script, clusterperm_script, and rsa_plotting_script
# one after another, you can call this script with "nohup" once.
# ---------------------------------------------------------------------------------

set -eu # subsequent commands which fail will cause the shell script to exit immediately

python rsa_analysis_script.py
python clusterperm_script.py
python rsa_plotting_script.py
