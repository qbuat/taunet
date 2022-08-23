# Tau Energy Scale Calibration using Neural Networks

The scripts in this repo are desgined to calibrate the tau energy scale (pt spectrum) at the ATLAS detector using Mixture Density Networks (MDN). 

## Running the Code

This code runs out of the box on a lxplus node with a cern computing account. If the user does not have access to these, the dataset will have to be downloaded. 

To run the code, first create a python envirnoment with the provided `requirements.txt` file as follows:

````
python3 -m venv tesenv
source tesenv/bin/activate
pip install -r requirements.txt
````

Note: the version of `pip` on the lxplus nodes is old and may not be able to find the desired version of `tensorflow`. Before installing requirements from the `txt` file consider running

````
pip install --upgrade pip
````

Once you have the environment set up you can run the file `fit.py` to learn a good network. Running `plot.py` will then create plots from this network. One good network is already in the folder `final_MDN`. To get plots for this using data from part of the dataset run 

````
python plot.py --path final_MDN --debug --nfiles 1
````