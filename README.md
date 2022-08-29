# Tau Energy Scale Calibration using Neural Networks

The scripts in this repo are desgined to calibrate the tau energy scale (pt spectrum) at the ATLAS detector using Mixture Density Networks (MDN). 

## Running the Code

This code runs out of the box on a lxplus node with a cern computing account. If the user does not have access to these, the dataset will have to be downloaded. 

To run the code, first create a python envirnoment with the provided `requirements.txt` file as follows:

````
cd taunet
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

## Future work

There is still some validation to be performed in order to merge with ATLAS software. Some ideas are listed below:

- Make some plots of the standard deviation given by the network and compare this to the resolution. Notice that this is not trivial to do based on the target we use to train out network. 
- Perhaps train the network with target of `ptTruthVisDressed`. This makes interpretation of the network standard deviation more straightforward. 
- Remove dependance on combined variables and attempt to gain same performance as in the above work. 
- Try networks with more gaussian components until performance begins to fall. 
- Perform a final hyperparameter scan to try and squeeze as much out of the network as possible. 

## Discussion of the work

To learn a bit more about the current state of this work, see the following presentations of the project: 

- An overview of the important results was [presented for the tau working group at ATLAS](https://indico.cern.ch/event/1189825/contributions/5006121/attachments/2493221/4281765/TES_determination_with_NN.pdf). 
- Specifics of the learning used were presented for the [EPE-ML group at the University of Washington](https://indico.cern.ch/event/1112960/contributions/4927575/attachments/2496439/4287590/TES_calibration_with_NNs_EPE_ML_meeting.pdf).
- A general overview of the project was [presented to the University of Washington REU program](https://archive.int.washington.edu/REU/2022/Cochran-Branson_talk.pdf).

Finally, a paper written for the UW REU program giving a full description of the project can be found on the UW REU 2022 page. 
