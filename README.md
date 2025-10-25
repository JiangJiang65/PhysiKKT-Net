# PhysiKKT-Net



## Getting started

### File Structure

To help you better understand this project, here is the main directory and file structure along with their explanations:

PhysiKKT-Net/
├── ml4physim_startingkit_powergrid/              # The folder used in the ML4PSC competition contains folders for storing data and components, etc.
├── Environment Setup.ipynb              # This file is used for installing dependencies and downloading the dataset.
├── evaluation_utils.py              # Evaluation script
├── LICENSE.txt              # Open source license
├── MSE_Physics.py              # The execution script used to test the PhysiKKT-Net method
├── MSE.py              # The execution script for comparing the PhysiKKT-Net method with the MSE method
├── network_mse.py              # The class file providing function for the simple MSE method
├── network_physics.py              # The class file providing functions for the PhysiKKT-Net method
├── new scaler.py              # The class file used to replace the official normalizer
└── README.md             # The project description document

### Setup

Run 'Environment Setup.ipynb' to install all the required dependencies and to download the dataset.

### Install the custom scaler

After you have downloaded the complete data, please replace the source codes of PowerGridScaler() function started at line 118 of MSE_Physics.py with codes in new scaler.py. If you use an IDE, you can hold ctrl and click the PowerGridScaler() to find its source code.

### Run experiments

You can run

python -u MSE_Physics.py

or

nohup python -u MSE_Physics.py > MSE_Physics.log 2>&1 &
