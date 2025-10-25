# PhysiKKT-Net



## Getting started



### Setup

Run 'Environment Setup.ipynb' to install all the required dependencies and to download the dataset.

### Install the custom scaler

After you have downloaded the complete data, please replace the source codes of PowerGridScaler() function started at line 118 of MSE_Physics.py with codes in new scaler.py. If you use an IDE, you can hold ctrl and click the PowerGridScaler() to find its source code.

### Run experiments

You can run

python -u MSE_Physics.py

or

nohup python -u MSE_Physics.py > MSE_Physics.log 2>&1 &
