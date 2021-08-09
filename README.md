# ML_surrogate_model_wildfires
ML surrogate model which combines ROM, LSTM and DA for dynamical systems such as wildfires

This git repo include the python code for constructing a machine learning surrogate model for wildfire (burned area) prediction and real-time adjusting via data assimilation

Requirements:

Python3

numpy
PIL
pickle
tensorflow
scipy
time
matplotlib
tqdm
random
math
fiona
geojson
geopandas


CA:
The code for CA simulations is based on the algorithm in "A cellular automata model for forest fire spread prediction: The case
 of the wildfire that swept through Spetses Island in 1990" (Author: A. Alexandridis a, D. Vakalis b, C.I. Siettos c, G.V. Bafas a)
and the work of https://github.com/XC-Li/Parallel_CellularAutomaton_Wildfire (Author: Xiaochi (George) Li)

We provide the landscape data of the Bear fire in California but data from other fire events can be easily incoperated

Simulation need to be generated for runing ROM methods.

ROM:
The code of both POD- and CAE- based ROM and reconstruction is provided. Different fire events can use the same CAE structure with a different cropping
layer. Once the latent space is created, simulations are required to construct the training dataset of LSTM. One only need to record
the simulation data in the latent space.

LSTM: The training of LSTM based on POD or CAE, including preprocessing, is included. Since CA simulations are stochastic, 
several simulations (120 in this study) are used to construct the surrogate model. Comparison is made in the test dataset (other CA
simulations that haven't been seen by the training of LSTM).

observation:
Data of daily-based satellite observations for the three wildfires and the preprocesssing code.

DA: 
Data assimilation in the latent space using satellite observation data with DI01 covariance tuning and prediction with or without DA. One
can simply disable DA/DI01 to make the comparison of LSTM/DA/LSTM+DA




 
