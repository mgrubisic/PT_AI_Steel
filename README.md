# PT_AI
PT_AI - Pytorch AI

Pytorch is not neccesary if the model is to be run on the CPU.

Basic functioning AI with dense layers and a few optimizers.
The AI model is an object which requires the user to choose number of neuron layers, activation, loss and accuracy funtion, and optimizer. 

Flags: 
As all models are in the sampe script, there are inserted flags to controll which models are run. 
to_Tensor: if the model are to be run on a GPU, this flag must be set to True.
DO_BAYES_OPT: do bayesian optimazation of the hyper parameters 


Goal:
Return the steel profile which yields the lowest GWP (global warming potential).
User is required to give the two following inputs: span length [m], permanent distributed load [kN/m] and variable distributed load [kN/m].

As of now the AI has been trained to return a IPE, HEA, HEB or KFHUP (kaldformet kvad. HUP). 


The following is considered in the AIs scheme: 
- moment capasity
- shear capasity
- deformation

Limitations: 
- pointloads not considered
- only evenly distributed loads
- buckling not considered
- warping not considered
- NNFS pacakge required (used for replicating outputs from numpy randn()) if run on CPU
- Pytorch package required if run on GPU (pytorch converts all numpy arrays to Tensor objects)
