# paper_phosphagen2

Simulation code for the paper "Optimal Neuromuscular Performance Requires Motor Neuron Phosphagen Kinases" by Justs et al. ([bioRxiv link](https://doi.org/10.1101/2025.03.18.643998)).

Instructions:

- Install miniconda ([instructions](https://docs.conda.io/en/latest/miniconda.html)).

- Create the `phos2` conda environment using `phos2.yml` ([instructions](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)).

- Open jupyter notebook from within the newly created `phos2` environment.

- Run the notebook `simulate.ipynb`. This performs the simulations, saves the data needed to create simulation figures in `figure-data`, and saves the figures in `figures`.

If issues arise while resolving the environment on a Windows system, additional steps may be required.  These are:

- Installing Visual Studio 2022

- Downgrading Python to Python version 3.12.04

- In the python terminal, running 'conda install m2w64-gcc-libgfortran'

- Finally, run 'pip install numbaminpack'
