# Comparing the representation of a simple visual stimulus across the cerebellar network
Analysis code and notebooks for the analysis and figures of the paper "Comparing the representation of a simple visual stimulus across the cerebellar network" (Prat, Petrucco, Štih, Portugues).

## Installation instructions

Download data and clone repo:
1. Download the data
2. Uncompress locally the folder
3. From the terminal, clone the `Prat_et_al` repo on your computer:
    ```bash
    > git clone https://github.com/portugueslab/Prat_et_al
    ```
4. `cd` to the package location:
    ```bash
    > cd Prat_et_al
    ```
5. [Optional] Create a new environment to run the script:
    ```bash
    > conda create -n prattest python==3.8.10
    > conda activate prattest
    ```
6. and install it in editable mode:
    ```bash
    > pip install -e . 
    ```

## Description of the repo
The final code to replicate all plots from the manuscript can be found in the `figures_notebooks` folder.

The code to perform the decoding analysis appearing in some of the figures can be found in the `decoding` folder. This should be run before the notebooks to reproduce the figures, as some data not provided in the zenodo link needs to be generated. 

The `luminance_analysis` module contains all utility functions used throughout the repository, for loading and organizing the data, performing its analysis as well as for the modeling and making the plots in the manuscript.
