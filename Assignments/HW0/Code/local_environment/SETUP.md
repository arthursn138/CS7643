# CS 4464/7643 Deep Learning Local Environment Setup

To run the notebook locally, we will give a recommended setup using miniforge and VSCode.

First, [install miniforge](https://github.com/conda-forge/miniforge#install) or any other conda installation you are comfortable with and [install VS Code](https://code.visualstudio.com/) along with the [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

Then, run this command within this folder: `conda env create -f environment.yml`

Finally, open the **root homework folder** with VS Code and ensure the ipykernel in the top right is changed to `dl_hw0`.

If you want to run anything within your command line, ensure that you have run `conda activate dl_hw0` to change
to the correct conda environment.