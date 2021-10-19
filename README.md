# Napari to process Zeiss lattice data

Please clone or download the repo. The conda environment required for running the llsz package can be installed using the environment.yml file. 

To enable fast configuration of conda environments, it is recommended to install [mamba](https://github.com/mamba-org/mamba) into your base environment.

    conda install mamba -n base -c conda-forge

Once this is installed, you can create a conda environment by replacing "conda" with "mamba"

Create the conda environment using .yml file

    conda create env create -f environment.yml


Due to potential configuration issues with gputools and pyopencl, they are to be installed separately.

Try installing gputools first:

    pip install gputools

If it works, then pypopencl should have been installed and you can start using the environment. 

****
**If gputools installation does not work**, then install pyopencl using [pre-built wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl) here.

For Python 3.9 use files with cp39 in their name. Download a file with name cl21 first, if that doesn't work, then try cl12. Download the file and then run 

    pip install pyopencl_file_name
 
Install gputools after successful pyopencl installation.
*****
## **Running llsz**

Once the environment is configured, you can open a terminal within the package folder and run:

    python main.py

This will start a napari instance.
The worfklow currently is:
* Open a czi file (skewed raw data)
* Deskew a single timepoint
* Use the deskewed stack and max projection for previewing and cropping an area of interest

![Preview of widget](resources/preview_video.gif)
