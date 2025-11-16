This repository contains the most important data, scripts, and results; all figures and analyses can be regenerated from raw data using the scripts provided.


Structure:

microdialysis_paper/

│── data/

│   ├── raw/             # LCMS data and Video tracking (not here -- download from Figshare 10.25378/janelia.30556511)

│   ├── processed/       # cleaned & normalized datasets

│   └── metadata/        # experiments metadata & notes

│── scripts/

│   ├── preprocessing/   # data preprocessing

│   ├── analysis/        # statistics and figures
│

├── figures/         	 # exported figures for manuscript

├── environment.yml      # Environment specification

└── README.md     

It is reccomended to run all the scripts in the environment specifications above:

If using mamba (reccomended), use

>> mamba env create -f environment.yml

or the equivalent conda command

>> conda env create -f environment.yml

and then

>> mamba activate Microdialysis      

or

>> conda activate Microdialysis       
