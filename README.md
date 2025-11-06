This repository contains the most important data, scripts, and results; all figures and analyses can be regenerated from raw data using the scripts provided.


Structure:

microdialysis_paper/

│── data/

│   ├── raw/             # LCMS data and Video tracking

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

>> conda env create -f environment.yml
>> conda activate Microdialysis       
