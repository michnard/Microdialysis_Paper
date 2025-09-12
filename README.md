This repository contains the data, scripts, and results; all figures and analyses can be regenerated from raw data using the scripts provided.


Structure:
microdialysis_paper/
│── data/
│   ├── raw/             # untouched instrument exports (immutable)
│   ├── processed/       # cleaned & normalized datasets
│   └── metadata/        # animal/sample metadata, QC notes
│
│── scripts/
│   ├── preprocessing/   # normalization, QC, filtering
│   ├── analysis/        # statistics and group comparisons
│   ├── figures/         # reproducible figure generation
│   └── utils/           # helper functions (e.g., plotting, stats)
│
│── results/
│   ├── figures/         # exported figures for manuscript
│   ├── tables/          # statistical summaries
│   └── logs/            # run logs, reproducibility notes
│
└── README.md            # this file



Python: pandas, numpy, matplotlib, statsmodels

Environment specification: see environment.yml
