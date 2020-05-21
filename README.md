







# Bellyband Synthetic Datasets


![image](https://github.com/drexelwireless/bellyband-datasets/blob/master/2020/DYSE_multitag/figures/17_15bpm3c40sps_transfft.png)


 - Synthetic Datasets used for simulation, training, and testing
 - Uses _SimBaby_, _MultiTag_, and _DYSE_
 - These datasets are _not_ subject to HIPAA, as they are not real (safe to disclose)




## File Naming Conventions

Each directory uses the following naming convention:

`**bpm**c**sps_**sw`

 - `**bpm`: _breaths per minute_
 - `**c`: _noise scale_ (factor used to scale noise using rayleigh fading model)
 - `**sps`: _samples per second_
 - `**sw`: length of _sliding window_ in seconds

In each directory is `snr_vals.csv`. The index of each row corresponds with the number assigned to each data set, i.e. `00`, `01`, `02`, etc.





## Recommended Tools

 - [_Python 3_ (CSV module)](https://docs.python.org/3/library/csv.html)
 - _SQLite_
 - _Bash_
 - [_CSVKit_](https://github.com/wireservice/csvkit)






