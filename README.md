# master_thesis


## Prerequisites
Ensure `javac` is set to 1.8 by running
``` python
javac -version
```

If you don't have any javac installed, you can install it with
``` python
sudo dnf install java-1.8.0-openjdk.x86_64
sudo dnf install java-1.8.0-openjdk-devel.x86_64
```

If the version is not set to 1.8, then run
``` python
sudo update-alternatives –config javac
```

You might also need to run `chmod +x` on `run_simulation`.


## Installation instructions
1. Create a virtual environment (only tested with python 3.7), for example by running
``` python
conda create -n venv python=3.7
```

Conda is recommended for making your life easier with installing pygraphviz in the next steps.
See https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html for installation instructions for conda.

2. Activate your virtual environment
``` python 
conda activate venv
```

3. For some reason installing `pygraphviz` fails, which is solved by running
 ``` python
 conda install pygraphviz
 ```
4. Install the requirements by running
``` python
pip install -r requirements.txt
```

5. Follow the instructions in the README in `AMLSim/jars` (or ask Ingeborg for a zip ;) and extract the zip in 
   `AMLSim/jars/`)
   
6. To have all the modules available, run
``` python
pip install -e .
```


## Running instructions
The script `start_new_run.py` should be run when you start a new simulation.
In this script you can set some of the parameters. You can also change them yourselves in the confs, but this requires 
some knowledge about the simulation and models.

If you want to continue a simulation, or have run `start_new_run.py` you can start the simulation by 
running the bash script `RUN_SIMULATION`. You can change the number of steps by changing the counter in the while 
condition (line 5).

If everything went okay, then you should see a directory with the date and time you started the simulation 
in `learning_agent/results`. This directory should contain the `amlsim_conf.json` and the `learning_conf.json` 
as well as `alertPatterns.csv`. They should include enough information to reproduce the same run 
(there will be some randomness in the simulation though). This directory also contains the output of the simulation in 
three files: `results.csv`, `laundered_actions.csv` and `prevented_actions.csv`.

#### WARNING
If you run a simulation you cannot open a file in the results folders since this folder's modification will change and be the latest directory.
The results are written to the latest directory and therefore these results will be overwritten.

### Parameters
There are a lot of variables that can be changed. (like a lot lot)
- business rules can be changed in `br_engine/business_rules.py`, 
  the days the rules should be applied to are defined in `confs/engine_conf.json`
- learning agent variables can be changed in `confs/learning_conf.json`
- simulation variables can be changed in `confs/amlsim_conf.json`, otherwise:
  - the AML agents in `AMLSim/paramFiles/ml_agent/alertPatterns.csv`
  - the normal agents in `AMLSim/paramFiles/ml_agent/accounts.csv` when ```is_aggregated_accounts=True```, 
    otherwise it should work in `confs/amlsim_conf.json`
  - the degrees in `AMLSim/paramFiles/ml_agent/degree.csv`. 
    Haven't tested yet, but they possibly can be generated with the script `AMLSim/scripts/generate_scalefree.py` given the arguments `num_vertices` and `edge_factor`
    
See ```PARAMETERS.md``` for an extensive explanation of all parameters.
