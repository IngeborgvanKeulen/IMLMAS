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

5. Follow the instructions in the README in `AMLSim/jars`
   
[comment]: <> (6. To have all the modules available, run)

[comment]: <> (``` python)

[comment]: <> (pip install -e .)

[comment]: <> (```)


## Running instructions
In the directory `toy_examples` the four different models (Random, SVM, DQN and BDQN) scripts can be found.
These scripts only work with some easy business rules that are defined in `create_action_space.py` in the same 
directory. You can change the used parameters and add/modify/delete business rules in these scripts.
Each one of them can be run with `python3 launder_as_much_<model>.py`. If everything went okay, then you should
see a directory `setting_1000_<id>` (check the path in the script to see where it should have written to). 

In this directory you can find a plot, showing the learning behaviour (in case of BDQN this one does not make much 
sense since it includes all heads), a json file containing the parameters used in the setting, and a CSV file
containing all the states-actions taken during the games. 
In case of BDQN the final weights of the heads are also stored.

For more complex business rules the file `Game.py` in the directory `q_agent` can be used. It is called with
`python3 Game.py`, all the needed params need to be defined in the script itself. Note that you need to link a 
business rule engine (some code that detects transactions according to some logic) yourself. 

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
