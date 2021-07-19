# Parameters explanation


## amlsim_conf.json
- `general`: used in `scripts/transaction_graph_generator.py` and in `src/amlsim/SimProperties.java`.
```json5
{
  "general": {
      "random_seed": 0,  // Seed of random number
      "simulation_name": "sample",  // Simulation name (identifier)
      "total_steps": 720,  // Total simulation steps
      "base_date": "2017-01-01"  // The date corresponds to the step 0 (the beginning date of this simulation)
  },
//...
}
```

- `default`: used in `scripts/transaction_graph_generator.py` and in `src/amlsim/SimProperties.java`. 
  See accounts for specifications on most variables.
  - `margin_ratio`: each member in an aml network will keep this ratio of the received amount
  - `cash_in` / `cash_out`: not used since the type of possible transactions is defined in `transactionType.csv`
  
- `input`: only used in `scripts/transaction_graph_generator.py`.
  - `is_aggregated_accounts`: when True, the accounts from the accounts file are used. 
    Otherwise, accounts are created randomly based on the default input parameters.
  - `accounts`: accounts should at least have :
    - `count`: number of accounts with the same distributions
    - `min_balance`: minimum initial balance
    - `max_balance`: maximum initial balance
    - `country`: Alpha-2 country code
    - `business_type`: type of business (bit straightforward)
    - `model`: Account behavior model ID (See also `AbstractTransactionModel.java`)
      - 0: Single transactions
      - 1: Fan-out
      - 2: Fan-in
      - 3: Mutual
      - 4: Forward
      - 5: Periodical
    - `bank_id`: Bank ID which these accounts belong to
  - `alert_patterns`: the aml patterns that should be created
    - `count` Number of typologies (transaction sets)
    - `type` Name of transaction type (`fan_in`, `fan_out`, `cycle`...) as the AML typology
    - `schedule_id` Transaction scheduling ID of the typology
      - 0: All member accounts send money in order with the same interval (number of days)
      - 1: All member accounts send money in order with random intervals
      - 2: All member accounts send money randomly
    - `min_accounts`: Minimum number of involved accounts
    - `max_accounts`: Maximum number of involved accounts
    - `min_amount` Minimum initial transaction amount
    - `max_amount` Maximum initial transaction amount
    - `min_period` Minimum overall transaction period (number of days)
    - `max_period` Maximum overall transaction period (number of days)
    - `bank_id` Bank ID which member accounts belong to (optional: if empty, no limitation for the bank ID) 
    - `is_sar` Whether the alert is SAR (True) or false alert (False) when you want to add normal accounts producing an aml pattern.
  - `degree`: This CSV file has three columns with header names: `Count`, `In-degree` and `Out-degree`. 
    Each CSV row indicates how many account vertices with certain in(out)-degrees should be generated. 
    Might be able to generate a degree file by running `scripts/generate_scalefree.py` which expects as arguments number of vertices and edges and the file name.
    When `isolated_aml` is true, the degrees won't completely match with the output.
  - `transaction_type`: This CSV file has two columns with header names: `Type` (transaction type name) 
  and `Frequency` (relative number of transaction frequency). In this case only the type "TRANSFER" is used.
```json5
{
//...
  "input": {
    "directory": "paramFiles/1K",  // Parameter directory
    "schema": "schema.json",  // Configuration file of output CSV schema
    "accounts": "accounts.csv",  // Account list parameter file
    "alert_patterns": "alertPatterns.csv",  // Alert list parameter file
    "degree": "degree.csv",  // Degree sequence parameter file
    "transaction_type": "transactionType.csv",  // Transaction type list file
    "is_aggregated_accounts": true  // Whether the account list represents aggregated (true) or raw (false) accounts
  },
//...
}
```

- `temporal`: used in `scripts/transaction_graph_generator.py` and in `src/amlsim/SimProperties.java`. 
  Contains some csv paths that are used temporarily during the dataset creation and are removed afterwards.
- `output`: used in `scripts/transaction_graph_generator.py` and in `src/amlsim/SimProperties.java`.
```json5
{
  //...
  "output": {
    "directory": "outputs",
    // Output directory
    "accounts": "accounts.csv",
    // Account list CSV
    "transactions": "transactions.csv",
    // All transaction list CSV
    "cash_transactions": "cash_tx.csv",
    // Cash transaction list CSV
    "alert_members": "alert_accounts.csv",
    // Alerted account list CSV
    "alert_transactions": "alert_transactions.csv",
    // Alerted transaction list CSV
    "sar_accounts": "sar_accounts.csv",
    // SAR account list CSV
    "party_individuals": "individuals-bulkload.csv",
    "party_organizations": "organizations-bulkload.csv",
    "account_mapping": "accountMapping.csv",
    "resolved_entities": "resolvedentities.csv",
    "transaction_log": "tx_log.csv",
    "counter_log": "tx_count.csv",
    "diameter_log": "diameter.csv"
  },
  //...
}
```

- `graph_generator`: only used in `scripts/transaction_graph_generator.py`.
  - `isolated_aml`: when true, the aml network will start as an isolated network (so the SAR accounts are not involved 
    in any other transactions beside the aml transactions). When false, the accounts can have normal transactions 
    between SAR and SAR and between normal and SAR.
  - `degree_threshold`: hub accounts with a larger degree (in + out) than the specified threshold are chosen as the main 
    account candidates of alert transaction sets. This should be set to 0 when `isolated_aml` is true.
  - `ratio_sar_to_normal`: adds edges from SAR accounts to normal accounts. The number of edges to be added is the 
    ratio normal_to_sar of SAR accounts
  - `ratio_normal_to_sar`: adds edges from normal accounts to SAR accounts. The number of edges to be added is the 
    ratio normal_to_sar of SAR accounts
  - `high_risk_countries`: a list of high risk countries (seems not used).
  - `high_risk_business`: a list of high rusk business types (seems not used)
  
- `aml_patterns`: only used in `scripts/transaction_graph_generator.py`. Contains the different typologies and corresponding enums.

- `simulator`: only used in `src/amlsim/SimProperties.java`.
  - compute_diameter: Compute diameter and average distance of the transaction graph.
  - transaction_limit
  - transaction_interval: Default transaction interval for normal accounts
  - sar_interval: not used?
  - sar_balance_ratio: not used?
  - numBranches: number of branches (for cash transactions, so not used?)
  
- `visualizer`: only used in `scripts/visualize/plot_distributions.py`. Contains the file names of the different images that will be created.

## engine_conf.json
- `days_to_process`: a list containing dates. Transactions are collected per day. The dates denote the transaction dates
  the AML engine should check.

## learning_conf.json
- `agent`: specifies which model for the agent should be used, choices are `baseline`, `reinforcement`, `svm` and `neural_network`.
- `step`: the current step of the simulation.
- `process_alerts`:
  - `paths`: the paths for inputs, outputs and logs.
  - `base_date`: seems to do nothing?
  - `full_investigation`: whether a full investigation should take place. This implies that if at least one transaction 
    in the AML network is found, then all transactions will eventually be found. In other words, if there is one AML 
    network and one transaction is found, recall should be 1.
- `some_model`: one of the possible agents with corresponding variables.
