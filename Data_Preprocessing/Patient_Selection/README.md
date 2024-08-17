# Patient_Selection

In this section we screened patients with sepsis by calculating the SOFA score for all patients in MIMIC III and using this to screen for time periods of illness.

### Step Zero

Convert the metrics needed to calculate SOFA in MySQL to csv.
```
python Data_Preprocessing/Patient_Selection/Convert_to_CSV.py
```

### Step One

Integrate the above clinical indicators into reformat.csv.
```
python Data_Preprocessing/Patient_Selection/Variable_Consolidation.py
```

### Step Two

Addition of liquid clinical indicators.

```
python Data_Preprocessing/Patient_Selection/Liquid_Consolidation.py
```

### Step Three

Calculating SOFA.

```
python Data_Preprocessing/Patient_Selection/Calculate_SOFA.py
```
### Step Four

Screening of sepsis patients for duration of illness.
You will need to run Sepsis_Time.sql.
Note: Please import sofa.csv into MySQL's mimic3_sepsis_cleaned before running Sepsis_Time.sql.


