# Final_Integration

Final screening, cleansing and repair of data

### Step Zero 

Take all the required clinical features from MySQL and convert them to csv.

```
python Data_Preprocessing/Final_Integration/SQL_to_CSV.py
```

### Step One

Data cleansing, patching and special variable calculations

```
python Data_Preprocessing/Final_Integration/Sepsis_Integrate.py
```

After this step, you will get 'mimictable.csv' and pickle file to be used in the next step.