# PREDICTORS OF SUBJECTIVE WELL-BEING

## Brief Description

The following code allows user to perform feature subset selection, calculate the prediction accuracy of the obtained subset and correspoding 
statistical significance of each factor, combine everything in the data table and calculate regression coefficients.

## Datasets 

The datasets can be found in the folder **"input data"**.

* **A008.W.pkl** > Feeling of Happiness
* **A170.W.pkl** > Satisfaction with Life
* **SWB.LS.pkl** > Subjective Well-Being, Life Satisfaction
* **ranking_iris.pkl** > Ranking scores of countries made by me
* **ranking_survey.pkl** > Ranking scores of countries from the survey
* **\<target\>_measure_units.csv** > The subset of factors with corresponding measure units 
* **indicators_shorter.csv** > Data table containing a shorter list of factors

## Step 1. Feature Subset Selection

* Name of the file: **"fss.py"**
* Read the data: Change the filepath in line 358.
* Running in different modes: Modes are described in function *model_acuracy*. To select a mode, set the variable **type** in line 360.
* Define the strenght of regularization: Change range in line 363.
* Output: List of 10 factors for each ranking method and R-squared for LR and RF.

## Step 2. Statistical Significance

* Name of the file: **"p-values.py"**
* Read the data: Change the filepath in line 129.
* Output: Dictionary of form [factor] --> (statistical significance, ranking method).   

## Step 3. Creating CSV Data Table

* Name of the file: **"data_table.py"**
* Pre-requirement: Before running this file you need to change some parameters in file *ffs.py*. The correct numbers are put in comments in functions *relief_top_attributes*, *linear_top_attributes* and *rf_top_attributes*.
* Read the data: Change the filepath in line 8.
* Import 2 additional data tables: Change the filepath in line 37 for *indicators_shorter.csv* and line 47 for *\<target\>_measure_units.csv*. 
* Output: Data table saved as **\<target\>_data_table.csv** in new folder **export data**.


## Step 4. Regression Coefficients

* Name of the file: **"multiple_linear_regression.py"**
* Read (all) the data: Change the filepath(s) in line 11, with manually chosen factors.
* Import additional data table: Change the file in line 94 for *indicators_shorter.csv*
* Output: Data table saved as **\<target\>_regression_coeffs.csv** in folder **export data**.

## Step 5.* Results of the Pilot Survey
Materials for further steps can be found in **survey** folder.

a. Rankings Scores of Countries

* Name of the file: **"country_preference_results.py"**
* Output: Data table saved as **out.csv**

b. Data Table of Demographics and Macroeconomic Statements

* Name of the file: **"demo_data_table.py"**
* Output: Data table saved as **demo_data.csv**
