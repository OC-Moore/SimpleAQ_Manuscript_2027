# SimpleAQ_Manuscript_2027
Scripts related to the Moore et. al 2027 paper analyzing SimpleAQ low-cost sensors
These scripts are walked through in order of execution. Please be aware that file paths will need to be updated to your personal machine.
1. First step is to download the data. Utilize the _**bulkdownload.txt**_ file for this step
2. Second step is the unzip the data in these files. Use the _**pike.sh**_ file for this step
3. Third step is to convert the text files into csv, while also filtering for the necessary information. Use the _**text_csv.sh**_ file for this
4. Fourth step is to combine the resulting csv files. Pull the humidity, temperature, and pollutant data from the csv as well as their respective time stamps into a new file. Name the headers to reflect the sensor nickname and the factor (rh, temp, pm, etc).
5. The next step is to pull the relevant EPA data. This requires creating an account to access the [API](https://aqs.epa.gov/aqsweb/documents/data_api.html).
   * Note: check for malfunctions in the EPA data file by checking in the qualifier column. A non-blank cell indicates a malfunction or reset of some kind.
6. Once these steps are completed, the file _**jp_compare_test.py**_ can be ran to compare the low-cost sensors at the site, and compare against the relevant EPA data.
7. Next, run _**jp_regress_coll_hourly.py**_ to run the regressions of the low-cost and EPA data. This is running a train test split on a portion of the data, and also plots several figures to describe comparability.
8. Next, to confirm the correct equation, run _**cross_validate_hourly.py**_. This file runs K validations instead of the least squares regression, this helps decrease randomness in the data. This file tests three different models as well as running the models on the combined datasets from more than one site.
9. Using the information from the last two files, pick the equation that most comprehensively explains the difference in data and corrects for it. In a separate excel file, apply the correction equation to the raw low-cost sensor data, creating a column for the corrected data.
10. Lastly, run the file _**corrected_compare.py**_ which runs the comparative statistics using the files with the correction equation applied.
