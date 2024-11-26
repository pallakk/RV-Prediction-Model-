import pandas as pd

# Load the CSV file
csv_file = 'patients_data.csv'
patients_data = pd.read_csv(csv_file)

# Load the Excel file
excel_file = 'RVoutcomes.xlsx'
rv_outcomes = pd.read_excel(excel_file)

# Assuming 'patId' is the common key to merge the files
# Merge the dataframes on 'patId'
merged_data = pd.merge(patients_data, rv_outcomes, on='patId', how='left')

# Save the merged dataframe back to a CSV
merged_csv_file = 'patients_data_with_RVoutcomes.csv'
merged_data.to_csv(merged_csv_file, index=False)

print(f"Merged data saved to {merged_csv_file}")
