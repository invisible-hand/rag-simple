import pandas as pd

# Replace 'mydata.csv' with your CSV filename
csv_file_path = "Bank_Personal_Loan_Modelling(1).csv"

# Read CSV into a Pandas DataFrame
df = pd.read_csv(csv_file_path)

# Convert DataFrame to Parquet
# Replace 'mydata.parquet' with your desired Parquet output filename
parquet_file_path = "bank_loan.parquet"
df.to_parquet(parquet_file_path, engine='pyarrow')

print("Conversion complete!")