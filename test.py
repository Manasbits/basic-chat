# Add this test before running the agent
import pandas as pd

# Load and inspect your CSV
df = pd.read_csv("query_results.csv")
print("CSV Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst few rows of Name column:")
print(df.iloc[:, 0].head())  # First column (Name)

# Test the exact search
name_col = "Name (Company Name as registered on stock exchanges)"
exact_match = df[df[name_col] == 'Reliance Industries']
print(f"\nExact match results: {len(exact_match)} rows")

partial_match = df[df[name_col].str.contains('Reliance', na=False)]
print(f"Partial match results: {len(partial_match)} rows")
if len(partial_match) > 0:
    print("Partial match data:")
    print(partial_match[name_col].values)