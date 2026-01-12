import pandas as pd

df = pd.read_csv("labeled_emails.csv")

sample_df = df.sample(n=5000, random_state=42)

sample_df.to_csv("labeled_emails_sample.csv", index=False)

print("Sample file created")
