import pandas as pd
df = pd.read_csv('D:\email dataset\enron\emails.csv')  #path of the dataset
if 'message' in df.columns:
         df = df.rename(columns={'message': 'raw_email'})
df = df[['raw_email']]  
df.to_csv('raw_emails.csv', index=False)
print(f"Loaded {len(df)} emails from CSV.")
print('Emails saved to raw_emails.csv')

     
