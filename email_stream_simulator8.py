import time
import random
import pandas as pd
from urgency_inference6 import predict_urgency

df = pd.read_csv("labeled_emails.csv")

print("Simulated email stream started\n")

while True:
    row = df.sample(1).iloc[0]
    urgency = predict_urgency(row["cleaned_text"])

    print("New Email")
    print("Subject:", row["subject"])
    print("Predicted Urgency:", urgency)
    print("-" * 40)

    time.sleep(random.randint(3, 6))
