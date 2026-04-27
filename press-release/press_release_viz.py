import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/Loan_default.csv")

# Check columns
print(df.columns.tolist())

# Make sure these column names match your file
credit_col = "CreditScore"
target_col = "Default"

# Create credit score bins
df["credit_score_group"] = pd.cut(
    df[credit_col],
    bins=[300, 580, 670, 740, 800, 900],
    labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    include_lowest=True
)

# Calculate default rate by group
default_rate = df.groupby("credit_score_group")[target_col].mean().reset_index()

# Plot
colors = ["red", "orange", "gold", "skyblue", "green"]

plt.figure(figsize=(8,5))
plt.bar(
    default_rate["credit_score_group"].astype(str),
    default_rate[target_col],
    color=colors
)
plt.xlabel("Credit Score Category")
plt.ylabel("Default Rate")
plt.title("Default Rate by Credit Score Category")
plt.xticks(rotation=20)
# plt.show()  # either move this after savefig, or remove it
plt.savefig("../press-release/default_rate_by_credit_score.png", dpi=300)
plt.show()  # show after saving