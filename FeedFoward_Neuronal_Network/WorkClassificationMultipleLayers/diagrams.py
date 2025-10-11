import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("employee_promotion.csv")

# Create a pairplot with 'Promotion' as the hue to distinguish promoted vs. non-promoted
sns.pairplot(df, hue="Promotion", diag_kind="hist")

# Display the plot
plt.show()