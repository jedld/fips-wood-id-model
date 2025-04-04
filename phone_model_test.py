import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Define phone models and the associated CSV column names for "Correct" and "Incorrect"
phone_models = {
    "iPhone SE": ("iPhone SE Correct", "iPhone SE Incorrect"),
    "iPhone 14 Pro": ("iPhone 14 Pro Correct", "iPhone 14 Pro Incorrect"),
    "Galaxy A52": ("Galaxy A52(baseline) Correct", "Galaxy A52(baseline) Incorrect"),
    "Poco F3": ("Poco F3 Correct", "Poco F3 Incorrect")
}

# Load the data
data = pd.read_csv('woodid_contingency_table.csv', index_col=0)

# Ensure all values are numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col])

# Create a contingency table for overall Chi-square test
# Each column is a phone model and the two rows are "Correct" and "Incorrect" totals.
contingency_data = {}
for phone, (col_correct, col_incorrect) in phone_models.items():
    contingency_data[phone] = [
        data[col_correct].sum(),
        data[col_incorrect].sum()
    ]
contingency_table = pd.DataFrame(contingency_data, index=['Correct', 'Incorrect'])

print("Contingency Table:")
print(contingency_table)
print("\n")

# Perform overall Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of freedom: {dof}")

# Interpret the overall results
alpha = 0.05
print("\nInterpretation:")
if p < alpha:
    print(f"The p-value ({p:.4f}) is less than {alpha}, so we reject the null hypothesis.")
    print("There is a significant association between phone model and performance.")
else:
    print(f"The p-value ({p:.4f}) is greater than {alpha}, so we fail to reject the null hypothesis.")
    print("There is no significant association between phone model and performance.")

# Display the expected frequencies for the overall test
print("\nExpected Frequencies:")
expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
print(expected_df)

# Calculate accuracy for each phone model
accuracies = {}
for phone, (col_correct, col_incorrect) in phone_models.items():
    total = data[col_correct].sum() + data[col_incorrect].sum()
    accuracies[phone] = data[col_correct].sum() / total * 100 if total > 0 else 0

print("\nAccuracy by Phone Model:")
for phone, acc in accuracies.items():
    print(f"{phone}: {acc:.2f}%")

# Calculate per-species accuracy for additional analysis
per_species_accuracy = pd.DataFrame(index=data.index)
for phone, (col_correct, col_incorrect) in phone_models.items():
    per_species_accuracy[phone] = 100 * np.where(
        (data[col_correct] + data[col_incorrect]) > 0,
        data[col_correct] / (data[col_correct] + data[col_incorrect]),
        0
    )

# Find species with different performance across phones
print("\nSpecies with different performance across phone models:")
variable_species = []
for species in per_species_accuracy.index:
    if per_species_accuracy.loc[species].std() > 1:  # Only consider meaningful differences
        print(f"{species}:")
        for phone in per_species_accuracy.columns:
            print(f"  {phone}: {per_species_accuracy.loc[species, phone]:.1f}%")
        variable_species.append(species)

# Pairwise Chi-square tests comparing each phone model (except the control) to Galaxy A52 (control)
def perform_pairwise_chi2(phone, phone_correct, phone_incorrect, galaxy_correct, galaxy_incorrect):
    table = pd.DataFrame({
        phone: [phone_correct, phone_incorrect],
        'Galaxy A52': [galaxy_correct, galaxy_incorrect]
    }, index=['Correct', 'Incorrect'])
    
    chi2_val, p_val, dof_val, expected_val = chi2_contingency(table)
    print(f"\nChi-square Test for {phone} vs Galaxy A52:")
    print(table)
    print(f"Chi-square statistic: {chi2_val:.4f}")
    print(f"p-value: {p_val:.4f}")
    print(f"Degrees of freedom: {dof_val}")
    if p_val < alpha:
        print(f"Result: p ({p_val:.4f}) < {alpha}, so we reject the null hypothesis.")
    else:
        print(f"Result: p ({p_val:.4f}) >= {alpha}, so we fail to reject the null hypothesis.")

# Galaxy A52 is the control model
control_phone = "Galaxy A52"
control_correct, control_incorrect = phone_models[control_phone]
galaxy_correct_sum = data[control_correct].sum()
galaxy_incorrect_sum = data[control_incorrect].sum()

for phone, (col_correct, col_incorrect) in phone_models.items():
    if phone == control_phone:
        continue
    perform_pairwise_chi2(
        phone,
        data[col_correct].sum(),
        data[col_incorrect].sum(),
        galaxy_correct_sum,
        galaxy_incorrect_sum
    )

# Visualize the results
plt.figure(figsize=(14, 10))

# Plot 1: Count bar chart for each model
plt.subplot(2, 2, 1)
contingency_table.T.plot(kind='bar', ax=plt.gca())
plt.title('Correct vs Incorrect Classifications by Phone Model')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Plot 2: Overall accuracy by model
plt.subplot(2, 2, 2)
accuracy_series = pd.Series(accuracies)
accuracy_series.plot(kind='bar', ax=plt.gca())
plt.title('Overall Accuracy by Phone Model')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.ylim(0, 100)

# Plot 3: Heatmap of the contingency table
plt.subplot(2, 2, 3)
sns.heatmap(contingency_table, annot=True, fmt='g', cmap='Blues')
plt.title('Contingency Table Heatmap')

# Plot 4: Per-species accuracy comparison for species showing variation
plt.subplot(2, 2, 4)
if variable_species:
    # Limit to 6 species if needed for readability
    species_subset = variable_species[:6] if len(variable_species) > 6 else variable_species
    per_species_subset = per_species_accuracy.loc[species_subset]
    per_species_subset.T.plot(kind='bar', ax=plt.gca())
    plt.title('Species with Variable Performance Across Phones')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.legend(fontsize='small')
else:
    plt.text(0.5, 0.5, "No species with variable performance found", 
             horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.savefig('phone_model_performance_analysis.png')
plt.show()