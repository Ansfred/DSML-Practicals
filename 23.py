# https://towardsdatascience.com/entropy-and-information-gain-in-decision-trees-c7db67a3a293
# https://www.featureranking.com/tutorials/machine-learning-tutorials/information-gain-computation/

import pandas as pd
import numpy as np
import math

data = {
    "Age": ["Young", "Young", "Middle", "Old", "Old", "Old", "Middle", "Young", "Young", "Old", "Young", "Middle",
            "Middle", "Old"],
    "Income": ["High", "High", "High", "Medium", "Low", "Low", "Low", "Medium", "Low", "Medium", "Medium", "Medium",
               "High", "Medium"],
    "Married": ["No", "No", "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "No"],
    "Health": ["Fair", "Good", "Fair", "Fair", "Fair", "Good", "Good", "Fair", "Fair", "Fair", "Good", "Good", "Fair",
               "Good"],
    "Class": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)
print()
print(df)
print()


def calculate_entropy(df, col="Class"):
    # Entropy = - (i=1)Σ(c) P(Xi)log2(P(Xi))
    # c = Total number of unique values in the column
    # P(Xi) = Probability of X where X ∈ Unique values
    df_freq_table = df[col].value_counts()
    # print(df_freq_table)
    freq_counts = []
    total_count = 0
    for i in df_freq_table:
        freq_counts.append(i)
        total_count += i
    probabilities = [x / total_count for x in freq_counts]  # P(Xi)
    entropy = 0
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    entropy *= -1
    return entropy


entropy = calculate_entropy(df)
print("Entropy: " + str(entropy))

entropy_list = []
weight_list = []

for i in df.keys():
    print("------------------------------------")
    print(i)
    for age in df[i].unique():  # Split the freq table on attribute "Age"
        df_feature_age = df[df[i] == age]
        # print(df_feature_age)
        entropy_level = calculate_entropy(df_feature_age)  # Calculate entropy of each split subset
        entropy_list.append(entropy_level)  # Store the entropy of each subset in a list
        weight_level = len(df_feature_age) / len(df)  # Weight Level = Len of subset / Len of all subsets combined
        weight_list.append(weight_level)

    feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))
    print("Remaining Impurity: " + str(feature_remaining_impurity))
    information_gain = entropy - feature_remaining_impurity
    print("Information Gain: " + str(information_gain))
    print("------------------------------------")
