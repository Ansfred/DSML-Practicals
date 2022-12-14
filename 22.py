cm = {  # confusion matrix
    "TP": 90,
    "FP": 140,
    "FN": 210,
    "TN": 9560
}

accuracy = (cm["TP"] + cm["TN"]) / (cm["TP"] + cm["FN"] + cm["FP"] + cm["TN"])
error_rate = (cm["FP"] + cm["FN"]) / (cm["TP"] + cm["FN"] + cm["FP"] + cm["TN"])
precision = cm["TP"] / (cm["TP"] + cm["FP"])  # Correctly predicted positives / Model predicted positives
recall = cm["TP"] / (cm["TP"] + cm["FN"])  # Correctly predicted positives / Actual positives

print("Accuracy = {}".format(accuracy))  # >
print("Error Rate = {}".format(error_rate))  # <
print("Precision = {}".format(precision))  # >
print("Recall = {}".format(recall))  # >