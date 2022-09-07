import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import *

log_path_train_batch = "./outputs/logs/batch_train_logs.csv"
log_path_train_epoch = "./outputs/logs/epoch_train_logs.csv"
log_path_test_batch = "./outputs/logs/batch_test_logs.csv"


df_train_batch = pd.read_csv(log_path_train_batch)
df_train_epoch = pd.read_csv(log_path_train_epoch)
df_test_batch = pd.read_csv(log_path_test_batch)

#------------------------------------------------------#
#------------------------------------------------------#
#plot of training the model for batches
#1....train loss plot
train_batch = np.array(df_train_batch["batch"])
train_loss = np.array(df_train_batch["loss"])
plt.title("Train Loss Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.plot(train_batch, train_loss)
plt.show()
plt.savefig("./outputs/reports/train_plots/train_loss.png")
plt.clf()
#2....train accuracy plot
train_accuracy = np.array(df_train_batch["accuracy"])
plt.title("Train Accuracy Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("Accuracy")
plt.plot(train_batch, train_accuracy)
plt.show()
plt.savefig("./outputs/reports/train_plots/train_accuracy.png")
plt.clf()
#3....train f1 score plot
train_f1_score = np.array(df_train_batch["f1_score"])
plt.title("Train F1_Score Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("F1_Score")
plt.plot(train_batch, train_f1_score)
plt.show()
plt.savefig("./outputs/reports/train_plots/train_f1_score.png")
plt.clf()
#4....train precision plot
train_precision = np.array(df_train_batch["precision"])
plt.title("Train Precision Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("Precision")
plt.plot(train_batch, train_precision)
plt.show()
plt.savefig("./outputs/reports/train_plots/train_precision.png")
plt.clf()
#5....train recall plot
train_recall = np.array(df_train_batch["recall"])
plt.title("Train Recall Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("Recall")
plt.plot(train_batch, train_recall)
plt.show()
plt.savefig("./outputs/reports/train_plots/train_recall.png")
plt.clf()
#------------------------------------------------------#
#------------------------------------------------------#

#plot of training the model for epochs
#1....train epochs loss plot
train_epochs = np.array(df_train_epoch["epoch"])
print(train_epochs)
train_epochs_loss = np.array(df_train_epoch["loss"])
val_epochs_loss = np.array(df_train_epoch["val_loss"])
plt.title("Train Loss Plot with Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(train_epochs, train_epochs_loss, train_epochs, val_epochs_loss)


plt.show()
plt.savefig("./outputs/reports/train_plots/train_val_epochs_loss.png")
plt.clf()
#2....train epochs accuracy plot
train_epochs_accuracy = np.array(df_train_epoch["accuracy"])
val_epochs_accuracy = np.array(df_train_epoch["val_accuracy"])
plt.title("Train Accuracy Plot with Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(train_epochs, train_epochs_accuracy, train_epochs, val_epochs_accuracy)

plt.show()
plt.savefig("./outputs/reports/train_plots/train_val_epochs_accuracy.png")
plt.clf()
#3....train epochs f1 score plot
train_epochs_f1_score = np.array(df_train_epoch["f1_score"])
val_epochs_f1_score = np.array(df_train_epoch["val_f1_score"])
plt.title("Train F1_Score Plot with Epochs")
plt.xlabel("Epochs")
plt.ylabel("F1_Score")
plt.plot(train_epochs, train_epochs_f1_score, train_epochs, val_epochs_f1_score)

plt.show()
plt.savefig("./outputs/reports/train_plots/train_val_epochs_f1_score.png")
plt.clf()
#4....train epochs precision plot
train_epochs_precision = np.array(df_train_epoch["precision"])
val_epochs_precision = np.array(df_train_epoch["val_precision"])
plt.title("Train Precision Plot with Epochs")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.plot(train_epochs, train_epochs_precision, train_epochs, val_epochs_precision)

plt.show()
plt.savefig("./outputs/reports/train_plots/train_val_epochs_precision.png")
plt.clf()
#5....train epochs recall plot
train_epochs_recall = np.array(df_train_epoch["recall"])
val_epochs_recall = np.array(df_train_epoch["val_recall"])
print(train_epochs_recall)
plt.title("Train Recall Plot with Epochs")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.plot(train_epochs, train_epochs_recall, train_epochs, val_epochs_recall)

plt.show()
plt.savefig("./outputs/reports/train_plots/train_val_epochs_recall.png")
plt.clf()
#------------------------------------------------------#
#------------------------------------------------------#
#plot of testing the model for batches
#1....train loss plot
test_batch = np.array(df_test_batch["batch"])
test_loss = np.array(df_test_batch["loss"])
plt.title("test Loss Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.plot(test_batch, test_loss)
plt.show()
plt.savefig("./outputs/reports/test_plots/test_loss.png")
plt.clf()
#2....train accuracy plot
test_accuracy = np.array(df_test_batch["accuracy"])
plt.title("test Accuracy Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("Accuracy")
plt.plot(test_batch, test_accuracy)
plt.show()
plt.savefig("./outputs/reports/test_plots/test_accuracy.png")
plt.clf()
#3....train f1 score plot
test_f1_score = np.array(df_test_batch["f1_score"])
plt.title("test F1_Score Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("F1_Score")
plt.plot(test_batch, test_f1_score)
plt.show()
plt.savefig("./outputs/reports/test_plots/test_f1_score.png")
plt.clf()
#4....train precision plot
test_precision = np.array(df_test_batch["precision"])
plt.title("test Precision Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("Precision")
plt.plot(test_batch, test_precision)
plt.show()
plt.savefig("./outputs/reports/test_plots/test_precision.png")
plt.clf()
#5....train recall plot
test_recall = np.array(df_test_batch["recall"])
plt.title("test Recall Plot with Batches")
plt.xlabel("Batches")
plt.ylabel("Recall")
plt.plot(test_batch, test_recall)
plt.show()
plt.savefig("./outputs/reports/test_plots/test_recall.png")
plt.clf()
    
#------------------------------------------------------#
#------------------------------------------------------#
            
