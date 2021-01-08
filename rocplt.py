###############################################################################
# rocplt.py
#
# sheneman@uidaho.edu
#
# Simple brutish helper script to Generate a ROC curve plot given a CSV of the 
# correct format.  Used to make the ROC CURVE figure in related manuscript.
#
# Usage: python rocplt.py
#
###############################################################################

import os, sys
import pandas as pd
import numpy
import matplotlib.pyplot as plt

#  Fle Header
#  SVM FPR,SVM TPR,SVM AUC,XGB FPR,XBG TPR,XGB AUC,RF_FPR,RF_TPR,RF AUC,MLP FPR,MLP TPR,MLP AUC,LDA FPR,LDA TPR,LDA AUC,CNN FPR,CNN TPR,CNN AUC

ROC_AUC_FILE = "./FINAL_ROC_AUC.csv"
OUTFILE      = "ROC_AUC.eps"


# load the results table to graph
result_table = pd.read_csv(ROC_AUC_FILE)
print(result_table.head())  # for debugging

fig = plt.figure(figsize=(12,8),dpi=600)



plt.plot(result_table['XGB FPR'],
result_table['XGB TPR'],
label="XGB AUC={:.3f}".format(result_table['XGB AUC'][0]))

plt.plot(result_table['RF FPR'],
result_table['RF TPR'],
label="   RF AUC={:.3f}".format(result_table['RF AUC'][0]))

plt.plot(result_table['MLP FPR'],
result_table['MLP TPR'],
label="MLP AUC={:.3f}".format(result_table['MLP AUC'][0]))

plt.plot(result_table['LDA FPR'],
result_table['LDA TPR'],
label="LDA AUC={:.3f}".format(result_table['LDA AUC'][0]))

plt.plot(result_table['CNN FPR'],
result_table['CNN TPR'],
label="CNN AUC={:.3f}".format(result_table['CNN AUC'][0]))

plt.plot(result_table['SVM FPR'],
result_table['SVM TPR'],
label="SVM AUC={:.3f}".format(result_table['SVM AUC'][0]))


plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(numpy.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(numpy.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.savefig(OUTFILE);

exit(0)

