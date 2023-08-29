import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, 
                             precision_recall_curve,
                             ConfusionMatrixDisplay,
                             confusion_matrix,
                             classification_report)

def plot_roc_curve(predictions,save_as=None):
    plt.figure(figsize=(10,9))
    plt.rcParams['font.size'] = 15
    fpr,tpr,thr = roc_curve(predictions['true_class'], predictions['prob'], pos_label=None, sample_weight=None, drop_intermediate=True)

    plt.grid()
    plt.plot(fpr,tpr,color='darkorange')
    plt.plot((0,1),(0,1),linestyle='--',color='grey',alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend([f'Anomaly Detection (AUC={auc(fpr,tpr):.2f})'])

    plt.title("ROC curve")

    if save_as is not None:
      plt.savefig(save_as)
    plt.show()
    return fpr,tpr,thr

def plot_pr_curve(predictions,save_as=None):
    plt.figure(figsize=(10,9))
    plt.rcParams['font.size'] = 15
    prec,rec,thr = precision_recall_curve(predictions['true_class'], predictions['prob'])

    plt.grid()
    plt.plot(prec,rec,color='darkorange')
    plt.plot((0,1),(0,1),linestyle='--',color='grey',alpha=0.5)
    plt.xlabel("Precision")
    plt.ylabel("Recall")

    plt.title("Precision/Recall curve")
    if save_as is not None:
      plt.savefig(save_as)
    plt.show()

def plot_classification_report(predictions, th, save_fig_as=None):
    y_pred = predictions['prob']>th
    y_true = predictions['true_class']

    cm = confusion_matrix(y_true,y_pred)
    disp  = ConfusionMatrixDisplay(cm)
    disp.plot()
    if save_fig_as is not None:
      plt.savefig(save_fig_as)
    plt.show()

    print("\nClassification Report")
    print(classification_report(y_true,y_pred))

def plot_distributions(predictions, th,save_as = None):
    plt.figure(figsize=(10,10))
    plt.yscale('log')
    predictions['prob'].hist(bins = 200,color='black')
    predictions[predictions['true_class']==0]['prob'].hist(bins = 200,color='lime',alpha=0.7,)
    predictions[predictions['true_class']==1]['prob'].hist(bins = 200,color='red',alpha=0.7)
    ylim = plt.gca().get_ylim()

    plt.vlines(th, *ylim,color='blue')
    plt.gca().set_ylim(ylim)
    plt.legend(['Threshold','Global Distrib.','Negative Distrib.','Positive Distrib.'])
    if save_as is not None:
      plt.savefig(save_as)
    plt.show()