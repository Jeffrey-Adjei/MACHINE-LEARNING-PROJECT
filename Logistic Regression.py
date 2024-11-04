import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, RocCurveDisplay
import matplotlib.pyplot as plt
from itertools import cycle

def load_data():
    data = pd.read_csv('car+evaluation/car.csv',  index_col=False, header=None)
    data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target']

    return data

def preprocess(data):
    x = data.drop('target', axis=1)
    y = data['target']

    enc = OneHotEncoder(sparse_output=False, drop='first')
    x_enc = enc.fit_transform(x)
    
    pca = PCA(n_components=0.95)
    x_pca = pca.fit_transform(x_enc)

    return x_pca, y

def evaluate_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    
    classes = np.unique(y_true)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Confusion Matrix')


def plot_roc_curve_ovr(y_true, y_prob, classes):
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = y_true_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "seagreen"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_true_bin[:, class_id],
            y_prob[:, class_id],
            name=f"ROC curve for {classes[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
    )

def evaluate(all_y_true, all_y_pred, all_y_prob, scores):
    avg = (100 * np.mean(scores), 100 * np.std(scores)/np.sqrt(len(scores)))
    print("Average score and standard deviation: (%.2f +- %.3f)%%\n" %avg)

    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))
    print("\n")

    evaluate_cm(all_y_true, all_y_pred)

    all_y_prob = np.array(all_y_prob)
    classes = np.unique(y)
    plot_roc_curve_ovr(all_y_true, all_y_prob, classes)
    plt.show()

def train_model(x, y, k):
    scores = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    kf = KFold(n_splits=k)

    for train, test in kf.split(x, y):
        log_reg = LogisticRegression()
        log_reg.fit(x[train], y[train])
        
        classifier_score = log_reg.score(x[test], y[test])
        scores.append(classifier_score)

        y_pred = log_reg.predict(x[test])
        y_prob = log_reg.predict_proba(x[test])

        all_y_true.extend(y[test])
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

    evaluate(all_y_true, all_y_pred, all_y_prob, scores)

data = load_data()
x, y = preprocess(data)
train_model(x, y, 5)

