import collections
import numpy as np
from typing import Dict
from river import metrics as river_metrics
from scipy import integrate
from metrics import CustomBins
from river import metrics
import warnings

# Suppress the specific deprecation warning
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)


class LOSClassificationReport(river_metrics.ClassificationReport):

    def update(self, y_true, y_pred, w=1):
        y_true_bin = CustomBins.get_bin_custom(y_true)
        y_pred_bin = CustomBins.get_bin_custom(y_pred)
        return super().update(y_true_bin, y_pred_bin, w)


class LOSCohenKappa(river_metrics.CohenKappa):

    def __init__(self, cm=None, weights=None):
        super().__init__(cm)
        self.weights = weights
        self.nbins = CustomBins.nbins

    def update(self, y_true, y_pred):
        y_true_bin = CustomBins.get_bin_custom(y_true)
        y_pred_bin = CustomBins.get_bin_custom(y_pred)
        self.cm.update(y_true_bin, y_pred_bin)
        return self

    def get(self):
        # Number of classes
        classes = self.cm.classes
        n_classes = len(classes)

        # Calculate expected matrix
        expected = np.zeros((n_classes, n_classes))
        sum0 = self.cm.sum_col
        sum1 = self.cm.sum_row
        total_sum0 = self.cm.n_samples

        # Calculate the numerator and denominator for kappa
        numerator = 0
        denominator = 0

        # Fill weight matrix according to the specified weights
        for i in range(n_classes):
            for j in range(n_classes):
                expected[i, j] = (sum0[j] * sum1[i]) / total_sum0
                if self.weights is None:
                    weights = 0 if i == j else 1
                elif self.weights == "linear":
                    weights = abs(i - j)
                else:  # quadratic
                    weights = (i - j)**2
                numerator += weights * self.cm[i][j]
                denominator += weights * expected[i, j]

        # Compute kappa
        if denominator == 0:
            return 0.0
        k = numerator / denominator
        return 1 - k

    def revert(self, y_true, y_pred):
        y_true_bin = CustomBins.get_bin_custom(y_true)
        y_pred_bin = CustomBins.get_bin_custom(y_pred)
        self.cm.revert(y_true_bin, y_pred_bin)
        return self


class PRAUC(river_metrics.ROCAUC):

    def __init__(self, n_thresholds=2, pos_val=True):
        self.pos_val = pos_val
        self.n_thresholds = n_thresholds
        self.thresholds = [i / (n_thresholds - 1) for i in range(n_thresholds)]
        self.thresholds[0] -= 1e-7
        self.thresholds[-1] += 1e-7
        self.cms = [river_metrics.ConfusionMatrix() for _ in range(n_thresholds)]

    def update(self, y_true, y_pred, w=1.0):
        p_true = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        for t, cm in zip(self.thresholds, self.cms):
            cm.update(y_true == self.pos_val, p_true > t, w)

    def get(self):
        tprs = [0] * self.n_thresholds
        fprs = [0] * self.n_thresholds

        def safe_div(a, b):
            if a == 0 and b == 0:
                return 1.0
            try:
                return a / b
            except ZeroDivisionError:
                return 0.0

        for i, cm in enumerate(self.cms):
            tp = cm.true_positives(self.pos_val)
            fp = cm.false_positives(self.pos_val)
            fn = cm.false_negatives(self.pos_val)
            tprs[i] = safe_div(tp, tp + fp)  # precision
            fprs[i] = safe_div(tp, tp + fn)  # recall

        return -integrate.trapezoid(x=fprs, y=tprs)


class MicroROCAUC(river_metrics.ROCAUC):

    def __init__(self, n_thresholds=10, pos_val=True):
        super().__init__(n_thresholds=n_thresholds, pos_val=pos_val)

    def update(self, y_true: dict, y_pred: dict):
        for yt, yp in zip(y_true.values(), y_pred.values()):
            super().update(yt, yp)
        return self

    def revert(self, y_true: dict, y_pred: dict):
        for yt, yp in zip(y_true.values(), y_pred.values()):
            super().revert(yt, yp)
        return self

    def works_with(self, y_true: dict, y_pred: dict):
        return isinstance(y_true, dict) and isinstance(y_pred, dict)


class MacroROCAUC(river_metrics.base.MultiClassMetric):

    def __init__(self, n_thresholds=10, pos_val=True):
        self._n_thresholds = n_thresholds  # TODO! does nothing
        self._pos_val = pos_val  # TODO! does nothing
        self._per_class_rocaucs = collections.defaultdict(river_metrics.ROCAUC)
        self._classes = set()

    def update(self, y_true: dict, y_pred: dict):
        for label_name, y_label in y_pred.items():
            self._classes.add(label_name)
            self._per_class_rocaucs[label_name].update(y_true[label_name], y_label)
        return self

    def get(self):
        # Calculate the macro-average ROC AUC
        return np.mean([rocauc.get() for rocauc in self._per_class_rocaucs.values()])

    def revert(self, y_true, y_pred):
        for label_name, y_label in y_pred.items():
            self._per_class_rocaucs[label_name].revert(y_true[label_name], y_label)
        return self

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, y_true, y_pred):
        return isinstance(y_true, dict) and isinstance(y_pred, dict)


if __name__ == '__main__':

    y_pred = [23, 60, 100, 130, 160, 190, 220, 250, 360, 830]
    y_true = [11.45, 35.07, 59.20, 83.38, 107.48, 131.57, 155.64, 179.66, 254.30, 585.32]

    # ---------------- Comparing Cohen's Kappa using River with sklearn --------------------
    print("--- Comparing Cohen's Kappa ---")
    metric = LOSCohenKappa(weights='linear')
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    """
    from time import time
    start = time()
    for _ in range(100000):
        metric.get()
    end = time()
    print("Time taken new implementation", end - start)
    #
    metric = river_metrics.CohenKappa()
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    from time import time
    start = time()
    for _ in range(100000):
        metric.get()
    end = time()
    print("Time taken legacy", end - start)
    metric.get()
    """
    print("Cohen's Kappa (river)", metric)

    from sklearn.metrics import cohen_kappa_score

    class CustomBins:
        inf = 1e18
        bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14),
                (14, +inf)]
        nbins = len(bins)
        # TODO: whats this
        means = [
            11.450379, 35.070846, 59.206531, 83.382723, 107.487817, 131.579534, 155.643957,
            179.660558, 254.306624, 585.325890
        ]

    def get_bin_custom(x, nbins, one_hot=False):
        for i in range(nbins):
            a = CustomBins.bins[i][0] * 24.0
            b = CustomBins.bins[i][1] * 24.0
            if a <= x < b:
                if one_hot:
                    ret = np.zeros((CustomBins.nbins,))
                    ret[i] = 1
                    return ret
                return i
        return None

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_pred]
    kappa = cohen_kappa_score(y_true_bins, prediction_bins, weights='linear')

    print("Cohen's Kappa (scikit)", kappa)

    from river import metrics
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

    # Binary classification data
    y_true = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    y_pred = [0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.35, 0.0, 0.8, 0.0, 0.2, 0.8]

    # ---------------- Comparing ROC AUC using River with sklearn --------------------
    print("--- Comparing AUC ROC ---")
    metric = metrics.ROCAUC(n_thresholds=20)
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    print("Auc-roc (river)", metric)
    # Compute roc curve
    auc_roc = roc_auc_score(y_true, y_pred)
    print("Auc-roc (scikit)", auc_roc)

    # ---------------- Comparing Precision-Recall AUC using River with sklearn --------------------
    print("--- Comparing AUC PR ---")
    metric = PRAUC()
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    pr_auc = metric.get()
    print("Auc-pr (river)", str(metric))
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, np.array(y_pred).astype(int).tolist())
    # Compute the area under the curve
    pr_auc = auc(recall, precision)
    print("Auc-pr (scikit)", pr_auc)

    # Multi-class classification data
    y_true_multi = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0],
                    [0, 0, 1], [1, 0, 0], [1, 0, 1]]
    y_true_labeled = list()
    for y in y_true_multi:
        y_true_labeled.append({str(i): yt for i, yt in enumerate(y)})
    y_pred_multi = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6], [0.7, 0.2, 0.1],
                    [0.1, 0.6, 0.3], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1], [0.3, 0.5, 0.2],
                    [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]]
    y_pred_labeled = list()
    for y in y_pred_multi:
        y_pred_labeled.append({str(i): yp for i, yp in enumerate(y)})

    # ---------------- Comparing Micro-Macro ROC AUC using River with sklearn --------------------
    print("--- Comparing Micro-Macro ROC AUC ---")
    micro_rocauc = MicroROCAUC()
    macro_rocauc = MacroROCAUC()

    for yt, yp in zip(y_true_labeled, y_pred_labeled):
        micro_rocauc = micro_rocauc.update(yt, yp)
        macro_rocauc = macro_rocauc.update(yt, yp)

    print(f'Micro-average Auc-roc (river): {micro_rocauc.get():.4f}')
    print(f'Macro-average Auc-roc (river): {macro_rocauc.get():.4f}')
    from sklearn.metrics import roc_auc_score

    # Compute micro-average ROC AUC using sklearn
    micro_rocauc_sklearn = roc_auc_score(y_true_multi,
                                         y_pred_multi,
                                         average='micro',
                                         multi_class='ovr')
    print(f'Micro-average auc-roc (sklearn): {micro_rocauc_sklearn:.4f}')

    # Compute macro-average ROC AUC using sklearn
    macro_rocauc_sklearn = roc_auc_score(y_true_multi,
                                         y_pred_multi,
                                         average='macro',
                                         multi_class='ovr')
    print(f'Macro-average auc-roc (sklearn): {macro_rocauc_sklearn:.4f}')
