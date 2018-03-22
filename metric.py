import mxnet as mx
import numpy as np

class FocalLoss(mx.metric.EvalMetric):
    def __init__(self, num=None):
        super(FocalLoss, self).__init__('focalloss', num)
        self.eps = 1e-12
        self.gamma = 2
        self.alpha = 0.25

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            self.sum_metric += self.alpha * (-1.0 * np.power(1.0-prob, self.gamma) * np.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]