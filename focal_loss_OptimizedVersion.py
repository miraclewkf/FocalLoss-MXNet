import mxnet as mx
import numpy as np

class FocalLossOperator(mx.operator.CustomOp):
    def __init__(self, gamma, alpha):
        super(FocalLossOperator, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, is_train, req, in_data, out_data, aux):
        y = mx.nd.exp(in_data[0] - mx.nd.max_axis(in_data[0], axis=1).reshape((in_data[0].shape[0], 1)))
        y /= mx.nd.sum(y, axis=1).reshape((in_data[0].shape[0], 1))

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y_numpy = out_data[0].asnumpy()
        label_numpy = in_data[1].asnumpy()
        self.pro_truth = mx.nd.array(y_numpy[np.arange(y_numpy.shape[0]), label_numpy.astype(np.int)])

        # i!=j
        pro_truth = (self.pro_truth + 1e-14).reshape((self.pro_truth.shape[0], 1))
        grad = self.alpha * mx.nd.power(1-pro_truth, self.gamma-1) * \
               (self.gamma * (-1 * pro_truth * out_data[0]) * mx.nd.log(pro_truth) + out_data[0] * (1 - pro_truth))

        # i==j
        pro_truth = self.pro_truth + 1e-14

        grad_numpy = grad.asnumpy()
        grad_numpy[np.arange(out_data[0].shape[0]), label_numpy.astype(np.int)] = (self.alpha * mx.nd.power(1 - pro_truth, self.gamma) * (
            self.gamma * pro_truth * mx.nd.log(pro_truth) + pro_truth - 1)).asnumpy()
        grad_numpy /= in_data[1].shape[0]

        self.assign(in_grad[0], req[0], mx.nd.array(grad_numpy))


@mx.operator.register('FocalLoss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, gamma, alpha):
        super(FocalLossProp, self).__init__(need_top_grad=False)

        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def list_arguments(self):
        return ['data', 'labels']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        out_shape = data_shape
        return [data_shape, labels_shape], [out_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLossOperator(self.gamma, self.alpha)
