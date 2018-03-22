### This is the optimized version of focal loss in MXNet, which is modified from [unsky/focal-loss](https://github.com/unsky/focal-loss) and speed up 30% than Original implement during the training. 

* The use of focal loss is same as [unsky/focal-loss](https://github.com/unsky/focal-loss))

```
from focal_loss_OptimizedVersion import *
label = mx.sym.Variable('focalloss_label')
net = mx.symbol.Custom(data=net, op_type='FocalLoss', labels = label, name='focalloss', alpha=0.25, gamma=2)
```

* Apart from `focal_loss_OptimizedVersion.py`, I alse provide `metric.py` for presenting focal loss value by taking image classification as example:

```
from metric import *
eval_metric = mx.metric.CompositeEvalMetric()
eval_metric.add(FocalLoss())

model = mx.mod.Module(
        context=mx.gpu(0),
        symbol=symbol,
        label_names=('focalloss_label',)
    )

model.fit(...,
	  eval_metric=eval_metric,
	  ...)
```

Attention: The value of alpha and gamma in `metric.py` should be equal to `mx.symbol.Custom(...,alpha, gamma)`
