# One cycle scheduler for tensorflow 2

This is a repository for one cycle policy for tensorflow 2.0.

The one cycle policy is a learning rate scheduler which allows to train neural networks very fast and in some cases it leads to the phenomenon of ![super convergence](https://arxiv.org/pdf/1708.07120.pdf).

The idea is to train neural network withing one cycle where withing first part of cycle the learning rate grows from small value to high value and then gradually reduces for the small value again. The training with high values learning rate have effect of high regularization, thus in order to compensate this effect it is necessary to decay weight decay at this moment as well as reduce momentum of the optimizer, while when the learning rate decreasing it is necessary to increase weight decay and momentum. The example for the one cycle policy with shift_peak=0.2 (growing lr part 20% from cycle) for the one epoch for the learning rate (min=0.0, max=1.0), momentum (min=0.85, max=0.95) and weight_decay (min=0.0, max=1e-4) on the images below.
<p align="center">
 <img src="https://github.com/Dtananaev/one_cycle_scheduler_tf/blob/main/images/one_cycle.png" width="240"/>
  <img src="https://github.com/Dtananaev/one_cycle_scheduler_tf/blob/main/images/one_cycle_momentum.png" width="240"/>
   <img src="https://github.com/Dtananaev/one_cycle_scheduler_tf/blob/main/images/one_cycle_wd.png" width="240"/>
</p>

# Installation

in order to install just use:
```
pip install one-cycle-tf
```

# Usage

1. In order to use the one cycle policy first you should find the maximal learning rate. It can be done by using  ![learning rate finder](https://github.com/surmenok/keras_lr_finder).

2. Then set the learning rate scheduler like usual scheduler for tensorflow 2.0:

```python
import tensorflow_addons as tfa
from one_cycle_tf import OneCycle
# 1. Set as maximal_learning_rate the values from lr_finder
# 2. Set as initial_learing_rate = maximal_learning_rate / 25.0 (best practice from fast AI)
# 3. The size of cycle in epoch (The one cycle will be withing 10 epoch in example below)
# 4. The shift peak affects the ratio between growing and decaying part of learning rate
# (in the example below shift_peak=0.3, which means the learning rate will grow to 
# maximal_learning_rate withing shift_peak * cycle_size = 0.3 * 10 = 3 epoch)
# 5. final_lr_scale - the scale of value to decay
# (in case if you want to decay more than initial value or less) 
# filal_lr = initial_learning_rate * final_lr_scale
lr_scheduler = OneCycle(initial_learning_rate=0.03/25.0,
                        maximal_learning_rate=0.03,
                        cycle_size=10, 
                        shift_peak=0.3,
                        final_lr_scale=1.0
                        )
# The example of adamw optimizer (adam with decopled weight decay)
# for tensorflow2.0 (you need to pre install tensorflow_addons)
optimizer = tfa.optimizers.AdamW(learning_rate=lr_scheduler)
```
3. If you want to decay momentum of the optimizer (in case of adamw this is beta_1 parameter): 
```python
# This is continuation of the code snippets above.
max_momentum = 0.95
min_momentum = 0.85
momentum_scheduler = OneCycle(initial_learning_rate=max_momentum,
                              maximal_learning_rate=min_momentum,
                              cycle_size=10,
                              shift_peak=0.3,
                              final_lr_scale=1.0
                             )
optimizer._set_hyper("beta_1", lambda: momentum_scheduler(optimizer.iterations))
```
4. If you want to decay weight decay (in case of adamw weight decay is a parameter of the optimizer):

```python
# This is continuation of the code snippets above.
max_wd = 1e-4
min_wd = 0.0
wd_scheduler = OneCycle(initial_learning_rate=max_wd,
                        maximal_learning_rate=min_wd,
                        cycle_size=10, 
                        shift_peak=0.3,
                        final_lr_scale=1.0
                       )
optimizer._set_hyper("weight_decay", lambda: wd_scheduler(optimizer.iterations))
```