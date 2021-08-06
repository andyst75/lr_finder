# LR Finder for pytorch models

**LR Finder** is designed to select the best initial value of learning rate.

Very easy to use:

```python
from lr_finder import LR_Finder

lr_finder = LR_Finder(model, criterion, torch.optim.Adam, dict(weight_decay=1e-4))
lr_finder.find(x, y, epochs=100, start_lr=0.05, eps_lr=0.5, steps=5)
```

Every iteration change **lr** and calculation loss (by criterion)
```python
lr *= eps_lr
```
