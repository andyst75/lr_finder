""" Find optimal LR for pytorch models """

from collections import OrderedDict

import matplotlib.pyplot as plt
from IPython.display import clear_output

from torch import Tensor, no_grad
from torch.nn import Module
from torch.optim import Optimizer

class LR_Finder():
    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer, optimizer_params: dict):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        
        model_dict = model.state_dict()
        base_dict = OrderedDict()
        for item_name, item_tensor in model_dict.items():
            base_dict[item_name] = item_tensor.clone()

        self.base_dict = base_dict
        
        self.result = dict(lr=[], loss=[])
        

    def find(self, x_data: Tensor, y_target: Tensor, epochs: int = 5,
             start_lr: float = 0.1, eps_lr: float = 0.1, steps: int = 5,
             plot=True, figsize=(12, 6)) -> dict:
        lrs = []
        losses = []
        lr = start_lr

        best_lr = lr
        best_loss = 1e18
        
        model = self.model

        for step in range(steps):
            lrs.append(lr)
            
            model_dict = OrderedDict()
            base_dict = self.base_dict
            for item_name, item_tensor in base_dict.items():
                model_dict[item_name] = item_tensor.clone()

            model.load_state_dict(model_dict)
            model.train()

            optimizer = self.optimizer(model.parameters(), lr=lr, **self.optimizer_params)

            for epoch in range(epochs):
                pred = model(x_data)
                loss = self.criterion(pred, y_target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            model.eval()
            with no_grad():
                pred = model(x_data)
                loss = self.criterion(pred, y_target).item()
                losses.append(loss)
                
                if loss < best_loss:
                    best_loss = loss
                    best_lr = lr
                
            if plot:
                clear_output(True)
                plt.figure(figsize=figsize)
                plt.title('LR / Losses')
                print(f'LR={lr:.5f}, loss={loss:.6f}, best loss={best_loss:.6f} with best lr={best_lr:.5f}')
                plt.plot(lrs, losses)
                plt.show()
                
            lr *= eps_lr
                
        self.result['lr'] = lrs
        self.result['loss'] = losses
        
        return self.result
