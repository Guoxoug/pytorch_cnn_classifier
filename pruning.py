import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import warnings
# Due to PyTorch's pruning implementation, pruned parameters become computed
# values which are not contiguous in memory. When training a model on the GPU
# with an LSTM this triggers a warning from CuDNN which is printed on every
# CUDA call (i.e. once per minibatch per GPU). We restrict this to only
# printing the warning once to avoid spamming the console whilst still
# informing the user of a potential problem.
warnings.simplefilter("once")


def schedule(
    initial_sparsity, final_sparsity, 
    step, total_steps, start_step=0
):
    """Compute desired sparsity of next step."""
    # s_t = s_f + (s_i - s_f)(1 - (step - step_0) / N)^3

    target_sparsity = (final_sparsity
                       + (initial_sparsity - final_sparsity)
                       * (1 - (step - start_step)/total_steps)**3
                       )

    return target_sparsity

# example has it such that each individual layer has its own sparsity
# I'm going to be lazy and just have a single one


class Pruner:
    def __init__(
        self, targets,
        pruning_steps,
        start_step=0,
        initial_sparsity=0,
        final_sparsity=0.5,
        prune_method=prune.l1_unstructured,
        prune_schedule=schedule
    ):
        self.targets = targets
        # targets need to be in the form (module, name)
        # weights must be direct attributes of module (and not of its children)
        self.total_pruning_steps = pruning_steps
        self.start_step = start_step
        self.steps_taken = 0
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.prune_method = prune_method
        self.prune_schedule = prune_schedule
        self.current_sparsity = initial_sparsity

        if initial_sparsity > 0:
            for module, params in self.targets:
                self.prune_method(module, params, initial_sparsity)

    def prune(self):
        """Prune weights, call like optimiser.step()"""
        if self.steps_taken == self.total_pruning_steps:
            # If already at target sparsity, don't prune anymore
            return
        self.steps_taken += 1
        if self.steps_taken > self.start_step:
            target_sparsity = self.prune_schedule(
                self.initial_sparsity,
                self.final_sparsity,
                self.steps_taken,
                self.total_pruning_steps,
                self.start_step
            )
            delta_sparsity = (target_sparsity - self.current_sparsity)/(1
                                                    - self.current_sparsity)
            self.current_sparsity = target_sparsity
            print(f"current sparsity: {target_sparsity:.3f}", " | ",
                  f"delta sparsity  : {delta_sparsity:.3f}")

            # actual pruning 
            for model, params in self.targets:
                self.prune_method(model, params, delta_sparsity)
  

                
            

    @property
    def done_pruning(self):
        return self.steps_taken == self.total_pruning_steps
