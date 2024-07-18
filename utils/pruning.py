import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch_pruning as tp

# class QuantProbabilityImportance(tp.importance.Importance):
#     def __call__(self, group, **kwargs):

# class L1ProbabilityPruningMethod(prune.BasePruningMethod):
#     """Prune filters based on L1 norm of the corresponding quantization probability."""
#     PRUNING_TYPE = 'structured'

#     def __init__(self, p, sparsity):
#         self.p = p
#         self.sparsity = sparsity

#     def compute_mask(self, t, default_mask):

def prune_conv_filters_l1(net, sparsity=0.05):
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
    return net