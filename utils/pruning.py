import random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch_pruning as tp
import logging
from torch.nn.utils import parameters_to_vector, vector_to_parameters

logger = logging.getLogger('main')

# class QuantProbabilityImportance(tp.importance.Importance):
#     def __call__(self, group, **kwargs):

# class L1ProbabilityPruningMethod(prune.BasePruningMethod):
#     """Prune filters based on L1 norm of the corresponding quantization probability."""
#     PRUNING_TYPE = 'structured'

#     def __init__(self, p, sparsity):
#         self.p = p
#         self.sparsity = sparsity

#     def compute_mask(self, t, default_mask):

def mask_network(mask, model):
    # Ensure mask and model are on the same device
    device = next(model.parameters()).device
    mask = mask.to(device)

    k = 0
    for name, param in model.named_parameters():
        t = param.numel()
        param.data = param.data * mask[k:(k + t)].view_as(param.data)
        k += t

    return model

def prune_depgraph(model, example_input, sparsity=0.5):
    """Physically removes filters from the model using torch-pruning.  """
    logger.debug("===== DEPGRAPH pruning =====")
    logger.debug("Sparsity: %s", sparsity)
    DG = tp.DependencyGraph().build_dependency(model, example_input)
    model.train() # reset the model to training mode

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv' in name:
            old_shape = module.weight.shape

            # importance calculation and sorting
            imp = torch.norm(module.weight_p, p=1, dim=(1, 2, 3))
            # imp = module.weight_p.flatten(1).abs().sum(1)
            argimp_sorted = torch.argsort(imp)
            nf_prune = round(sparsity * argimp_sorted.numel())
            idxs_to_prune = argimp_sorted[:nf_prune]

            # prune the filters
            group = DG.get_pruning_group(module, tp.prune_conv_out_channels, idxs=idxs_to_prune)
            group.prune()

            logger.debug('Pruned %s/%s filters  from %s', nf_prune, len(imp), name)
            logger.debug(f'Pruning indices: {sorted(idxs_to_prune.tolist())}')
            # logger.debug('Shape: %s -> %s', old_shape, module.weight.shape)

def make_mask_torch_pruneln(model, sparsity=0.5):
    """Creates a mask for enforcing sparsity in the convolution filters based on global structured pruning.

    This function evaluates the importance of each filter in the model and generates a binary mask to (approximately) enforce a desired sparsity level.
    Elements in the logits tensor with values exceeding a threshold are set to 1 (active),
    and the remaining elements are set to 0 (inactive).

    Args:
        p_net (nn.Module): The network with the quantization probabilities as weights.
        sparsity (float, optional): The target sparsity level (percentage of elements to keep active). Defaults to 0.05 (5%).
    
    Returns:
        torch.Tensor: The generated sparse mask (flattened) for all weights in the network.
    """
    logger.debug("===== TORCH PRUNING =====")
    logger.debug("Sparsity: %s", sparsity)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and 'conv' in name: # only prune convolutional layers
            # note that this isn't pruning the filters but it actually gives good results too
            # prune.ln_structured(module, name='weight', n=0, amount=sparsity, dim=1) 

            prune.ln_structured(module, name='weight', n=1, amount=sparsity, dim=0) 
            prune.l1_unstructured(module, name='bias', amount=0)
            idxs_to_prune = torch.nonzero(torch.all(module.weight_mask == 0, dim=(1,2,3))).squeeze()
            nf_prune = idxs_to_prune.numel()
            prune.remove(module, 'weight') #  apply the mask permanently 
            prune.remove(module, 'bias')
            logger.debug('Pruned %s/%s filters  from %s (shape: %s)', nf_prune, module.out_channels, name, module.weight.shape)
            logger.debug(f'Pruning indices: {idxs_to_prune.tolist()}')
            # logger.debug('Actual zero filters: %s/%s', torch.sum(torch.all(module.weight == 0, dim=(0,2,3))).item(), module.weight.shape[1])

    mask = parameters_to_vector(model.parameters())

    return mask != 0

def make_mask_unstructured(logits, sparsity=0.95):
    """Creates a mask for enforcing sparsity based on a thresholding strategy.

    This function generates a binary mask from a provided tensor (logits) to enforce a desired sparsity level.
    Elements in the logits tensor with values exceeding a threshold are set to 1 (active),
    and the remaining elements are set to 0 (inactive).

    Args:
        logits (torch.Tensor): The input tensor used to generate the mask.
        sparsity (float, optional): The target sparsity level (percentage of elements to keep active). Defaults to 0.05 (5%).

    Returns:
        torch.Tensor: The generated sparse mask (same size as the input logits).
    """
    logger.debug("===== UNSTRUCTURED PRUNING =====")
    
    num_elements = logits.numel()
    num_to_keep = int((1-sparsity) * num_elements)
    logger.debug(f"Number of elements to keep: {num_to_keep}")

    # Get the threshold and top elements
    values, indices = torch.topk(logits.view(-1), num_to_keep, largest=True)
    threshold = values.min()
    logger.debug(f"Threshold value: {threshold}")

    # Identify elements equal to the threshold
    equal_to_threshold = logits.view(-1) == threshold
    num_equal_elements = equal_to_threshold.sum().item()
    logger.debug(f"Number of elements equal to threshold: {num_equal_elements}")

    # Calculate the number of elements to randomly select among equals
    num_to_randomly_select = int(max(
        0, num_to_keep - (values > threshold).sum().item()))
    logger.debug(f"Number of elements to randomly select: {num_to_randomly_select}")

    if num_to_randomly_select and num_equal_elements > num_to_randomly_select:
        logger.warning("Warning: Random selection among elements equal to the threshold is being performed to maintain sparsity.")
        equal_indices = torch.where(equal_to_threshold)[0].tolist()
        selected_indices = random.sample(equal_indices, num_to_randomly_select)
        equal_to_threshold[:] = 0  # Reset all equal elements to zero
        equal_to_threshold[selected_indices] = 1  # Set selected indices to one

    # Create sparse mask
    sparse_mask = (logits.view(-1) > threshold) | equal_to_threshold
    # sparse_mask_prob = mask_prob.view(-1) * sparse_mask
    # hard_quant = torch.round(sparse_mask_prob)

    return sparse_mask

def prune_conv_filters_l1(net, sparsity=0.05):
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
    return net