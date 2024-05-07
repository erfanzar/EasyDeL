from fjformer import CheckpointManager, float_tensor_to_dtype
from flax.traverse_util import flatten_dict


def match_keywords(string, positives, negatives):
    for positive in positives:
        if positive not in string:
            return False
    for negative in negatives:
        if negative in string:
            return False
    return True


def load_and_convert_checkpoint(path):
    import torch
    _, flax_params = CheckpointManager.load_state_checkpoint(path)
    flax_params = flatten_dict(flax_params['params'], sep='.')
    torch_params = {}
    for key, tensor in flax_params.items():
        if match_keywords(key, ["kernel"], ["norm", 'ln_f']):
            tensor = tensor.T
        tensor = float_tensor_to_dtype(tensor, 'fp16')
        torch_params[key] = torch.from_numpy(tensor)
    return torch_params


def get_pytorch_model_and_config_by_type():
    ...
