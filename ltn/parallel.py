import copy
import torch
from torch.func import stack_module_state, functional_call
import ltn
from ltn.core import process_ltn_objects
from ltn import LTNObject
import logging

from functorch import vmap, combine_state_for_ensemble

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)




    # output = predicates[0](data, return_as_tensor=True)
    # output = output.unsqueeze(0)
    

def multi_unary_predicates(batch, data, params, buffers, base_model, as_ltn_objects=False):

    base_model = base_model.to('meta')
    _, vars_, output_shape = process_ltn_objects([data])

    actual_params = dict()
    for k, v in params.items():
        actual_params[k] = v[batch].to(data.value.device)
    
    def call_single_model(params, buffers, data):
        kwargs = {"return_as_tensor": True}
        return functional_call(base_model, (params, buffers), data, kwargs)

    output = torch.vmap(call_single_model, in_dims=(0, 0, None))(actual_params, buffers, data).view(len(batch), *output_shape)


    if as_ltn_objects:
        outputs = []
        for i in range(output.shape[0]):
            outputs.append(LTNObject(output[i], vars_))

        return outputs
    else:
        
        return output.reshape(len(batch), *output_shape)

    

def multi_binary_predicates(batch, data_x, data_y, params, buffers, base_model, as_ltn_objects=True):
    
    base_model = base_model.to('meta')

    _, vars_, output_shape = process_ltn_objects([data_x, data_y])
    
    actual_params = dict()
    for k, v in params.items():
        actual_params[k] = v[batch].to(data_x.value.device)
    
    def call_single_model(params, buffers, data_x, data_y):
        args = (data_x, data_y)
        kwargs = {"return_as_tensor": True}
        return functional_call(base_model, (params, buffers), args, kwargs)

    output = torch.vmap(call_single_model, (0, 0, None, None))(actual_params, buffers, data_x, data_y)

    if not torch.all(torch.where(torch.logical_and(output >= 0., output <= 1.), 1., 0.)):
        raise ValueError("Expected the output of a predicate to be in the range [0., 1.], but got some values "
                         "outside of this range. Check your predicate implementation!")

    
    if as_ltn_objects:
        outputs = []
        for i in range(output.shape[0]):
            outputs.append(LTNObject(output[i], vars_))

        return outputs
    else:
        return output.reshape(len(batch), *output_shape)
