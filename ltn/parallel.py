import copy
import torch
from torch.func import stack_module_state, functional_call
from ltn.core import process_ltn_objects
from ltn import LTNObject

def multi_unary_predicates(predicates, data, as_ltn_objects=True):
    base_model = copy.deepcopy(predicates[0])
    base_model = base_model.to('meta')

    _, vars_, output_shape = process_ltn_objects([data])
    
    params, buffers = stack_module_state(predicates)

    def call_single_model(params, buffers, data):
        kwargs = {"return_as_tensor": True}
        return functional_call(base_model, (params, buffers), data, kwargs)

    
    output = torch.vmap(call_single_model, (0, 0, None))(params, buffers, data)

    if not torch.all(torch.where(torch.logical_and(output >= 0., output <= 1.), 1., 0.)):
        raise ValueError("Expected the output of a predicate to be in the range [0., 1.], but got some values "
                                 "outside of this range. Check your predicate implementation!")


    if as_ltn_objects:
        outputs = []
        for i in range(output.shape[0]):
            outputs.append(LTNObject(output[i], vars_))

        return outputs
    else:
        return output.reshape(len(predicates), *output_shape)

def multi_binary_predicates(predicates, data_x, data_y, as_ltn_objects=True):
    base_model = copy.deepcopy(predicates[0])
    base_model = base_model.to('meta')

    _, vars_, output_shape = process_ltn_objects([data_x, data_y])
    
    params, buffers = stack_module_state(predicates)

    def call_single_model(params, buffers, data_x, data_y):
        args = (data_x, data_y)
        kwargs = {"return_as_tensor": True}
        return functional_call(base_model, (params, buffers), args, kwargs)

    output = torch.vmap(call_single_model, (0, 0, None, None))(params, buffers, data_x, data_y)

    if not torch.all(torch.where(torch.logical_and(output >= 0., output <= 1.), 1., 0.)):
        raise ValueError("Expected the output of a predicate to be in the range [0., 1.], but got some values "
                         "outside of this range. Check your predicate implementation!")

    
    if as_ltn_objects:
        outputs = []
        for i in range(output.shape[0]):
            outputs.append(LTNObject(output[i], vars_))

        return outputs
    else:
        return output.reshape(len(predicates), *output_shape)
