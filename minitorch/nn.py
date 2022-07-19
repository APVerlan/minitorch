from minitorch.autodiff import Context
from minitorch.tensor import Tensor
from .fast_ops import FastOps
from .tensor_functions import Function
from . import operators

import numpy as np


def tile(input_tensor: Tensor, kernel_shape: tuple[int, int]) -> tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input_tensor.shape
    kh, kw = kernel_shape

    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw
    
    result_tensor = (
        input_tensor.contiguous()
        .view(batch, channel, new_height, kh, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)
    ).contiguous().view(batch, channel, new_height, new_width, kh * kw)

    # print('sad', 
    #     input_tensor.contiguous().view(batch, channel, new_height, kh, new_width, kw).permute(0, 1, 2, 4, 3, 5)
    # )
    return result_tensor, new_height, new_width




def avgpool2d(input_tensor: Tensor, kernel: tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, _, _ = input_tensor.shape

    tiled, new_height, new_width = tile(input_tensor, kernel)

    # print(tiled.mean(dim=len(tiled.shape) - 1))

    result_tensor = tiled.mean(dim=len(tiled.shape) - 1).view(batch, channel, new_height, new_width)
    # print(result_tensor)
    return result_tensor


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input_tensor: Tensor, dim: int):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input_tensor, dim)
    return out == input_tensor


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, dim: int):
        "Forward of max should be max reduction"
        arg = argmax(input_tensor, dim)
        ctx.save_for_backward(arg)
        
        return 

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        raise NotImplementedError("Need to implement for Task 4.4")


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with randoom positions dropped out
    """
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")
