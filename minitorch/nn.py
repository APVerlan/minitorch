from minitorch.autodiff import Context
from minitorch.tensor import Tensor
from .fast_ops import FastOps
from .tensor_functions import Function, rand
from . import operators


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

    result_tensor = tiled.mean(len(tiled.shape) - 1).view(batch, channel, new_height, new_width)
    return result_tensor


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input_tensor: Tensor, dim: int) -> Tensor:
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
    def forward(ctx: Context, input_tensor: Tensor, dim: int) -> Tensor:
        "Forward of max should be max reduction"
        arg = argmax(input_tensor, dim)
        ctx.save_for_backward(arg)
        
        return (arg * input_tensor).sum(dim) / arg.sum(dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        "Backward of max should be argmax (see above)"
        arg = ctx.saved_values

        return grad_output * arg


max = Max.apply


def softmax(input_tensor: Tensor, dim: int) -> Tensor:
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
    return input_tensor.exp() / input_tensor.exp().sum(dim)


def logsoftmax(input_tensor: Tensor, dim: int) -> Tensor:
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
    return input_tensor - input_tensor.exp().sum(dim).log()


def maxpool2d(input_tensor: Tensor, kernel: tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, _, _ = input_tensor.shape
    
    tiled, new_height, new_width = tile(input_tensor, kernel)

    result_tensor = max(tiled, len(tiled.shape) - 1).view(batch, channel, new_height, new_width)
    return result_tensor


def dropout(input_tensor: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with randoom positions dropped out
    """
    if ignore:
        return input_tensor

    ratios = rand(input_tensor.shape)
    return input_tensor * (ratios > rate)
