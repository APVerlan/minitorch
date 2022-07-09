import random
import numba
import numpy as np

from .operators import prod
from typing import Any, Optional, Sequence, Union
from numpy import array, float64, ndarray, int64

MAX_DIMS = 32


class TensorData:
    ...


class IndexingError(RuntimeError):
    'Exception raised for indexing errors.'
    pass


def index_to_position(index: np.ndarray[Any, int64], strides: np.ndarray[Any, int64]) -> int:
    '''
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index (array-like): index tuple of ints
        strides (array-like): tensor strides

    Returns:
        int : position in storage
    '''
    res = 0
    for idx, stride in zip(index, strides):
        res += idx * stride

    return int(res)


def to_index(ordinal: int, shape: np.ndarray[Any, int64], out_index: np.ndarray[Any, int64]) -> None:
    '''
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal (int): ordinal position to convert.
        shape (tuple): tensor shape.
        out_index (array): the index corresponding to position.

    Returns:
      None : Fills in `out_index`.

    '''
    pos = int(ordinal)
    for i, s in enumerate(np.flip(shape)):
        out_index[-(i+1)] = pos % s
        pos = pos // s


def broadcast_index(big_index: ndarray[Any, int64],
                    big_shape: ndarray[Any, int64],
                    shape: ndarray[Any, int64],
                    out_index: ndarray[Any, int64]) -> None:
    '''
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index (array-like): multidimensional index of bigger tensor
        big_shape (array-like): tensor shape of bigger tensor
        shape (array-like): tensor shape of smaller tensor
        out_index (array-like): multidimensional index of smaller tensor

    Returns:
        None : Fills in `out_index`.
    '''
    out_index[:] = big_index[len(big_shape) - len(shape):]
    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0


def shape_broadcast(shape1: Sequence[int], shape2: Sequence[int]) -> Sequence[int]:
    '''
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 (tuple) : first shape
        shape2 (tuple) : second shape

    Returns:
        tuple : broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    '''
    final_shape = []

    if len(shape1) > len(shape2):
        shape1, shape2 = shape2, shape1

    shape1 = list(reversed(shape1))
    shape2 = list(reversed(shape2))

    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            if shape1[i] == 1:
                final_shape.append(shape2[i])
            elif shape2[i] == 1:
                final_shape.append(shape1[i])
            else:
                raise IndexingError('')
        else:
            final_shape.append(shape1[i])

    for i in range(len(shape1), len(shape2)):
        final_shape.append(shape2[i])

    return tuple(reversed(final_shape))


def strides_from_shape(shape: Sequence[int]) -> tuple[int]:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    def __init__(self, storage: Sequence[float], shape: Sequence[int], strides: Optional[Sequence[int]] = None) -> None:
        if isinstance(storage, ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), 'Strides must be tuple'
        assert isinstance(shape, tuple), 'Shape must be tuple'

        if len(strides) != len(shape):
            raise IndexingError(f'Len of strides {strides} must match {shape}.')

        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape

        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        '''
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        '''
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a, shape_b):
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, tuple[int]]) -> int:
        if isinstance(index, int):
            index = array([index])
        if isinstance(index, tuple):
            index = array(index)

        # Check for errors
        if index.shape[0] != len(self.shape):
            raise IndexingError(f'Index {index} must be size of {self.shape}.')
        for i, ind in enumerate(index):
            if ind >= self.shape[i]:
                raise IndexingError(f'Index {index} out of range {self.shape}.')
            if ind < 0:
                raise IndexingError(f'Negative indexing for {index} not supported.')

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self):
        lshape = array(self.shape)
        out_index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> tuple[int]:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: Union[Sequence[int], int]) -> float:
        return self._storage[self.index(key)]

    def set(self, key: Union[Sequence[int], int], val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self):
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        '''
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            :class:`TensorData`: a new TensorData with the same storage and a new dimension order.
        '''

        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f'Must give a position to each dimension. Shape: {self.shape} Order: {order}'

        return TensorData(self._storage, tuple(self._shape[list(order)]), tuple(self._strides[list(order)]))

    def to_string(self) -> str:
        s = ''
        for index in self.indices():
            l = ''
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = '\n%s[' % ('\t' * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f'{v:3.2f}'
            l = ''
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += ']'
                else:
                    break
            if l:
                s += l
            else:
                s += ' '
        return s
