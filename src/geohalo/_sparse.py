"""Apply spatial sparse matrices without copying or upcasting a whole batch."""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

_SMALL_GRID_CELLS = 10_000
_BATCH_BYTES = 1024 * 1024


@dataclass(frozen=True)
class GridMatrix:
    matrix: sp.csr_matrix
    source_shape: tuple[int, int]
    cells: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def for_reduction(cls, matrix: sp.csr_matrix, source_shape: tuple[int, int]) -> "GridMatrix":
        """Compact large operators to their referenced columns, keeping coefficient order."""
        if matrix.shape[1] <= _SMALL_GRID_CELLS:
            return cls(matrix, source_shape)
        columns, indices = np.unique(matrix.indices, return_inverse=True)
        # Gathering nearly the whole grid adds indexing work without a useful
        # memory saving. Keep those operators on the bounded full-slice path.
        if columns.size > matrix.shape[1] // 2:
            return cls(matrix, source_shape)
        compact = sp.csr_matrix(
            (matrix.data, indices, matrix.indptr), shape=(matrix.shape[0], columns.size),
        )
        return cls(compact, source_shape, np.divmod(columns, source_shape[1]))

    @cached_property
    def _descending_matrix(self) -> sp.csr_matrix:
        """Remap columns to stored latitude order without reordering coefficients."""
        n_lat, n_lon = self.source_shape
        row, col = np.divmod(self.matrix.indices, n_lon)
        indices = (n_lat - 1 - row) * n_lon + col
        # Share coefficients and row pointers; only column indices change.
        return sp.csr_matrix((self.matrix.data, indices, self.matrix.indptr), shape=self.matrix.shape)

    def apply(self, values: np.ndarray, *, descending: bool) -> np.ndarray:
        """Apply to (..., latitude, longitude), returning (..., matrix rows)."""
        if values.shape[-2:] != self.source_shape:
            raise ValueError(f"expected trailing source shape {self.source_shape}, got {values.shape}")
        batch_shape = values.shape[:-2]
        dtype = np.result_type(values.dtype, self.matrix.dtype)
        out = np.empty((*batch_shape, self.matrix.shape[0]), dtype=dtype)
        matrix = self.matrix
        if self.cells is None and descending:
            matrix = self._descending_matrix

        # Small contiguous grids benefit from batched products. Bound each
        # block so SciPy's contiguous copy/upcast cannot scale with batch size.
        n_source = math.prod(self.source_shape)
        bytes_per_slice = (
            n_source * (values.dtype.itemsize + dtype.itemsize) + matrix.shape[0] * dtype.itemsize
        )
        block_size = max(1, _BATCH_BYTES // max(bytes_per_slice, 1))
        if n_source <= _SMALL_GRID_CELLS and block_size > 1 and values.flags.c_contiguous:
            batch_size = math.prod(batch_shape)
            flat = values.reshape(batch_size, n_source)
            result = out.reshape(batch_size, matrix.shape[0])
            for start in range(0, batch_size, block_size):
                stop = start + block_size
                result[start:stop] = flat[start:stop] @ matrix.T
            return out

        if self.cells is not None:
            rows, cols = self.cells
            if descending:
                rows = self.source_shape[0] - 1 - rows
        for index in np.ndindex(batch_shape):
            # Index the batch before flattening: a strided array must never
            # trigger a reshape copy of the entire input. Gather first when
            # only a subset of source cells contributes to the reduction.
            step = values[index]
            flat = step.ravel() if self.cells is None else step[rows, cols]
            out[index] = matrix @ flat
        return out
