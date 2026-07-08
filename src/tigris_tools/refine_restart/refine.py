from __future__ import annotations

import numpy as np


def refine_cell_centered(array: np.ndarray, factor: int) -> np.ndarray:
    _validate_factor(factor)
    out = array
    for axis in (-3, -2, -1):
        out = np.repeat(out, factor, axis=axis)
    return out


def coarsen_cell_centered(array: np.ndarray, factor: int) -> np.ndarray:
    _validate_factor(factor)
    out = array
    for axis in (-1, -2, -3):
        out = _coarsen_axis_mean(out, axis, factor)
    return out


def refine_face_centered(array: np.ndarray, factor: int, distinguished_axis: int) -> np.ndarray:
    _validate_factor(factor)
    distinguished_axis = _normalize_axis(distinguished_axis, array.ndim)
    out = array
    for axis in range(array.ndim):
        if axis != distinguished_axis:
            out = np.repeat(out, factor, axis=axis)

    old_len = out.shape[distinguished_axis]
    new_len = (old_len - 1) * factor + 1
    new_shape = list(out.shape)
    new_shape[distinguished_axis] = new_len
    refined = np.empty(new_shape, dtype=out.dtype)

    left_slices = [slice(None)] * out.ndim
    right_slices = [slice(None)] * out.ndim
    target_slices = [slice(None)] * out.ndim
    for idx in range(old_len - 1):
        left_slices[distinguished_axis] = idx
        right_slices[distinguished_axis] = idx + 1
        left = out[tuple(left_slices)]
        right = out[tuple(right_slices)]
        for m in range(factor):
            target_slices[distinguished_axis] = idx * factor + m
            weight = m / factor
            refined[tuple(target_slices)] = left + (right - left) * weight
    left_slices[distinguished_axis] = old_len - 1
    target_slices[distinguished_axis] = new_len - 1
    refined[tuple(target_slices)] = out[tuple(left_slices)]
    return refined


def coarsen_face_centered(array: np.ndarray, factor: int, distinguished_axis: int) -> np.ndarray:
    _validate_factor(factor)
    distinguished_axis = _normalize_axis(distinguished_axis, array.ndim)
    out = array
    for axis in sorted((ax for ax in range(array.ndim) if ax != distinguished_axis), reverse=True):
        out = _coarsen_axis_mean(out, axis, factor)
    index = [slice(None)] * out.ndim
    index[distinguished_axis] = slice(None, None, factor)
    return out[tuple(index)]


def refine_face_x1(array: np.ndarray, factor: int) -> np.ndarray:
    return refine_face_centered(array, factor, distinguished_axis=-1)


def refine_face_x2(array: np.ndarray, factor: int) -> np.ndarray:
    return refine_face_centered(array, factor, distinguished_axis=-2)


def refine_face_x3(array: np.ndarray, factor: int) -> np.ndarray:
    return refine_face_centered(array, factor, distinguished_axis=-3)


def coarsen_face_x1(array: np.ndarray, factor: int) -> np.ndarray:
    return coarsen_face_centered(array, factor, distinguished_axis=-1)


def coarsen_face_x2(array: np.ndarray, factor: int) -> np.ndarray:
    return coarsen_face_centered(array, factor, distinguished_axis=-2)


def coarsen_face_x3(array: np.ndarray, factor: int) -> np.ndarray:
    return coarsen_face_centered(array, factor, distinguished_axis=-3)


def _coarsen_axis_mean(array: np.ndarray, axis: int, factor: int) -> np.ndarray:
    axis = _normalize_axis(axis, array.ndim)
    length = array.shape[axis]
    if length % factor != 0:
        raise ValueError(f"axis {axis} length {length} is not divisible by factor {factor}")
    shape = list(array.shape)
    shape[axis : axis + 1] = [length // factor, factor]
    return array.reshape(shape).mean(axis=axis + 1)


def _validate_factor(factor: int) -> None:
    if factor <= 0:
        raise ValueError("factor must be positive")


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} out of range for ndim {ndim}")
    return axis
