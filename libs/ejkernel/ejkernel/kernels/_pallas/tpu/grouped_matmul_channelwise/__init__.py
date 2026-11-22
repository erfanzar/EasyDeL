"""Explicit streaming A16 CHANNELWISE Pallas TPU kernel."""

from ._interface import grouped_matmul_channelwise

__all__ = ("grouped_matmul_channelwise",)
