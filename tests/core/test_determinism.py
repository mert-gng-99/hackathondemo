"""Tests for src.core.determinism."""
from __future__ import annotations

import os

import pyarrow as pa

from src.core import determinism


class TestPinThreads:
    def test_sets_omp_env_var(self):
        original = os.environ.pop("OMP_NUM_THREADS", None)
        try:
            determinism.pin_threads()
            assert os.environ["OMP_NUM_THREADS"] == "1"
        finally:
            if original is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = original

    def test_sets_openblas_env_var(self):
        original = os.environ.pop("OPENBLAS_NUM_THREADS", None)
        try:
            determinism.pin_threads()
            assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        finally:
            if original is None:
                os.environ.pop("OPENBLAS_NUM_THREADS", None)
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = original

    def test_sets_mkl_env_var(self):
        original = os.environ.pop("MKL_NUM_THREADS", None)
        try:
            determinism.pin_threads()
            assert os.environ["MKL_NUM_THREADS"] == "1"
        finally:
            if original is None:
                os.environ.pop("MKL_NUM_THREADS", None)
            else:
                os.environ["MKL_NUM_THREADS"] = original

    def test_pins_pyarrow_cpu_count_to_1(self):
        pa.set_cpu_count(4)
        determinism.pin_threads()
        assert pa.cpu_count() == 1

    def test_pins_pyarrow_io_thread_count_to_1(self):
        pa.set_io_thread_count(4)
        determinism.pin_threads()
        assert pa.io_thread_count() == 1

    def test_does_not_override_existing_env(self):
        """User explicitly setting OMP_NUM_THREADS=2 must win — pin_threads()
        uses os.environ.setdefault so an upstream override is preserved."""
        original = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "2"
        try:
            determinism.pin_threads()
            assert os.environ["OMP_NUM_THREADS"] == "2"
        finally:
            if original is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = original

    def test_idempotent(self):
        determinism.pin_threads()
        determinism.pin_threads()
        assert pa.cpu_count() == 1
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
