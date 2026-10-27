import sys
import os

sys.path.append(os.path.abspath("../../flexBox_cpp/source/build/bindings"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build/bindings")))

import numpy as np
import flexBoxPy

_flexBoxPyCuda = None


def gpu_available():
    """True if the CUDA build of FlexBox (flexBoxPyCuda) can be imported."""
    global _flexBoxPyCuda
    if _flexBoxPyCuda is None:
        try:
            import flexBoxPyCuda
            _flexBoxPyCuda = flexBoxPyCuda
        except ImportError:
            return False
    return True


class FlexBoxSolver:
    def __init__(self, maxIt=1000, tol=1e-6, verbose=1, checkError=1, device="cpu"):
        """
        Python-Wrapper for FlexBox C++ Solver

        device: "cpu" (OpenMP, number of threads via OMP_NUM_THREADS) or "gpu" (CUDA build flexBoxPyCuda)
        """
        if device == "gpu":
            if not gpu_available():
                raise ImportError("flexBoxPyCuda not found: build FlexBox with -DUSE_CUDA=ON")
            self.backend = _flexBoxPyCuda
        elif device == "cpu":
            self.backend = flexBoxPy
        else:
            raise ValueError("device has to be 'cpu' or 'gpu'")

        self.problem_dict = {
            "firstRun": True,
            "params": {
                "maxIt": maxIt,
                "tol": tol,
                "verbose": verbose,
                "checkError": checkError
            },
            "x": [],      # List of primal variables (NumPy Arrays)
            "duals": []   # List of dual terms (dicts)
        }

    def add_primal(self, arr: np.ndarray):
        """
        adds primal variable.
        """
        self.problem_dict["x"].append(arr)

    def add_dual(self, prox_type, alpha, f_list, corresponding_primals, operator_dict=None, min_val=None, max_val=None):
        """
        adds a dual term.
        prox_type:
        alpha: weighting of the terms
        f_list: List of NumPy arrays
        corresponding_primals: List of indices of the primal variables
        operator_dict: optional, if operators are needed
        min_val, max_val: bounds, only needed for prox_type "BoxConstraintProxDual"
        """
        term = {
            "type": prox_type,
            "factor": alpha,
            "f": f_list,
            "correspondingPrimals": corresponding_primals
        }

        if operator_dict is not None:
            term["operators"] = operator_dict

        if prox_type == "BoxConstraintProxDual":
            if min_val is None or max_val is None:
                raise ValueError("BoxConstraintProxDual needs min_val and max_val")
            term["minVal"] = float(min_val)
            term["maxVal"] = float(max_val)

        self.problem_dict["duals"].append(term)

    def solve(self):
        """
        does the optimization
        """
        try:
            x_out, y_out = self.backend.run(self.problem_dict)
        except MemoryError as e:
            if self.backend is flexBoxPy:
                raise
            # not enough GPU memory (e.g. the GPU is used by another process): solve this problem on the CPU
            print(f"FlexBox: GPU out of memory ({e}), this problem is solved on the CPU instead")
            self.backend = flexBoxPy
            self.problem_dict["firstRun"] = True
            x_out, y_out = self.backend.run(self.problem_dict)

        self.problem_dict["firstRun"] = False
        return x_out, y_out