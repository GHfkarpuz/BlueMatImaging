import sys
import os

sys.path.append(os.path.abspath("../../flexBox_cpp/source/build/bindings"))

import numpy as np
import flexBoxPy

class FlexBoxSolver:
    def __init__(self, maxIt=1000, tol=1e-6, verbose=1, checkError=1):
        """
        Python-Wrapper for FlexBox C++ Solver
        """
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

    def add_dual(self, prox_type, alpha, f_list, corresponding_primals, operator_dict=None):
        """
        adds a dual term.
        prox_type:
        alpha: weighting of the terms
        f_list: List of NumPy arrays
        corresponding_primals: List of indices of the primal variables
        operator_dict: optional, if operators are needed
        """
        term = {
            "type": prox_type,
            "factor": alpha,
            "f": f_list,
            "correspondingPrimals": corresponding_primals
        }

        if operator_dict is not None:
            term["operators"] = operator_dict

        self.problem_dict["duals"].append(term)

    def solve(self):
        """
        does the optimization
        """
        x_out, y_out = flexBoxPy.run(self.problem_dict)
        
        self.problem_dict["firstRun"] = False
        return x_out, y_out