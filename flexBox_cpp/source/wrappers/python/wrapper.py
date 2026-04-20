import numpy as np
import flexbox_py

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
            "x": [],      # Liste von Primal-Variablen (NumPy Arrays)
            "duals": []   # Liste von Dual-Terms (dicts)
        }

    def add_primal(self, arr: np.ndarray):
        """
        adds primal variable.
        """
        self.problem_dict["x"].append(arr)

    def add_dual(self, prox_type, alpha, f_list, corresponding_primals, operator_dict=None):
        """
        Fügt einen Dual-Term hinzu.
        prox_type: z.B. "L1IsoProxDual"
        alpha: Gewichtung des Terms
        f_list: Liste von NumPy Arrays
        corresponding_primals: Liste von Indizes der Primal-Variablen
        operator_dict: optional, falls Operator benötigt
        """
        term = {
            "type": prox_type,
            "factor": alpha,
            "f": f_list,
            "correspondingPrimals": corresponding_primals
        }

        if operator_dict is not None:
            term["operator"] = operator_dict  # Optional, wenn du flexLinearOperator nutzen willst

        self.problem_dict["duals"].append(term)

    def solve(self):
        """
        does the optimization
        """
        x_out, y_out = flexbox_py.run(self.problem_dict)
        # Nach dem ersten Run nicht mehr als firstRun markieren
        self.problem_dict["firstRun"] = False
        return x_out, y_out