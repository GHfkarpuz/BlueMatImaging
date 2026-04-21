import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")

import numpy as np
import wrapper


def main():
    # create solver
    solver = wrapper.FlexBoxSolver(maxIt=100, tol=1e-6)

    # primal variable as an example
    x0 = np.zeros((10, 10))
    solver.add_primal(x0)

    # as an example
    f = np.zeros((10, 10))

    solver.add_dual(
        prox_type="L1IsoProxDual",
        alpha=1.0,
        f_list=[f],
        corresponding_primals=[0]
    )

    # Solve
    x_out, y_out = solver.solve()

    print("Primal result:", x_out)
    print("Dual result:", y_out)


if __name__ == "__main__":
    main()