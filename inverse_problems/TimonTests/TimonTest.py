import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")


import numpy as np
import matplotlib.pyplot as plt

from wrapper import FlexBoxSolver # python wrapper


# test file for calculations for the source reconstruction in EEG


# parameters
weight_data = 1.0
weight_L1 = 0.1


# load
data = np.load("Leadfield.npz")

# extract + fix memory layout
Leadfield = np.ascontiguousarray(data["L"], dtype=np.float64)

# normalize (optional)
Leadfield = Leadfield / np.max(np.abs(Leadfield))

# load potential
potential_0 = np.load("potential_0.npz")["d"]
potential_0 = potential_0 / np.max(np.abs(potential_0))
print("potential_0 norm:", np.linalg.norm(potential_0))
print("Leadfield norm:", np.linalg.norm(Leadfield))

print("shape:", Leadfield.shape)
print("dtype:", Leadfield.dtype)
print("isfortran:", np.isfortran(Leadfield))
print("nnz:", np.count_nonzero(Leadfield))


# initialise solver
solver = FlexBoxSolver(maxIt=10000, tol=1e-6, verbose=1)

# dimensions
numSensors = Leadfield.shape[0]      # 203
numSources = Leadfield.shape[1]      # 182658

# add a primal variable which will hold the weights of the dipoles
solver.add_primal(np.zeros(numSources))


corresponding_primals = [0]

# operator. It has to be extended for the 3D-case.
fullMatrixOp = {
    "type": "fullMatrix",
    "matrix": Leadfield,
}

# add duals
solver.add_dual(
        prox_type = "L2proxDualData",
        alpha = weight_data,
        f_list = [potential_0],
        corresponding_primals = corresponding_primals,
        operator_dict = [fullMatrixOp]
    )


#generate list of gradient operators with one entry for each dimension len(inputdimension)
identityOp = {
        "type": "identityOperator",
        "nPx": numSources
    }

solver.add_dual(
    prox_type="L1AnisoProxDual",
    alpha=weight_L1,
    f_list=[np.zeros(numSensors)],  # no Offset
    corresponding_primals=[0],
    operator_dict=[identityOp]
)


# start the solver
v_out, _ = solver.solve()

v_vec = v_out[0]

print(len(v_vec))
print(v_vec)
np.save("j_potential_0_1000_iterations.npy",v_vec)
