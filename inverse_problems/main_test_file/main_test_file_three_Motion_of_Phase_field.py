import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/tools")

import cv2
import numpy as np
import glob
import gc
import matplotlib.pyplot as plt
import os
from wrapper import FlexBoxSolver, gpu_available # python wrapper
import errorFunctions as ef
from itertools import product
from skimage.restoration import denoise_nl_means
from image_registration import *

num_iterations = 30

# "gpu": CUDA build of FlexBox (flexBoxPyCuda), "cpu": OpenMP build (threads via OMP_NUM_THREADS, default 8)
device = "gpu" if gpu_available() else "cpu"
#device = "cpu"
print("FlexBox runs on", device)

throughVideoPath = True
syntheticData = False
#video_path = "/data/dust/user/karpuzfa/media4.avi"
#video_path = "rect_slow_a.mp4"
video_path = "/data/dust/user/karpuzfa/media3.avi"
#video_path = "crackCorrection.mp4"
#image_folder = "/home/karpuzfa/CodeFatih/BlueMatImaging/inverse_problems/main_test_file/parameter_search/synthetic_data/rect_slow_a/clean"
image_folder = "/data/dust/user/karpuzfa/resultRectangle/resultRectangleresizeValue1.0epsilon7.01.0L21.01.01.01.01.01.01.01.0L1OpticalFlow0.0TVIso1.0"
image_folder_gt = "MiddleburyDataSetGroundTruth/Hydrangea"
syntheticData_folder = "generatedData"


groundFlowExists = False
if groundFlowExists:
    flo_path = image_folder_gt

joint_approach_with_phase_field = True

best_aae = 10000.0
best_epe = 10000.0
best_aae_epe_combined = [10000.0,10000.0]
best_aae_parameters = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
best_epe_parameters = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
best_aae_epe_combined_parameters = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
name = "Media3Joint3NewPhaseFieldMotionOrder1subtractedPoresNew"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video '{video_path}' does not exist. Stopping execution.")



resize_value = 0.5 #choose a value between 0.0 and 1.0

cut = True
#cut = False
cut_width = [0.5 , 0.0]
cut_height = [0.0 , 0.0]

#start_sec = 21.0
#start_sec = 45.0
#end_sec = 75.0
#end_sec = 76.0
start_sec = 45.0
end_sec = 50.0
n_frames = 7

#start_sec = 44.0
#end_sec = 53.0
#n_frames = 7


"""
start_sec = 30.0
end_sec = 55.0
n_frames = 100
"""
"""
start_sec = 72.0
end_sec = 74.0
n_frames = 7
"""
"""
cut = True
cut_width = [0.45 , 0.0]
cut_height = [0.0 , 0.0]

start_sec = 58.5
end_sec = 60.5
n_frames = 3
"""

#denoiseByAveraging = True
denoiseByAveraging = False
denoiseWithSameFrames = False

denoise_start_sec = 79.0
denoise_end_sec = 104.8
denoise_n_frames = 7


figures_frequence = 5
errors_frequence = 1
image_registration_frequence = 15
"""
# Parameter
weight_u_1_all = [1.0]    # Datenterm u
weight_u_2_all = [0.3]    # space regularizer of u
weight_u_3_all = [1.2]    # time regularizer (second order) of u
weight_u_4_all = [1.0]    # motion term for u

weight_pf_1_all = [0.5]    # regularizer for phase field
weight_pf_2_all = [1.0]    # weight AT terms
weight_pf_3_all = [1.0]    # motion term for phase field
weight_pf_4_all = [0.5]    # weight for the coupling of the motion field and phase field

weight_v_1_all = [1.0]    # motion term for v
weight_v_2_all = [0.2]   # Regularizer of v
weight_v_3_all = [0.5]    # weight for the coupling of the phase field and the motion field
 


epsilon_all = [0.03] # thickness of segment


order = 1
"""
"""
weight_u_1_all = [0.173]
weight_u_2_all = [0.3]       # wirkungslos, TV(u) ist aus
weight_u_3_all = [0.0178]
weight_u_4_all = [1.0]       # wirkungslos, Bewegungsterm in u ist aus
weight_pf_1_all = [0.0105]
weight_pf_2_all = [0.0146]
weight_pf_3_all = [0.00015]
weight_pf_4_all = [4.92]
weight_v_1_all = [9.18]
weight_v_2_all = [89.8]
weight_v_3_all = [0.125]

epsilon_all = [0.0137]

order = 1
"""

weight_u_1_all = [1.0]
weight_u_2_all = [0.4]  
weight_u_3_all = [0.8]
weight_u_4_all = [0.5]
weight_pf_1_all = [1.0]
weight_pf_2_all = [1.0]
weight_pf_3_all = [0.5]
weight_pf_4_all = [0.13]
weight_v_1_all = [1.0]
weight_v_2_all = [0.15]
weight_v_3_all = [0.09]
"""

weight_u_1_all = [1.0]
weight_u_2_all = [0.0,0.2,1.0,5.0]  
weight_u_3_all = [1000000000000000000000.6]
weight_u_4_all = [1000000000000000000000.0]   
weight_pf_1_all = [1.0]
weight_pf_2_all = [1.0]
weight_pf_3_all = [1.0]
weight_pf_4_all = [1.0]
weight_v_1_all = [1.0]
weight_v_2_all = [2.3]
weight_v_3_all = [0.5]
"""
epsilon_all = [4.0]

order = 0

phase_field_exponent = 1.0

# phase field of the pore walls p (static edges): min  weight_pore_1/2*||p - s||^2 + weight_pore_2*||d_t p|| + weight_pore_3*||d_tt p||
# short dips of s in time (moving meniscus) are removed, edges that stay in all frames (pore walls) remain p = 0
pore_time_norm = 1      # 0: off (p = s), 1: L1, 3: L2 for the time derivatives of p
weight_pore_1 = 1.0     # fidelity to the phase field s
weight_pore_2 = 0.0     # first time derivative of p
weight_pore_3 = 0.0     # second time derivative of p
subtractPores = True


def phase_field_masks(phase_field, phase_field_pore):
    """mask of the motion terms (p - s) and weight of the regularisers (1 - p + s), both clipped to [0, 1]
    (p - s < 0 where s = 1 but the time smoothed p < 1 would flip the sign of the motion terms)"""
    if subtractPores:
        mask_motion = np.clip(phase_field_pore - phase_field, 0.0, 1.0) ** phase_field_exponent
        mask_reg = np.clip(1.0 - phase_field_pore + phase_field, 0.0, 1.0) ** phase_field_exponent
        return mask_motion, mask_reg
    else:
        mask_motion = np.clip(np.ones(nPx*len(frames)) - phase_field, 0.0, 1.0) ** phase_field_exponent
        mask_reg = np.clip(phase_field, 0.0, 1.0) ** phase_field_exponent
        return mask_motion, mask_reg



# time derivative in the motion terms of u and s: forward or backward difference depending on the phase field
time_switch = False   # motion terms: keep False (same forward differences as in the v step); True: a*(u_{t+1} - u_t) + (1 - a)*(u_t - u_{t-1}) per pixel, a from the phase field of the previous
                      # iteration: pixel enters the interface in the next frame (s drops) -> backward, otherwise forward


def time_switch_weights(phase_field, nPx, T):
    """weights a of the forward difference (1: forward, 0: backward), from the phase field of the previous iteration"""
    s = np.asarray(phase_field).reshape(T, nPx)            # frames after each other
    a = np.ones_like(s)
    a[:-1] = 1.0 - np.clip(s[:-1] - s[1:], 0.0, 1.0)       # pixel enters the interface (s drops) -> backward difference
    a[0] = 1.0                                             # first frame: no backward difference
    a[-1] = 0.0                                            # last frame: no forward difference
    return a.reshape(-1)


def time_derivative_op(a, dims):
    """forward time difference (a is None) or the phase field dependent mix of forward and backward differences"""
    dt_fwd = {"type": "gradientOperator", "gradType": "forward", "gradDirection": 2, "inputDimension": dims, "isMinus": False}
    if a is None:
        return dt_fwd
    dt_bwd = {"type": "gradientOperator", "gradType": "backward", "gradDirection": 2, "inputDimension": dims, "isMinus": False}
    return {"type": "concatOperator", "operation": "addition", "isMinus": False,
            "A": {"type": "concatOperator", "operation": "composition", "isMinus": False,
                  "A": {"type": "diagonalOperator", "diagonalElements": a}, "B": dt_fwd},
            "B": {"type": "concatOperator", "operation": "composition", "isMinus": False,
                  "A": {"type": "diagonalOperator", "diagonalElements": 1.0 - a}, "B": dt_bwd}}


weight_v_time = 0.1       # > 0: term weight_v_time * ||mask_reg * d_t v|| in the v step (norm as motion_phase_field_coupling_norm);
                          # with time_switch the time difference is chosen from the phase field like for u (forward / backward)
time_switch_vt = True    # term weight_v_time * ||mask_reg * d_t v||: phase field dependent forward / backward difference
time_switch_utt = True   # True: also the term weight_u_3 * ||s * u_tt|| uses phase field dependent stencils:
                          # backward u_t - 2u_{t-1} + u_{t-2} or forward u_{t+2} - 2u_{t+1} + u_t, whichever does not cross a
                          # change of the phase field (without a change: backward as before)


def time_switch_weights_2nd(phase_field, nPx, T):
    """weights b of the forward second difference (1: forward stencil t..t+2, 0: backward stencil t-2..t):
    b = clip(change in the backward stencil - change in the forward stencil, 0, 1), change = |s_{k+1} - s_k|"""
    s = np.asarray(phase_field).reshape(T, nPx)
    c = np.abs(s[1:] - s[:-1])                              # c[k]: change between frame k and k+1
    pad = np.zeros((2, nPx))
    cp = np.concatenate([pad, c, pad])                      # cp[k + 2] = c[k]
    t = np.arange(T)
    back = np.maximum(cp[t + 1], cp[t])                     # c[t-1], c[t-2]
    fwd = np.maximum(cp[t + 2], cp[t + 3])                  # c[t], c[t+1]
    b = np.clip(back - fwd, 0.0, 1.0)
    b[max(T - 2, 0):] = 0.0                                 # no complete forward stencil at the last two frames
    return b.reshape(-1)


def time_second_derivative_op(b, dims):
    """second time difference: backward (b is None, as before) or the phase field dependent mix of forward and backward"""
    bwd = {"type": "gradientOperator", "gradType": "backward", "gradDirection": 2, "inputDimension": dims}
    bwd2 = {"type": "concatOperator", "operation": "composition", "A": bwd, "B": bwd, "isMinus": False}
    if b is None:
        return bwd2
    fwd = {"type": "gradientOperator", "gradType": "forward", "gradDirection": 2, "inputDimension": dims}
    fwd2 = {"type": "concatOperator", "operation": "composition", "A": fwd, "B": fwd, "isMinus": False}
    return {"type": "concatOperator", "operation": "addition", "isMinus": False,
            "A": {"type": "concatOperator", "operation": "composition", "isMinus": False,
                  "A": {"type": "diagonalOperator", "diagonalElements": b}, "B": fwd2},
            "B": {"type": "concatOperator", "operation": "composition", "isMinus": False,
                  "A": {"type": "diagonalOperator", "diagonalElements": 1.0 - b}, "B": bwd2}}


def switched_motion(v, a, nPx):
    """motion that belongs to the chosen difference: v_t for the forward, v_{t-1} for the backward difference"""
    v_prev = np.concatenate([v[:nPx], v[:-nPx]])           # v_{t-1} (frame 0: v_0)
    return a * v + (1.0 - a) * v_prev

# coarse to fine pyramid with warping for the motion field v (only for the optical flow term, motionTerm_v = 1)
v_pyramid = True              # True: v step on an image pyramid with warping instead of one linearisation around v = 0
v_pyramid_levels = 2           # number of coarser levels (each halves the image size); 0: only warping on the full grid
v_pyramid_warps = 2            # re-linearisations (FlexBox solves) per level
v_pyramid_outer = 5            # pyramid only in the outer iterations i < v_pyramid_outer, afterwards one warping step on the full grid
v_pyramid_gradient = "mean"    # gradient of the linearisation: "warped" (grad of img_{j+1}(x + v)) or "mean" (mean with grad img_j)
v_pyramid_median = 0           # median filter (3 or 5) of v after every solve, 0: off
v_pyramid_source = "s"         # "s": optical flow of the phase field (as the v step below), "u": of the image u
v_pyramid_pool = "minmax"      # downsampling of s and mask_reg: "minmax" (minimum over 2x2, thin interfaces survive), "area" (averaging)
v_pyramid_mask_erode = 1       # widen the interface in mask_reg by this many px (3x3 minimum filter): for a 1-2 px thin interface the
                               # motion information sits on its flanks, where mask_reg ~ 1 would pin v to 0; 0: off


def _pool2(a, mode):
    """2x2 minimum or maximum (odd sizes padded by edge replication), same size as cv2.pyrDown"""
    H, W = a.shape
    p = np.pad(a, ((0, H % 2), (0, W % 2)), mode="edge")
    b = p.reshape(p.shape[0] // 2, 2, p.shape[1] // 2, 2)
    return (b.min(axis=(1, 3)) if mode == "min" else b.max(axis=(1, 3))).astype(np.float32)


def _down_low(a):
    """downsampling of the phase field and of the weights that are small at the interfaces: with v_pyramid_pool = "minmax"
    the minimum over 2x2 blocks, so a thin interface (s ~ 0 on 1-2 px) stays an interface on every coarse level
    (averaging mixes it with the background s ~ 1 and it fades after one or two levels)"""
    if v_pyramid_pool == "minmax":
        return _pool2(a, "min")
    return cv2.resize(a.astype(np.float32), ((a.shape[1] + 1) // 2, (a.shape[0] + 1) // 2), interpolation=cv2.INTER_AREA)


def _warp_fwd(img, vy, vx):
    """img(x + v) with bilinear interpolation (vy along the rows = direction 0, vx along the columns = direction 1)"""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img.astype(np.float32), (xx + vx).astype(np.float32), (yy + vy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def v_step_pyramid(img_flat, s_flat, v0_flat, v1_flat, mask_reg, shape, n_t, levels, warps, img_is_s):
    """v step with a coarse to fine pyramid and warping, forward differences as in the v step below
    (v_j is the motion from frame j to j+1, img_{j+1}(x + v_j) = img_j(x)):
        min_v  weight_v_1 ||grad(w) . (v - v0) + w - img_j||            w = img_{j+1}(x + v0), linearised around v0
             + weight_v_2 R(grad v) + weight_v_3 ||mask_reg * v|| + weight_v_time ||mask_reg * d_t v||
    (norms as motionTerm_v_norm, reg_v_norm, motion_phase_field_coupling_norm; d_t v phase field dependent with time_switch_vt).
    img is the phase field s (img_is_s) or the image u. mask_reg is first eroded by v_pyramid_mask_erode px.
    On the coarse levels s and mask_reg are downsampled with
    _down_low (thin interfaces survive), the image u with cv2.pyrDown. Returns [v0, v1] as flat arrays."""
    shape = tuple(int(x) for x in shape)
    to_fr = lambda a, shp: [x.reshape(shp, order="F").astype(np.float32) for x in np.split(np.asarray(a, dtype=np.float32), n_t)]
    flat = lambda fr: np.concatenate([x.flatten(order="F") for x in fr]).astype(np.float32)

    def resize_v(vy, vx, shp):
        # motion in pixels of the new grid: resize the field and scale it with the size ratio
        sy, sx = shp[0] / vy[0].shape[0], shp[1] / vy[0].shape[1]
        interp = cv2.INTER_AREA if sy < 1.0 else cv2.INTER_LINEAR
        return ([cv2.resize(v, (shp[1], shp[0]), interpolation=interp) * sy for v in vy],
                [cv2.resize(v, (shp[1], shp[0]), interpolation=interp) * sx for v in vx])

    I, S, R = [to_fr(img_flat, shape)], [to_fr(s_flat, shape)], [to_fr(mask_reg, shape)]
    if v_pyramid_mask_erode > 0:
        R[0] = [cv2.erode(f, np.ones((3, 3), np.uint8), iterations=int(v_pyramid_mask_erode)) for f in R[0]]
    for _ in range(levels):
        I.append([_down_low(f) if img_is_s else cv2.pyrDown(f) for f in I[-1]])
        S.append([_down_low(f) for f in S[-1]])
        R.append([_down_low(f) for f in R[-1]])
    vy, vx = resize_v(to_fr(v0_flat, shape), to_fr(v1_flat, shape), I[-1][0].shape)

    data_prox = {1: "L1proxDualData", 2: "L2proxDualData"}.get(motionTerm_v_norm)
    norm_prox = {1: "L1IsoProxDual", 2: "L1AnisoProxDual", 3: "L2proxDual"}
    for lev in range(levels, -1, -1):
        if lev < levels:
            vy, vx = resize_v(vy, vx, I[lev][0].shape)
        Il, Sl, Rl = I[lev], S[lev], R[lev]
        H, W = Il[0].shape
        dims, n = [H, W, n_t], H * W * n_t
        for _ in range(warps):
            # data term linearised around the current motion (last frame: no successor, no data)
            d0 = [np.zeros((H, W), np.float32) for _ in range(n_t)]
            d1 = [np.zeros((H, W), np.float32) for _ in range(n_t)]
            f = [np.zeros((H, W), np.float32) for _ in range(n_t)]
            for j in range(n_t - 1):
                w = _warp_fwd(Il[j + 1], vy[j], vx[j])
                gy, gx = np.gradient(w)
                if v_pyramid_gradient == "mean":
                    gy_j, gx_j = np.gradient(Il[j])
                    gy, gx = 0.5 * (gy + gy_j), 0.5 * (gx + gx_j)
                d0[j], d1[j] = gy, gx
                f[j] = gy * vy[j] + gx * vx[j] - (w - Il[j])

            solver = FlexBoxSolver(maxIt=maxIterations_third_solver, tol=tol_third_solver, verbose=verbose_third_solver, device=device)
            solver.add_primal(flat(vy))
            solver.add_primal(flat(vx))
            if data_prox is not None:
                solver.add_dual(prox_type=data_prox, alpha=weight_v_1, f_list=[flat(f)], corresponding_primals=[0, 1],
                                operator_dict=[{"type": "diagonalOperator", "diagonalElements": flat(d0)},
                                               {"type": "diagonalOperator", "diagonalElements": flat(d1)}])
            # regulariser of v (as in the v step below)
            if reg_v_norm != 0:
                op_x = {"type": "gradientOperator", "gradType": "forward", "gradDirection": 0, "inputDimension": dims}
                op_y = {"type": "gradientOperator", "gradType": "forward", "gradDirection": 1, "inputDimension": dims}
                if reg_v_weighted_by_phase_field:
                    op_x = dict(op_x, type="weightedGradientOperator", weights=flat(Rl))
                    op_y = dict(op_y, type="weightedGradientOperator", weights=flat(Rl))
                for p in (0, 1):
                    solver.add_dual(prox_type=norm_prox[reg_v_norm], alpha=weight_v_2, f_list=[np.zeros(n, dtype=np.float32)],
                                    corresponding_primals=[p], operator_dict=[op_x, op_y])
            # coupling ||mask_reg * v|| and ||mask_reg * d_t v|| (as in the v step below)
            if motion_phase_field_coupling_norm != 0:
                diag = {"type": "diagonalOperator", "diagonalElements": flat(Rl)}
                a_l = time_switch_weights(flat(Sl), H * W, n_t) if time_switch_vt else None
                time_op = {"type": "concatOperator", "operation": "composition", "isMinus": False, "A": diag,
                           "B": time_derivative_op(a_l, dims)}
                for p in (0, 1):
                    solver.add_dual(prox_type=norm_prox[motion_phase_field_coupling_norm], alpha=weight_v_3,
                                    f_list=[np.zeros(n, dtype=np.float32)], corresponding_primals=[p], operator_dict=[diag])
                    if weight_v_time > 0:
                        solver.add_dual(prox_type=norm_prox[motion_phase_field_coupling_norm], alpha=weight_v_time,
                                        f_list=[np.zeros(n, dtype=np.float32)], corresponding_primals=[p], operator_dict=[time_op])
            out, _ = solver.solve()
            del solver
            vy, vx = to_fr(out[0], (H, W)), to_fr(out[1], (H, W))
            if v_pyramid_median in (3, 5):
                vy = [cv2.medianBlur(v, v_pyramid_median) for v in vy]
                vx = [cv2.medianBlur(v, v_pyramid_median) for v in vx]
    return [flat(vy), flat(vx)]



image_registration_frequence = 1000000000   # Registrierung aus



phase_field_box_constraint = True # constraint 0 <= phase_field <= 1 inside the second solver
reg_v_weighted_by_phase_field = True # True: TV of v is weighted with the phase field (sum s*|grad v|), so v can jump at the interfaces (s = 0)

tol = 1e-5 #for outer iterations

#parameters for first solver
maxIterations_first_solver = 200
tol_first_solver = 1e-6
verbose_first_solver = 0

#parameters for second solver
maxIterations_second_solver = 200
tol_second_solver = 1e-6
verbose_second_solver = 0

#parameters for third solver
maxIterations_third_solver = 200
tol_third_solver = 1e-6
verbose_third_solver = 0


dataTerm_norm_list                          = ["","L1",             "L2"                    ]
reg_u_norm_list                             = ["","TVIso",          "TVAniso",          "L2"]
reg_u_t_norm_list                           = ["","TVIso",          "TVAniso",          "L2"]
motionTerm_u_list                           = ["","OpticalFlow",    "MassPreservation"      ]
motionTerm_u_norm_list                      = ["","L1Iso",          "L1Aniso",          "L2"]
motionTerm_u_upwind                         = ["",True, False]

reg_phase_field_norm_list                   = ["","TVIso",          "TVAniso",          "L2"]
phase_field_AT_norm_list                    = ["","L1Iso",          "L1Aniso",          "L2"]
motionTerm_phase_field_list                 = ["","OpticalFlow",    "MassPreservation"      ]
motionTerm_phase_field_norm_list            = ["","L1",             "L2"]
phase_field_motion_coupling_norm_list       = ["","TVIso",          "TVAniso",          "L2"]

motionTerm_v_list                           = ["","OpticalFlow",    "MassPreservation"      ]
motionTerm_v_norm_list                      = ["","L1",             "L2"                    ]
reg_v_norm_list                             = ["","TVIso",          "TVAniso",          "L2"]
motion_phase_field_coupling_norm_list       = ["","TVIso",          "TVAniso",          "L2"]

"""
dataTerm_norm_all = [2] # choose from dataTerm_list above
reg_u_norm_all = [1]
reg_u_t_norm_all = [3]
motionTerm_u_all = [1]
motionTerm_u_norm_all = [1]
motionTerm_u_upwind_all = [1]

reg_phase_field_norm_all = [3]          
phase_field_AT_norm_all = [2]           
motionTerm_phase_field_all = [1]        
motionTerm_phase_field_norm_all = [2]
phase_field_motion_coupling_norm_all = [1] 

motionTerm_v_all = [1]
motionTerm_v_norm_all = [1]
reg_v_norm_all = [1]
motion_phase_field_coupling_norm_all = [1] 
"""
"""
dataTerm_norm_all = [2]                     
reg_u_norm_all = [1]                        
reg_u_t_norm_all = [1]                      
motionTerm_u_all = [1]                      
motionTerm_u_norm_all = [1]
motionTerm_u_upwind_all = [1]
reg_phase_field_norm_all = [3]              
phase_field_AT_norm_all = [1]               
motionTerm_phase_field_all = [1]            
motionTerm_phase_field_norm_all = [1]       
phase_field_motion_coupling_norm_all = [3]  
motionTerm_v_all = [1]                      
motionTerm_v_norm_all = [2]                 
reg_v_norm_all = [1]                        
motion_phase_field_coupling_norm_all = [1]


dataTerm_norm_all = [2]                     
reg_u_norm_all = [1]                        
reg_u_t_norm_all = [1]                      
motionTerm_u_all = [1]                      
motionTerm_u_norm_all = [1]
motionTerm_u_upwind_all = [1]
reg_phase_field_norm_all = [3]              
phase_field_AT_norm_all = [1]               
motionTerm_phase_field_all = [1]            
motionTerm_phase_field_norm_all = [1]       
phase_field_motion_coupling_norm_all = [3]  
motionTerm_v_all = [1]                      
motionTerm_v_norm_all = [2]                 
reg_v_norm_all = [1]                        
motion_phase_field_coupling_norm_all = [1]
"""

dataTerm_norm_all = [2]                     
reg_u_norm_all = [3]                        
reg_u_t_norm_all = [1]                      
motionTerm_u_all = [1]                      
motionTerm_u_norm_all = [1]
motionTerm_u_upwind_all = [1]
reg_phase_field_norm_all = [3]              
phase_field_AT_norm_all = [3]               
motionTerm_phase_field_all = [1]            
motionTerm_phase_field_norm_all = [1]       
phase_field_motion_coupling_norm_all = [3]  
motionTerm_v_all = [1]                      
motionTerm_v_norm_all = [1]                 
reg_v_norm_all = [3]                        
motion_phase_field_coupling_norm_all = [3]
"""
dataTerm_norm_all = [2]                     
reg_u_norm_all = [1]                        
reg_u_t_norm_all = [0]                      
motionTerm_u_all = [1]                      
motionTerm_u_norm_all = [1]
motionTerm_u_upwind_all = [1]
reg_phase_field_norm_all = [0]              
phase_field_AT_norm_all = [0]               
motionTerm_phase_field_all = [0]            
motionTerm_phase_field_norm_all = [0]       
phase_field_motion_coupling_norm_all = [0]  
motionTerm_v_all = [1]                      
motionTerm_v_norm_all = [1]                 
reg_v_norm_all = [1]                        
motion_phase_field_coupling_norm_all = [0]
"""

def read_flo_file(file_path):
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError(f"Ungültiges .flo Format (Magic Number falsch) in {file_path}")
        
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        
        # Daten einlesen: abwechselnd vx, vy für jedes Pixel
        data = np.fromfile(f, np.float32, count=2 * w * h)
        flow = data.reshape((h, w, 2))
        
        # In separate vx und vy Maps aufteilen
        return flow[:, :, 0], flow[:, :, 1]


folder_name_main = (f"/data/dust/user/karpuzfa/result{name}")

            
# create the folder in which we will store the figures
if not os.path.exists(folder_name_main):
    os.makedirs(folder_name_main)

file_path = os.path.join(folder_name_main, f"errors{name}.txt")
"""
# write the parameters in the textfile
with open(file_path, "w", encoding="utf-8") as f:
    f.write(f"the parameters are\nweight_1: {weight_1}\nweight_2: {weight_2}\nweight_3: {weight_3}\nweight_pf_1: {weight_pf_1}\nweight_pf_2: {weight_pf_2}")
"""
cap = cv2.VideoCapture(video_path)
if throughVideoPath:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"There are {fps} frames per second")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    frame_indices = np.linspace(start_frame, end_frame, n_frames).astype(int)
    frames = []
    target_set = set(frame_indices)


    idx = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx in target_set:
            if frame is None:
                idx += 1
                pass

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY
            )

            if cut:
                x1 = int(width * cut_width[0])
                x2 = int(width * (1 - cut_width[1]))
                y1 = int(height * cut_height[0])
                y2 = int(height * (1 - cut_height[1]))
                frame_gray = frame_gray[y1:y2, x1:x2]

            if frame_gray.size == 0:
                idx += 1
                pass
            if resize_value != 1.0:
                frame_gray = cv2.resize(frame_gray, None, fx=resize_value, fy=resize_value)
            
            #frame_gray = cv2.fastNlMeansDenoisingMulti(frame_gray, imgToDenoiseIndex=2, temporalWindowSize=5, h=7, templateWindowSize=7, searchWindowSize=21)
            frame_gray = cv2.fastNlMeansDenoising(src=frame_gray, dst=None, h=15, templateWindowSize=7,searchWindowSize=21)
            frame_gray = frame_gray.astype(np.float32)
            frames.append(frame_gray)

        idx += 1
    #frames = cv2.fastNlMeansDenoisingMulti(frames, imgToDenoiseIndex=2, temporalWindowSize=5, h=7, templateWindowSize=7, searchWindowSize=21)
    #for f in frames:
    #    f=f.astype(np.float32)

else:
    if not syntheticData:
        png_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])

        if len(png_files) == 0:
            raise FileNotFoundError(f"no png files found in '{image_folder}'")

        print(f"{len(png_files)} png files were found")

        frames = []
        frames_color = []

        for file_name in png_files:
            file_path = os.path.join(image_folder, file_name)
            
            frame_gray  = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            frame_color = cv2.imread(file_path, cv2.IMREAD_COLOR)
            frame_color = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)
            
            if frame_gray is None:
                print(f"Warning: could not read {file_name}. We skip.")
                continue

            height, width = frame_gray.shape

            if cut:
                x1 = int(width * cut_width[0])
                x2 = int(width * (1 - cut_width[1]))
                y1 = int(height * cut_height[0])
                y2 = int(height * (1 - cut_height[1]))
                frame_gray  = frame_gray[y1:y2, x1:x2]
                frame_color = frame_color[y1:y2, x1:x2]

            frame_gray  = cv2.resize(frame_gray,  None, fx=resize_value, fy=resize_value)
            frame_color = cv2.resize(frame_color, None, fx=resize_value, fy=resize_value)
            
            frame_gray = frame_gray.astype(np.float32)
            
            frames.append(frame_gray)
            frames_color.append(frame_color)

        n_frames = len(frames)
        print(f"succesfully loaded {n_frames} frames.")
    else:
        frames = []

        u_noisy = np.load("generatedData/u_noisy.npy")
        u_clean = np.load("generatedData/u_clean.npy")
        motionFields_gt = np.load("generatedData/w_gt.npy")
        motionfield_0_gt = motionFields_gt[:,:,:,0]
        motionfield_1_gt = motionFields_gt[:,:,:,1]

        u_noisy_list        = list(u_noisy)
        u_clean_list        = list(u_clean)
        motionfield_0_gt_list = list(motionfield_0_gt)
        motionfield_1_gt_list = list(motionfield_1_gt)

        for j in range(len(u_noisy_list)):
            height, width = u_noisy_list[j].shape

            if cut:
                x1 = int(width  * cut_width[0])
                x2 = int(width  * (1 - cut_width[1]))
                y1 = int(height * cut_height[0])
                y2 = int(height * (1 - cut_height[1]))
                u_noisy_list[j]         = u_noisy_list[j][y1:y2, x1:x2]
                u_clean_list[j]         = u_clean_list[j][y1:y2, x1:x2]
                motionfield_0_gt_list[j] = motionfield_0_gt_list[j][y1:y2, x1:x2]
                motionfield_1_gt_list[j] = motionfield_1_gt_list[j][y1:y2, x1:x2]

            if resize_value != 1.0 and resize_value < 1.0:
                u_noisy_list[j]         = cv2.resize(u_noisy_list[j], None, fx=resize_value, fy=resize_value)
                u_clean_list[j]         = cv2.resize(u_clean_list[j], None, fx=resize_value, fy=resize_value)
                motionfield_0_gt_list[j] = cv2.resize(motionfield_0_gt_list[j], None, fx=resize_value, fy=resize_value)
                motionfield_1_gt_list[j] = cv2.resize(motionfield_1_gt_list[j], None, fx=resize_value, fy=resize_value)

                # Bug 2 fix: Zuweisung nicht vergessen
                motionfield_0_gt_list[j] = motionfield_0_gt_list[j].astype(np.float32) * resize_value
                motionfield_1_gt_list[j] = motionfield_1_gt_list[j].astype(np.float32) * resize_value
            
            u_noisy_list[j] = u_noisy_list[j].astype(np.uint8)
            u_noisy_denoised = cv2.fastNlMeansDenoising(src=u_noisy_list[j], dst=None, h=20, templateWindowSize=7,searchWindowSize=21)
            u_noisy_list[j]= u_noisy_denoised.astype(np.float32)

            frames.append(u_noisy_list[j])
            

"""
mask_=frames.copy()
for i in range(len(frames)):
    mask_[i][(frames[i]<85.0) | (frames[i]>127.0)] = 1.0
    mask_[i][(frames[i]>=85.0) & (frames[i]<=127.0)] = 0.0
mask = np.concatenate([mask_[i].flatten(order="F") for i in range(len(mask_))])
"""
print(f"Extracted {len(frames)} frames")

timeIntervall = float(len(frames))

nPx = frames[0].shape[0] * frames[0].shape[1]

# Dimensionen (0=Y/Höhe, 1=X/Breite, 2=Z/Zeit)
inputDimension = [frames[0].shape[0], frames[0].shape[1]]
dimensionsWithTime = [inputDimension[0], inputDimension[1], len(frames)]
print(inputDimension)

# GPU memory check: the solvers need about 700 bytes per voxel (x, y, t) at the peak
# (measured: 6.8 GB for 819 x 410 x 30). If the GPU has not enough free memory, the CPU version is used.
if device == "gpu":
    needed_gb = 700e-9 * nPx * len(frames)
    try:
        import subprocess
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        free_gb = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu_id]).decode().split()[0]) / 1024.0
        print(f"GPU memory: needed about {needed_gb:.1f} GB, free {free_gb:.1f} GB")
        if free_gb < 1.2 * needed_gb:
            print("not enough free GPU memory, FlexBox runs on the CPU")
            device = "cpu"
    except Exception as e:
        print(f"could not check the GPU memory ({e}), FlexBox stays on the GPU")



# save the frames that we start with
for i in range(len(frames)):
    result_1_u = frames[i].reshape(inputDimension, order="F")
    magnitude_u = np.sqrt(result_1_u**2)

    umax1 = np.max(np.abs(result_1_u))
    umax_mag = np.percentile(magnitude_u, 99)

    plt.figure(figsize=(7,7))

    #plt.title(f"raw image sequence")
    #plt.title(f"preDenoised by fastNlMeansDenoising")
    plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(folder_name_main, f"{name},{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    plt.close()
# save the frames that we start with
"""
for i in range(len(frames)):
    plt.figure(figsize=(7,7))
    #plt.imshow(frames[i].astype(np.uint8))
    plt.imshow(frames_color[i] / 255.0)
    plt.axis("off")
    file_path = os.path.join(folder_name_main, f"{name},{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close()
"""
if(denoiseByAveraging):
    if denoiseWithSameFrames:
        denoisedImagebyAveraging = np.zeros(nPx)
        for f in frames:
            denoisedImagebyAveraging += f
        denoisedImagebyAveraging=denoisedImagebyAveraging/(float(len(frames)))
        for i in range(len(frames)):
            frames[i]-=denoisedImagebyAveraging
            frames[i]=cv2.normalize(frames[i], None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #frames[i] = cv2.fastNlMeansDenoising(frames[i], None, 20, 7, 15)
    else:
        denoisedImagebyAveraging = np.zeros(inputDimension, dtype=np.float32)

        denoise_start_frame = int(denoise_start_sec * fps)
        denoise_end_frame = int(denoise_end_sec * fps)
        denoise_start_frame = max(0, denoise_start_frame)
        denoise_end_frame = min(total_frames - 1, denoise_end_frame)

        denoise_frame_indices = np.arange(denoise_start_frame, denoise_end_frame + 1)
        denoise_frames = []
        target_set = set(denoise_frame_indices)


        idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx in target_set:
                if frame is None:
                    idx += 1
                    pass

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if cut:
                    x1 = int(width * cut_width[0])
                    x2 = int(width * (1 - cut_width[1]))
                    y1 = int(height * cut_height[0])
                    y2 = int(height * (1 - cut_height[1]))
                    frame_gray = frame_gray[y1:y2, x1:x2]

                if frame_gray.size == 0:
                    idx += 1
                    pass
                frame_gray = cv2.resize(frame_gray, None, fx=resize_value, fy=resize_value)
                frame_gray = frame_gray.astype(np.float32)
                #frame_gray -= denoisedImage
                #frame_gray = cv2.normalize(frame_gray, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                denoisedImagebyAveraging += frame_gray

            idx += 1

        denoisedImagebyAveraging = denoisedImagebyAveraging/float(len(denoise_frame_indices))

        for i in range(len(frames)):
            frames[i]-=denoisedImagebyAveraging
            frames[i]+=np.ones(frames[i].shape)*255.0
            frames[i]*=0.5
            frames[i] = frames[i].astype(np.uint8)
            frames[i] = cv2.fastNlMeansDenoising(src=frames[i], dst=None, h=40, templateWindowSize=5,searchWindowSize=15)
            frames[i] = frames[i].astype(np.float32)
            #frames[i]+=np.ones(nPx).reshape(inputDimension, order="F")
            #frames[i]=cv2.normalize(frames[i], None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    # save the denoised frames that we start with
    for i in range(len(frames)):
        result_1_u = frames[i].reshape(inputDimension, order="F")
        magnitude_u_new = np.sqrt(result_1_u**2)

        umax1 = np.max(np.abs(result_1_u))
        umax_mag = np.percentile(magnitude_u, 99)

        plt.figure(figsize=(7,7))

        plt.title(f"magnitude of new u_{i+1}")
        plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
        plt.axis("off")

        file_path = os.path.join(folder_name_main, f"subtractedAverageFrames{name},{i}.png")
        plt.savefig(file_path, dpi=200, bbox_inches="tight")

        plt.close()

cap.release()

# Initialisierung der Primalen als C-Contiguous flache Arrays (Inhalt wird im Fortran-Stil verwaltet)
reconstructed_image = np.concatenate([frames[i].flatten(order="F") for i in range(len(frames))])

#motionField_0 = ((np.random.normal(0.0, 0.1, nPx * len(frames)).astype(np.float32))
#motionField_1 = ((np.random.normal(0.0, 0.1, nPx * len(frames)).astype(np.float32))

#motionField_0 = ((np.zeros(nPx * len(frames)).astype(np.float32))
#motionField_1 = ((np.zeros(nPx * len(frames)).astype(np.float32))

# approximate the motionfields by the difference of consecutive frames

motionField_0_list = []
motionField_1_list = []
"""
for j in range(len(frames) - 1):
    diff = frames[j+1] - frames[j]
    motionField_0_list.append(diff.flatten(order="F"))
    motionField_1_list.append(diff.flatten(order="F"))

# last frame is the copy of the second last frame
motionField_0_list.append(motionField_0_list[-1])
motionField_1_list.append(motionField_1_list[-1])

"""
motionField_0_list = [np.zeros(nPx) for i in range(len(frames))]
motionField_1_list = [np.zeros(nPx) for i in range(len(frames))]

motionField_0 = np.concatenate(motionField_0_list)
motionField_1 = np.concatenate(motionField_1_list)

phase_field_list = [np.ones(nPx) for i in range(len(frames))]
#phase_field_list = [np.zeros(nPx) for i in range(len(frames))]
phase_field = np.concatenate(phase_field_list)

phase_field_pore_list = [np.ones(nPx) for i in range(len(frames))]
#phase_field_list = [np.zeros(nPx) for i in range(len(frames))]
phase_field_pore = np.concatenate(phase_field_pore_list)

if not joint_approach_with_phase_field:
    phase_field_list = [np.zeros(nPx) for i in range(len(frames))]
    #phase_field_list = [np.zeros(nPx) for i in range(len(frames))]
    phase_field = np.concatenate(phase_field_list)

    reg_phase_field_norm_all = [0]          
    phase_field_AT_norm_all = [0]           
    motionTerm_phase_field_all = [0]        
    motionTerm_phase_field_norm_all = [0]
    phase_field_motion_coupling_norm_all = [0] 



"""
# save the frames that we start with
for i in range(len(frames)):
    result_1_u = frames[i].reshape(inputDimension, order="F")
    magnitude_u = np.sqrt(result_1_u**2)

    umax1 = np.max(np.abs(result_1_u))
    umax_mag = np.percentile(magnitude_u, 99)

    result_1_v = motionField_0_list[i].reshape(inputDimension, order="F")
    result_2_v = motionField_1_list[i].reshape(inputDimension, order="F")

    magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)
    vmax1 = np.max(np.abs(result_1_v))
    vmax2 = np.max(np.abs(result_2_v))
    vmax_mag = np.percentile(magnitude_v, 99)
    
    print(f"vmax1 for i = {i} is ", vmax1)
    print(f"vmax2 for i = {i} is ", vmax2)

    plt.figure(figsize=(7,7))

    plt.subplot(2,2,1)
    plt.title("magnitude of u")
    plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax1)
    #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")
    
    plt.subplot(2,2,3)
    plt.title("v_x")
    plt.imshow(result_1_v, cmap="seismic", vmin=-vmax1, vmax=vmax1)
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.title("v_y")
    plt.imshow(result_2_v, cmap="seismic", vmin=-vmax2, vmax=vmax2)
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.title("magnitude of v")
    plt.imshow(magnitude_v, cmap="gray", vmin=0, vmax=vmax_mag)
    plt.axis("off")

    file_path = os.path.join(folder_name_main, f"{name},{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    plt.close()
"""

#motionField_0 = ((
#    np.random.normal(0.0, 0.001, nPx * len(frames)).astype(np.float32)
#)
#motionField_1 = ((
#    np.random.normal(0.0, 0.001, nPx * len(frames)).astype(np.float32)
#)
#motionField_0_ = ((np.concatenate([np.array(frames[i+1]-frames[i]).astype(np.float32) for i in range(len(frames)-1)]))
#motionField_0 = np.concatenate([motionField_0_[0],motionField_0_[1],motionField_0_[2],np.zeros(nPx)])

#motionField_1_ = ((np.concatenate([np.array(frames[i+1]-frames[i]).astype(np.float32) for i in range(len(frames)-1)]))
#motionField_1 = np.concatenate([motionField_1_[0],motionField_1_[1],motionField_1_[2],np.zeros(nPx)])


framesBig = np.concatenate([f.flatten(order="F") for f in frames])

#mask = np.ones(nPx*len(frames))
#mask[(framesBig>85.0)&(framesBig<127.0)]=0.0
#mask[(framesBig<130.0)&(framesBig>135.0)]=0.0

"""
# save the frames that we start with
for i in range(len(frames)):
    result_1_u = framesBig[i].reshape(inputDimension, order="F")
    magnitude_u = np.sqrt(result_1_u**2)

    umax1 = np.max(np.abs(result_1_u))
    umax_mag = np.percentile(magnitude_u, 99)

    plt.figure(figsize=(7,7))

    plt.title("magnitude of u")
    plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(folder_name, f"{name},{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    plt.close()
"""


for weight_u_1, weight_u_2, weight_u_3, weight_u_4, weight_pf_1, weight_pf_2, weight_pf_3, weight_pf_4, weight_v_1, weight_v_2, weight_v_3, epsilon, dataTerm_norm, reg_u_norm, reg_u_t_norm, motionTerm_u, motionTerm_u_norm, motionTerm_u_upwind, reg_phase_field_norm, phase_field_AT_norm, motionTerm_phase_field, motionTerm_phase_field_norm, phase_field_motion_coupling_norm, motionTerm_v, motionTerm_v_norm, reg_v_norm, motion_phase_field_coupling_norm in product(weight_u_1_all, weight_u_2_all, weight_u_3_all, weight_u_4_all, weight_pf_1_all, weight_pf_2_all, weight_pf_3_all, weight_pf_4_all, weight_v_1_all, weight_v_2_all, weight_v_3_all, epsilon_all, dataTerm_norm_all, reg_u_norm_all, reg_u_t_norm_all, motionTerm_u_all, motionTerm_u_norm_all, motionTerm_u_upwind_all, reg_phase_field_norm_all, phase_field_AT_norm_all, motionTerm_phase_field_all, motionTerm_phase_field_norm_all, phase_field_motion_coupling_norm_all, motionTerm_v_all, motionTerm_v_norm_all, reg_v_norm_all, motion_phase_field_coupling_norm_all):
    for i in range(num_iterations):
        print(f"it's the {i}. outer iteration")
        mask_motion, mask_reg = phase_field_masks(phase_field, phase_field_pore)
        a_time = time_switch_weights(phase_field, nPx, len(frames)) if time_switch else None
        mf0_t = switched_motion(motionField_0, a_time, nPx) if time_switch else motionField_0
        mf1_t = switched_motion(motionField_1, a_time, nPx) if time_switch else motionField_1
        b_time = time_switch_weights_2nd(phase_field, nPx, len(frames)) if time_switch_utt else None

        print("Typ von weight_pf_2:", type(weight_pf_2), "Inhalt:", weight_pf_2)

        if((i+1) % image_registration_frequence == 0):
            framesBig_list = np.split(framesBig, len(frames))
            frames_2d = [framesBig_list[j].reshape(inputDimension, order="F") 
                        for j in range(len(frames))]
            
            # Bug-Fix: framesBig_ zurückschreiben + Absicherung
            try:
                frames_stabilized, shifts = stabilize_frames_feature_based(frames_2d)
            except Exception as e:
                print(f"Image registration fehlgeschlagen: {e} – überspringe.")
                frames_stabilized = frames_2d  # Fallback: keine Stabilisierung
            
            # Bug-Fix: stabilisierte Frames tatsächlich verwenden
            inputDimension = frames_stabilized[0].shape
            nPx = inputDimension[0] * inputDimension[1]
            dimensionsWithTime = [inputDimension[0], inputDimension[1], len(frames)]
            framesBig = np.concatenate([frames_stabilized[j].flatten(order="F") 
                                        for j in range(len(frames))])
            
            # WICHTIG: auch die anderen Felder an neue Dimension anpassen!
            # (motionField_0, motionField_1, phase_field, reconstructed_image
            #  müssen ebenfalls neu initialisiert oder gecroppt werden)
            print("Image registration abgeschlossen.")
        """
        if((i+1)%image_registration_frequence==0):
            framesBig=np.split(framesBig, len(frames))
            for j in range(len(frames)):
                framesBig[j]=framesBig[j].reshape(inputDimension, order="F")
            framesBig_, shifts = stabilize_frames_feature_based(framesBig)
            inputDimension = framesBig_[0].shape
            dimensionsWithTime = [inputDimension[0], inputDimension[1], len(frames)]
            framesBig = np.concatenate([framesBig[j].flatten(order="F") for j in range(len(frames))])
            print("been here------------------------------------")
        """

        # copy for testing convergence
        motionField_0_old = motionField_0.copy()
        motionField_1_old = motionField_1.copy()
        reconstructed_image_old = reconstructed_image.copy()
        phase_field_old = phase_field.copy()
        
        # first solver, image reconstruction
        solver = FlexBoxSolver(maxIt=maxIterations_first_solver, tol=tol_first_solver, verbose=verbose_first_solver, device=device)
        #if i==0:
        #    solver = FlexBoxSolver(maxIt=5000, tol=tol_first_solver, verbose=verbose_first_solver, device=device)
        solver.add_primal(reconstructed_image)
        corresponding_primals = [0]

        if i>=0:
            # Data Term: (0.5) * ||u - f||^2
            if(dataTerm_norm==0):
                pass
            else:
                data_op = {
                    "type": "identityOperator",
                    "nPx": nPx * len(frames)
                }
                if(dataTerm_norm==1):
                    solver.add_dual(
                        prox_type="L1proxDualData",
                        alpha=weight_u_1 ,
                        f_list=[framesBig],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[data_op]
                    )
                if(dataTerm_norm==2):
                    solver.add_dual(
                        prox_type="L2proxDualData",
                        alpha=weight_u_1 ,
                        f_list=[framesBig],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[data_op]
                    )
            
            # spatial TV of u
            if(reg_u_norm==0):
                pass
            else:
                if joint_approach_with_phase_field:
                    weightedGradientOp_x = {
                        "type": "weightedGradientOperator",
                        "weights": mask_reg,
                        "gradType": "forward",
                        "gradDirection": 0,
                        "inputDimension": dimensionsWithTime
                    }
                    weightedGradientOp_y = {
                        "type": "weightedGradientOperator",
                        "weights": mask_reg,
                        "gradType": "forward",
                        "gradDirection": 1,
                        "inputDimension": dimensionsWithTime
                    }
                    if(reg_u_norm==1):
                        solver.add_dual(
                            prox_type="L1IsoProxDual",
                            alpha=weight_u_2 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[weightedGradientOp_x, weightedGradientOp_y]
                        )
                    if(reg_u_norm==2):
                        solver.add_dual(
                            prox_type="L1AnisoProxDual",
                            alpha=weight_u_2 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[weightedGradientOp_x, weightedGradientOp_y]
                        )
                    if(reg_u_norm==3):
                        solver.add_dual(
                            prox_type="L2proxDual",
                            alpha=weight_u_2 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[weightedGradientOp_x, weightedGradientOp_y]
                        )
                else:
                    grad_x = {
                        "type": "gradientOperator",
                        "gradType": "forward",
                        "gradDirection": 0,
                        "inputDimension": dimensionsWithTime
                    }
                    grad_y = {
                        "type": "gradientOperator",
                        "gradType": "forward",
                        "gradDirection": 1,
                        "inputDimension": dimensionsWithTime
                    }
                    if(reg_u_norm==1):
                        solver.add_dual(
                            prox_type="L1IsoProxDual",
                            alpha=weight_u_2 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[grad_x, grad_y]
                        )
                    if(reg_u_norm==2):
                        solver.add_dual(
                            prox_type="L1AnisoProxDual",
                            alpha=weight_u_2 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[grad_x, grad_y]
                        )
                    if(reg_u_norm==3):
                        solver.add_dual(
                            prox_type="L2proxDual",
                            alpha=weight_u_2 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[grad_x, grad_y]
                        )
            if i>1:
                # time TV second order of u
                if(reg_u_t_norm==0):
                    pass
                else:
                    diagonalOp_PhaseField = {
                        "type": "diagonalOperator",
                        "diagonalElements": mask_reg
                    }
                    grad_t = {
                        "type": "gradientOperator",
                        "gradType": "backward",
                        "gradDirection": 2,
                        "inputDimension": dimensionsWithTime
                    }
                    
                    grad_squared_t = time_second_derivative_op(b_time, dimensionsWithTime)
                    fullOpFirstOrder = {
                        "type": "concatOperator",
                        "operation": "composition",
                        "A": diagonalOp_PhaseField,
                        "B": grad_t,
                        "isMinus": False
                    }
                    fullOpSecondOrder = {
                        "type": "concatOperator",
                        "operation": "composition",
                        "A": diagonalOp_PhaseField,
                        "B": grad_squared_t,
                        "isMinus": False
                    }
                    if(reg_u_t_norm==1):
                        """
                        solver.add_dual(
                            prox_type="L1IsoProxDual",
                            alpha=weight_u_3 ,
                            f_list=[(np.zeros(nPxweight_u_4 * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[fullOpFirstOrder]
                        )
                        """
                        solver.add_dual(
                            prox_type="L1IsoProxDual",
                            alpha=weight_u_3 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[fullOpSecondOrder]
                        )
                    if(reg_u_t_norm==2):
                        """
                        solver.add_dual(
                            prox_type="L1AnisoProxDual",
                            alpha=weight_u_3 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[fullOpFirstOrder]
                        )
                        """
                        solver.add_dual(
                            prox_type="L1AnisoProxDual",
                            alpha=weight_u_3 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[fullOpSecondOrder]
                        )
                    if(reg_u_t_norm==3):
                        """
                        solver.add_dual(
                            prox_type="L2proxDual",
                            alpha=weight_u_3 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[fullOpFirstOrder]
                        )
                        """
                        solver.add_dual(
                            prox_type="L2proxDual",
                            alpha=weight_u_3 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[fullOpSecondOrder]
                        )
                """
                # time TV second order of u
                if(reg_u_t_norm==0):
                    pass
                else:
                    diagonalOp_PhaseField = {
                        "type": "diagonalOperator",
                        "diagonalElements": mask_reg
                    }
                    grad_t = {
                        "type": "gradientOperator",
                        "gradType": "forward",
                        "gradDirection": 2,
                        "inputDimension": dimensionsWithTime
                    }
                    
                    grad_squared_t = {
                        "type": "concatOperator",
                        "operation": "composition",
                        "A": grad_t,
                        "B": grad_t,
                        "isMinus": False
                    }
                    fullOpFirstOrder = {
                        "type": "concatOperator",
                        "operation": "composition",
                        "A": diagonalOp_PhaseField,
                        "B": grad_t,
                        "isMinus": False
                    }
                    fullOpSecondOrder = {
                        "type": "concatOperator",
                        "operation": "composition",
                        "A": diagonalOp_PhaseField,
                        "B": grad_squared_t,
                        "isMinus": False
                    }
                    if(reg_u_t_norm==1):
                        solver.add_dual(
                            prox_type="L1IsoProxDual",
                            alpha=weight_u_3 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[fullOpFirstOrder]
                        )
                    if(reg_u_t_norm==2):
                        solver.add_dual(
                            prox_type="L1AnisoProxDual",
                            alpha=weight_u_3 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[fullOpFirstOrder]
                        )
                    if(reg_u_t_norm==3):
                        solver.add_dual(
                            prox_type="L2proxDual",
                            alpha=weight_u_3 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[fullOpFirstOrder]
                        )
                
                # spatial TV of u
                if(reg_u_norm==0):
                    pass
                else:
                    grad_x = {
                        "type": "gradientOperator",
                        "gradType": "central",
                        "gradDirection": 0,
                        "inputDimension": dimensionsWithTime
                    }
                    grad_y = {
                        "type": "gradientOperator",
                        "gradType": "central",
                        "gradDirection": 1,
                        "inputDimension": dimensionsWithTime
                    }
                    grad_squared_x = {
                        "type": "concatOperator",
                        "operation": "composition",
                        "A": grad_x,
                        "B": grad_x,
                        "isMinus": False
                    }
                    grad_squared_y = {
                        "type": "concatOperator",
                        "operation": "composition",
                        "A": grad_y,
                        "B": grad_y,
                        "isMinus": False
                    }
                    if(reg_u_norm==1):
                        solver.add_dual(
                            prox_type="L1IsoProxDual",
                            alpha=weight_u_2 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[grad_x, grad_y]
                        )
                    
                    if(reg_u_norm==2):
                        solver.add_dual(
                            prox_type="L1AnisoProxDual",
                            alpha=weight_u_2 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[weightedGradientOp1, weightedGradientOp2]
                        )
                    if(reg_u_norm==3):
                        solver.add_dual(
                            prox_type="L2proxDual",
                            alpha=weight_u_2 ,
                            f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                            corresponding_primals=[corresponding_primals[0]],
                            operator_dict=[weightedGradientOp1, weightedGradientOp2]
                        )
                """
                # motion term for u
                if(motionTerm_u==0):
                    pass
                elif(motionTerm_u == 1):
                    #optical flow for u
                    if(motionTerm_u_norm==0):
                        pass
                    else:
                        diagonalOp_PhaseField = {
                            "type": "diagonalOperator",
                            "diagonalElements": mask_motion
                        }
                        upwindGradientOp1 = {
                            "type": "upwindGradientOperator",
                            "motionField": mf0_t,
                            #"gradType": "forward",
                            "gradDirection": 0,
                            "inputDimension": dimensionsWithTime
                        }
                        upwindGradientOp2 = {
                            "type": "upwindGradientOperator",
                            "motionField": mf1_t,
                            #"gradType": "forward",
                            "gradDirection": 1,
                            "inputDimension": dimensionsWithTime
                        }
                        weightedGradientOp1 = {
                            "type": "weightedGradientOperator",
                            "weights": mf0_t,
                            "gradType": "forward",
                            "gradDirection": 0,
                            "inputDimension": dimensionsWithTime
                        }
                        weightedGradientOp2 = {
                            "type": "weightedGradientOperator",
                            "weights": mf1_t,
                            "gradType": "forward",
                            "gradDirection": 1,
                            "inputDimension": dimensionsWithTime
                        }
                        concatGradientOp = {}
                        if(motionTerm_u_upwind==1):
                            concatGradientOp = {
                            "type": "concatOperator",
                            "operation": "addition",
                            "A": upwindGradientOp1,
                            "B": upwindGradientOp2,
                            "isMinus": False
                            }
                        else:
                            concatGradientOp = {
                            "type": "concatOperator",
                            "operation": "addition",
                            "A": weightedGradientOp1,
                            "B": weightedGradientOp2,
                            "isMinus": False
                            }
                        timeDerivativeOp = time_derivative_op(a_time, dimensionsWithTime)
                        concatGradientTimeDerivativeOp = {
                            "type": "concatOperator",
                            "operation": "addition",
                            "A": concatGradientOp,
                            "B": timeDerivativeOp,
                            "isMinus": False
                        }
                        concatOp = {
                            "type": "concatOperator",
                            "operation": "composition",
                            "A": diagonalOp_PhaseField,
                            "B": concatGradientTimeDerivativeOp,
                            "isMinus": False
                        }
                        if(motionTerm_u_norm==1):
                            solver.add_dual(
                                prox_type="L1IsoProxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0]],
                                operator_dict=[concatOp]
                            )
                            """
                            solver.add_dual(
                                prox_type="L1IsoProxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                                operator_dict=[upwindGradientOp1,upwindGradientOp2,timeDerivativeOp]
                            )
                            """
                        if(motionTerm_u_norm==2):
                            solver.add_dual(
                                prox_type="L1AnisoProxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0]],
                                operator_dict=[concatOp]
                            )
                            """
                            solver.add_dual(
                                prox_type="L1AnisoProxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                                operator_dict=[upwindGradientOp1,upwindGradientOp2,timeDerivativeOp]
                            )
                            """
                        if(motionTerm_u_norm==3):
                            solver.add_dual(
                                prox_type="L2proxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0]],
                                operator_dict=[concatOp]
                            )
                elif(motionTerm_u == 2):
                    #mass preservation for u
                    if(motionTerm_u_norm==0):
                        pass
                    else:
                        diagonalOp_PhaseField = {
                            "type": "diagonalOperator",
                            "diagonalElements": (np.ones(nPx*len(frames))-phase_field)
                        }
                        upwindDivergenceOp1 = {
                            "type": "upwindDivergenceOperator",
                            "weights": mf0_t,
                            "gradType": "forward",
                            "gradDirection": 0,
                            "inputDimension": dimensionsWithTime
                        }
                        upwindDivergenceOp2 = {
                            "type": "upwindDivergenceOperator",
                            "weights": mf1_t,
                            "gradType": "forward",
                            "gradDirection": 1,
                            "inputDimension": dimensionsWithTime
                        }

                        weightedDivergenceOp1 = {
                            "type": "weightedDivergenceOperator",
                            "weights": mf0_t,
                            "gradType": "forward",
                            "gradDirection": 0,
                            "inputDimension": dimensionsWithTime
                        }
                        weightedDivergenceOp2 = {
                            "type": "weightedDivergenceOperator",
                            "weights": mf1_t,
                            "gradType": "forward",
                            "gradDirection": 1,
                            "inputDimension": dimensionsWithTime
                        }
                        concatDivergenceOp = {}
                        if(motionTerm_u_upwind==1):
                            concatDivergenceOp = {
                            "type": "concatOperator",
                            "operation": "addition",
                            "A": upwindDivergenceOp1,
                            "B": upwindDivergenceOp2,
                            "isMinus": False
                            }
                        else:
                            concatDivergenceOp = {
                            "type": "concatOperator",
                            "operation": "addition",
                            "A": weightedDivergenceOp1,
                            "B": weightedDivergenceOp2,
                            "isMinus": False
                            }
                        

                        concatUpwindDivergenceOp = {
                            "type": "concatOperator",
                            "operation": "addition",
                            "A": upwindDivergenceOp1,
                            "B": upwindDivergenceOp2,
                            "isMinus": False
                        }
                        timeDerivativeOp = time_derivative_op(a_time, dimensionsWithTime)
                        concatDivergenceTimeDerivativeOp = {
                            "type": "concatOperator",
                            "operation": "addition",
                            "A": concatUpwindDivergenceOp,
                            "B": timeDerivativeOp,
                            "isMinus": False
                        }
                        concatPhasefieldOp = {
                            "type": "concatOperator",
                            "operation": "composition",
                            "A": diagonalOp_PhaseField,
                            "B": concatDivergenceTimeDerivativeOp,
                            "isMinus": False
                        }
                        if(motionTerm_u_norm==1):
                            solver.add_dual(
                                prox_type="L1IsoProxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0]],
                                operator_dict=[concatPhasefieldOp]
                            )
                            """
                            solver.add_dual(
                                prox_type="L1IsoProxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                                operator_dict=[upwindDivergenceOp1,upwindDivergenceOp2,timeDerivativeOp]
                            )
                            """
                        if(motionTerm_u_norm==2):
                            solver.add_dual(
                                prox_type="L1AnisoProxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0]],
                                operator_dict=[concatPhasefieldOp]
                            )
                            """
                            solver.add_dual(
                                prox_type="L1AnisoProxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                                operator_dict=[upwindDivergenceOp1,upwindDivergenceOp2,timeDerivativeOp]
                            )
                            """
                        if(motionTerm_u_norm==3):
                            solver.add_dual(
                                prox_type="L2proxDual",
                                alpha=weight_u_4 ,
                                f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                                corresponding_primals=[corresponding_primals[0]],
                                operator_dict=[concatPhasefieldOp]
                            )
                
            print("before first solver")
            u_out, _ = solver.solve()
            print("after first solver")

            #reorder
            reconstructed_image = np.array(u_out[0], dtype=np.float32)
            
            del solver

        reconstructed_image_frames = np.split(reconstructed_image, len(frames))

        motionField_0_frames = np.split(motionField_0, len(frames))
        motionField_1_frames = np.split(motionField_1, len(frames))
        
        # second solver, phase field
        solver = FlexBoxSolver(maxIt=maxIterations_second_solver, tol=tol_second_solver, verbose=verbose_second_solver, device=device)
        solver.add_primal(phase_field)
        corresponding_primals = [0]

        # Term: ||phase_field *grad(u)||
        if(reg_phase_field_norm==0):
            pass
        else:
            gradientDiagonalOp_x = {
                "type": "gradientDiagonalOperator",
                "weights": reconstructed_image,
                "gradType": "forward",
                "gradDirection": 0,
                "inputDimension": dimensionsWithTime
            }
            gradientDiagonalOp_y = {
                "type": "gradientDiagonalOperator",
                "weights": reconstructed_image,
                "gradType": "forward",
                "gradDirection": 1,
                "inputDimension": dimensionsWithTime
            }
            if(reg_phase_field_norm==1):
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=weight_pf_1 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[gradientDiagonalOp_x, gradientDiagonalOp_y]
                )
            if(reg_phase_field_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_pf_1 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[gradientDiagonalOp_x, gradientDiagonalOp_y]
                )
            if(reg_phase_field_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_pf_1 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[gradientDiagonalOp_x, gradientDiagonalOp_y]
                )

        # AT terms weight*(epsilos*||grad(phase_field)||+frac{1}{4*epsilon}||1-s||)
        if(phase_field_AT_norm==0):
            pass
        else:
            grad_x = {
                "type": "gradientOperator",
                "gradType": "forward",
                "gradDirection": 0,
                "inputDimension": dimensionsWithTime
            }
            grad_y = {
                "type": "gradientOperator",
                "gradType": "forward",
                "gradDirection": 1,
                "inputDimension": dimensionsWithTime
            }
            identityOp = {
                "type": "identityOperator",
                "nPx": nPx * len(frames)
            }
            grad_squared_x = {
                "type": "concatOperator",
                "operation": "composition",
                "A": grad_x,
                "B": grad_x,
                "isMinus": False
            }
            grad_squared_y = {
                "type": "concatOperator",
                "operation": "composition",
                "A": grad_y,
                "B": grad_y,
                "isMinus": False
            }
            if(phase_field_AT_norm==1):
                if(order==0):
                    solver.add_dual(
                        prox_type="L1IsoProxDual",
                        alpha=((weight_pf_2*np.sqrt(epsilon)) / 2.0),
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[grad_x, grad_y]
                    )
                    solver.add_dual(
                        prox_type="L1proxDualData",
                        alpha=(weight_pf_2/(2*np.sqrt(epsilon))) ,
                        f_list=[(np.ones(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[identityOp]
                    )
                if(order==1):
                    solver.add_dual(
                        prox_type="L1IsoProxDual",
                        alpha=((weight_pf_2*np.sqrt(epsilon*epsilon*epsilon))),
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[grad_squared_x, grad_squared_y]
                    )
                    solver.add_dual(
                        prox_type="L1proxDualData",
                        alpha=(weight_pf_2/(np.sqrt(epsilon))) ,
                        f_list=[(np.ones(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[identityOp]
                    )
            if(phase_field_AT_norm==2):
                if(order==0):
                    solver.add_dual(
                        prox_type="L1AnisoProxDual",
                        alpha=((weight_pf_2*np.sqrt(epsilon)) / 2.0),
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[grad_x, grad_y]
                    )
                    solver.add_dual(
                        prox_type="L1proxDualData",
                        alpha=(weight_pf_2/(np.sqrt(epsilon))) ,
                        f_list=[(np.ones(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[identityOp]
                    )
                if(order==1):
                    solver.add_dual(
                        prox_type="L1AnisoProxDual",
                        alpha=((weight_pf_2*np.sqrt(epsilon*epsilon*epsilon))),
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[grad_squared_x, grad_squared_y]
                    )
                    solver.add_dual(
                        prox_type="L1proxDualData",
                        alpha=(weight_pf_2/(np.sqrt(epsilon))) ,
                        f_list=[(np.ones(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[identityOp]
                    )
            if(phase_field_AT_norm==3):
                if(order==0):
                    solver.add_dual(
                        prox_type="L2proxDual",
                        alpha=((weight_pf_2*np.sqrt(epsilon)) / 2.0),
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[grad_x, grad_y]
                    )
                    solver.add_dual(
                        prox_type="L2proxDualData",
                        alpha=(weight_pf_2/(np.sqrt(epsilon*epsilon))) ,
                        f_list=[(np.ones(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[identityOp]
                    )
                if(order==1):
                    solver.add_dual(
                        prox_type="L2proxDual",
                        alpha=((weight_pf_2*np.sqrt(epsilon*epsilon*epsilon))),
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[grad_squared_x, grad_squared_y]
                    )
                    solver.add_dual(
                        prox_type="L2proxDualData",
                        alpha=(weight_pf_2/(np.sqrt(epsilon))) ,
                        f_list=[(np.ones(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[identityOp]
                    )
        #calculate the optical flow and mass preservation data term for the motion term 
        
        # motion term for s
        if(motionTerm_phase_field==0):
            pass
        elif(motionTerm_phase_field == 1):
            #optical flow for u
            if(motionTerm_phase_field_norm==0):
                pass
            else:
                upwindGradientOp1 = {
                    "type": "upwindGradientOperator",
                    "motionField": mf0_t,
                    #"gradType": "forward",
                    "gradDirection": 0,
                    "inputDimension": dimensionsWithTime
                }
                upwindGradientOp2 = {
                    "type": "upwindGradientOperator",
                    "motionField": mf1_t,
                    #"gradType": "forward",
                    "gradDirection": 1,
                    "inputDimension": dimensionsWithTime
                }
                weightedGradientOp1 = {
                    "type": "weightedGradientOperator",
                    "weights": mf0_t,
                    "gradType": "forward",
                    "gradDirection": 0,
                    "inputDimension": dimensionsWithTime
                }
                weightedGradientOp2 = {
                    "type": "weightedGradientOperator",
                    "weights": mf1_t,
                    "gradType": "forward",
                    "gradDirection": 1,
                    "inputDimension": dimensionsWithTime
                }
                concatGradientOp = {}
                if(motionTerm_u_upwind==1):
                    concatGradientOp = {
                        "type": "concatOperator",
                        "operation": "addition",
                        "A": upwindGradientOp1,
                        "B": upwindGradientOp2,
                        "isMinus": False
                    }
                else:
                    concatGradientOp = {
                        "type": "concatOperator",
                        "operation": "addition",
                        "A": weightedGradientOp1,
                        "B": weightedGradientOp2,
                        "isMinus": False
                    }
                timeDerivativeOp = time_derivative_op(a_time, dimensionsWithTime)
                concatGradientTimeDerivativeOp = {
                    "type": "concatOperator",
                    "operation": "addition",
                    "A": concatGradientOp,
                    "B": timeDerivativeOp,
                    "isMinus": False
                }
                if(motionTerm_phase_field_norm==1):
                    solver.add_dual(
                        prox_type="L1IsoProxDual",
                        alpha=weight_pf_3 ,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[concatGradientTimeDerivativeOp]
                    )
                    """
                    solver.add_dual(
                        prox_type="L1IsoProxDual",
                        alpha=weight_pf_3 ,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                        operator_dict=[upwindGradientOp1,upwindGradientOp2,timeDerivativeOp]
                    )
                    """
                if(motionTerm_phase_field_norm==2):
                    solver.add_dual(
                        prox_type="L1AnisoProxDual",
                        alpha=weight_pf_3 ,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[concatGradientTimeDerivativeOp]
                    )
                    """
                    solver.add_dual(
                        prox_type="L1AnisoProxDual",
                        alpha=weight_pf_3 ,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                        operator_dict=[upwindGradientOp1,upwindGradientOp2,timeDerivativeOp]
                    )
                    """
                if(motionTerm_phase_field_norm==3):
                    solver.add_dual(
                        prox_type="L2proxDual",
                        alpha=weight_pf_3 ,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[concatGradientTimeDerivativeOp]
                    )
        if(phase_field_motion_coupling_norm==0):
            pass
        else:
            diagonalOp_x = {
                "type": "diagonalOperator",
                "diagonalElements": motionField_0
            }
            diagonalOp_y = {
                "type": "diagonalOperator",
                "diagonalElements": motionField_1
            }
            if(phase_field_motion_coupling_norm==1):
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=weight_pf_4 ,
                    f_list=[np.zeros(nPx*len(frames))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[diagonalOp_x,diagonalOp_y]
                )
            elif(phase_field_motion_coupling_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_pf_4 ,
                    f_list=[np.zeros(nPx*len(frames))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[diagonalOp_x,diagonalOp_y]
                )
            if(phase_field_motion_coupling_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_pf_4 ,
                    f_list=[np.zeros(nPx*len(frames))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[diagonalOp_x,diagonalOp_y]
                )

        # box constraint 0 <= phase_field <= 1
        if phase_field_box_constraint:
            identityOp_box = {
                "type": "identityOperator",
                "nPx": nPx * len(frames)
            }
            solver.add_dual(
                prox_type="BoxConstraintProxDual",
                alpha=1.0,
                f_list=[np.zeros(nPx*len(frames))],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[identityOp_box],
                min_val=0.0,
                max_val=1.0
            )

        print("before second solver")
        phase_field_out, _ = solver.solve()
        print("after second solver")


        phase_field = np.array(phase_field_out[0], dtype=np.float32)
        phase_field = np.clip(phase_field, 0.0, 1.0)

        # second,second solver, phase field of the pore walls, computed only on the first, middle and last frame:
        # min weight_pore_1/2*||p - s||^2 + weight_pore_2*||d_t p|| + weight_pore_3*||d_tt p||,  0 <= p <= 1
        # (between these key frames the meniscus has moved far, so it is not an edge in all of them, the pore walls are)
        pore_key_frames = sorted(set([0, len(frames) // 2, len(frames) - 1]))
        n_key = len(pore_key_frames)
        phase_field_key = np.concatenate([phase_field[k * nPx:(k + 1) * nPx] for k in pore_key_frames])
        dimensionsWithTime_key = [inputDimension[0], inputDimension[1], n_key]
        phase_field_pore_key = phase_field_key.copy()

        solver = FlexBoxSolver(maxIt=maxIterations_second_solver, tol=tol_second_solver, verbose=verbose_second_solver, device=device)
        solver.add_primal(phase_field_pore_key)
        corresponding_primals = [0]

        # fidelity to the phase field s of the key frames
        solver.add_dual(
            prox_type="L2proxDualData",
            alpha=weight_pore_1 ,
            f_list=[phase_field_key],
            corresponding_primals=[corresponding_primals[0]],
            operator_dict=[{"type": "identityOperator", "nPx": nPx * n_key}]
        )

        # time regularisation of p over the key frames
        if(pore_time_norm != 0):
            grad_t = {
                "type": "gradientOperator",
                "gradType": "forward",
                "gradDirection": 2,
                "inputDimension": dimensionsWithTime_key
            }
            grad_squared_t = {
                "type": "concatOperator",
                "operation": "composition",
                "A": grad_t,
                "B": grad_t,
                "isMinus": False
            }
            pore_prox = "L1IsoProxDual" if pore_time_norm == 1 else "L2proxDual"
            for weight_pore, op in ((weight_pore_2, grad_t), (weight_pore_3, grad_squared_t)):
                if weight_pore > 0:
                    solver.add_dual(
                        prox_type=pore_prox,
                        alpha=weight_pore_2 ,
                        f_list=[(np.zeros(nPx * n_key, dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[op]
                    )

        # box constraint 0 <= phase_field_pore <= 1
        if phase_field_box_constraint:
            identityOp_box = {
                "type": "identityOperator",
                "nPx": nPx * n_key
            }
            solver.add_dual(
                prox_type="BoxConstraintProxDual",
                alpha=weight_pore_3,
                f_list=[np.zeros(nPx*n_key)],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[identityOp_box],
                min_val=0.0,
                max_val=1.0
            )

        print("before second,second solver")
        phase_field_pore_out, _ = solver.solve()
        print("after second,second solver")

        b_time = time_switch_weights_2nd(phase_field, nPx, len(frames)) if time_switch_utt else None

        phase_field_pore_key = np.clip(np.array(phase_field_pore_out[0], dtype=np.float32), 0.0, 1.0)
        # one frame for the whole sequence: a pixel is a pore wall (p = 0) only if it is an edge in all key frames,
        # a meniscus that is an edge in only one key frame gets p = 1 (maximum over the key frames)
        phase_field_pore_frame = phase_field_pore_key.reshape(n_key, nPx).max(axis=0)
        phase_field_pore = np.tile(phase_field_pore_frame, len(frames))

        phase_field_pore_frames = np.split(phase_field_pore, len(frames))
        phase_field_frames = np.split(phase_field, len(frames))
        mask_motion, mask_reg = phase_field_masks(phase_field, phase_field_pore)  # new s and p for the v step

        # third solver for v
        solver = FlexBoxSolver(maxIt=maxIterations_third_solver, tol=tol_third_solver, verbose=verbose_third_solver, device=device)
        solver.add_primal(motionField_0)
        solver.add_primal(motionField_1)
        corresponding_primals = [0, 1]

        # calcualtion of -u_t as the data for the motion term
        """
        negUT_list = [np.zeros(nPx, dtype=np.float32) for j in range(len(frames))]
        for j in range(len(frames) - 1):
            negUT_list[j] = -(reconstructed_image_frames[j+1] - reconstructed_image_frames[j]).flatten(order="F")
        #for j in range(len(frames)):
            #negUT_list[j] *= (np.ones(nPx, dtype=np.float32)-phase_field_frames[j])
        
        #negUT_list[len(frames) - 1] = negUT_list[len(frames) - 2]
        negUT = np.concatenate(negUT_list)
        """
        negPFT_list = [np.zeros(nPx, dtype=np.float32) for j in range(len(frames))]
        for j in range(0, len(frames)-1):  # startet bei 1, nicht 0
            negPFT_list[j] = -(phase_field_frames[j+1] - phase_field_frames[j]).flatten(order="F")
        negPFT = np.concatenate(negPFT_list)

        # motion term for v
        if(motionTerm_v==0):
            pass
        elif(motionTerm_v == 1):
            #optical flow for v
            if(motionTerm_v_norm==0):
                pass
            else:
                opticalFlowOp1 = {
                    "type": "opticalFlowOp",
                    "image": phase_field,
                    "direction": 0,
                    "inputDimension": dimensionsWithTime
                }
                opticalFlowOp2 = {
                    "type": "opticalFlowOp",
                    "image": phase_field,
                    "direction": 1,
                    "inputDimension": dimensionsWithTime
                }
                concatMassPreservationOp = {
                    "type": "concatOperator",
                    "operation": "addition",
                    "A": opticalFlowOp1,
                    "B": opticalFlowOp2,
                    "isMinus": False
                }
                if(motionTerm_v_norm==1):
                    solver.add_dual(
                        prox_type="L1proxDualData",
                        alpha=weight_v_1 ,
                        f_list=[negPFT],
                        corresponding_primals=[corresponding_primals[0], corresponding_primals[1]],
                        operator_dict=[opticalFlowOp1,opticalFlowOp2]
                        #operator_dict=[concatMassPreservationOp]
                    )
                if(motionTerm_v_norm==2):
                    solver.add_dual(
                        prox_type="L2proxDualData",
                        alpha=weight_v_1 ,
                        f_list=[negPFT],
                        corresponding_primals=[corresponding_primals[0], corresponding_primals[1]],
                        operator_dict=[opticalFlowOp1,opticalFlowOp2]
                        #operator_dict=[concatMassPreservationOp]
                    )
        elif(motionTerm_v == 2):
            #mass preservation for v
            if(motionTerm_v_norm==0):
                pass
            else:
                diagonalOp = {
                    "type": "diagonalOperator",
                    "diagonalElements": (np.ones(nPx*len(frames))-phase_field)
                }
                massPreservationOp1 = {
                    "type": "massPreservationOp",
                    "image": reconstructed_image,
                    "direction": 0,
                    "inputDimension": dimensionsWithTime
                }
                massPreservationOp2 = {
                    "type": "massPreservationOp",
                    "image": reconstructed_image,
                    "direction": 1,
                    "inputDimension": dimensionsWithTime
                }
                concatMassPreservationOp = {
                    "type": "concatOperator",
                    "operation": "addition",
                    "A": massPreservationOp1,
                    "B": massPreservationOp2,
                    "isMinus": False
                }
                concatWithPhaseFieldOp1 ={
                    "type": "concatOperator",
                    "operation": "composition",
                    "A": diagonalOp,
                    "B": massPreservationOp1,
                    "isMinus": False
                }
                concatWithPhaseFieldOp2 ={
                    "type": "concatOperator",
                    "operation": "composition",
                    "A": diagonalOp,
                    "B": massPreservationOp2,
                    "isMinus": False
                }
                if(motionTerm_v_norm==1):
                    solver.add_dual(
                        prox_type="L1proxDualData",
                        alpha=weight_v_1 ,
                        f_list=[(np.ones(nPx*len(frames))-phase_field)*negUT],
                        corresponding_primals=[corresponding_primals[0], corresponding_primals[1]],
                        operator_dict=[concatWithPhaseFieldOp1,concatWithPhaseFieldOp2]
                        #operator_dict=[concatMassPreservationOp]
                    )
                if(motionTerm_v_norm==2):
                    solver.add_dual(
                        prox_type="L2proxDualData",
                        alpha=weight_v_1 ,
                        f_list=[(np.ones(nPx*len(frames))-phase_field)*negUT],
                        corresponding_primals=[corresponding_primals[0], corresponding_primals[1]],
                        operator_dict=[concatWithPhaseFieldOp1,concatWithPhaseFieldOp2]
                        #operator_dict=[concatMassPreservationOp]
                    )
        # TV for v
        if(reg_v_norm==0):
            pass
        else:
            grad_x = {
                "type": "gradientOperator",
                "gradType": "forward",
                "gradDirection": 0,
                "inputDimension": dimensionsWithTime
            }
            grad_y = {
                "type": "gradientOperator",
                "gradType": "forward",
                "gradDirection": 1,
                "inputDimension": dimensionsWithTime
            }
            if reg_v_weighted_by_phase_field:
                # s * grad v: no smoothing of v across the interfaces (s = 0), full smoothing elsewhere (s = 1)
                grad_x = dict(grad_x, type="weightedGradientOperator", weights=mask_reg)
                grad_y = dict(grad_y, type="weightedGradientOperator", weights=mask_reg)
            if(reg_v_norm==1):
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=weight_v_2 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[grad_x,grad_y]
                )
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=weight_v_2 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[grad_x,grad_y]
                )
            if(reg_v_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_v_2 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[grad_x,grad_y]
                )
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_v_2 ,
                    #alpha=5.0 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[grad_x,grad_y]
                )
            if(reg_v_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_v_2 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[grad_x,grad_y]
                )
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_v_2 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[grad_x,grad_y]
                )
        """
        if(motion_phase_field_coupling_norm==0):
            pass
        else:
            diagonal_PhasefieldOp = {
                "type": "diagonalOperator",
                "diagonalElements": phase_field
            }
            grad_t = {
                    "type": "gradientOperator",
                    "gradType": "backward",
                    "gradDirection": 2,
                    "inputDimension": dimensionsWithTime
                }
            grad_squared_t = {
                "type": "concatOperator",
                "operation": "composition",
                "A": grad_t,
                "B": grad_t,
                "isMinus": False
            }
            fullOpFirstOrder = {
                "type": "concatOperator",
                "operation": "composition",
                "A": diagonal_PhasefieldOp,
                "B": grad_t,
                "isMinus": False
            }
            fullOpSecondOrder = {
                "type": "concatOperator",
                "operation": "composition",
                "A": diagonal_PhasefieldOp,
                "B": grad_squared_t,
                "isMinus": False
            }
            if(motion_phase_field_coupling_norm==1):
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=(weight_v_3*12) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[fullOpSecondOrder]
                )
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=(weight_v_3*12) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[fullOpSecondOrder]
                )
            if(motion_phase_field_coupling_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=(weight_v_3*12) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[fullOpSecondOrder]
                )
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=(weight_v_3*12) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[fullOpSecondOrder]
                )
            if(motion_phase_field_coupling_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=(weight_v_3*12) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[fullOpSecondOrder]
                )
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=(weight_v_3*12) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[fullOpSecondOrder]
                )
        """
        if(motion_phase_field_coupling_norm==0):
            pass
        else:
            diagonalMaskOp = {
                "type": "diagonalOperator",
                "diagonalElements": mask_reg
            }
            if(motion_phase_field_coupling_norm==1):
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=weight_v_3 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[diagonalMaskOp]
                )
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=weight_v_3 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[diagonalMaskOp]
                )
            if(motion_phase_field_coupling_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_v_3 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[diagonalMaskOp]
                )
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_v_3 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[diagonalMaskOp]
                )
            if(motion_phase_field_coupling_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_v_3 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[diagonalMaskOp]
                )
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_v_3 ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[diagonalMaskOp]
                )
        """
        if(motion_phase_field_coupling_norm==0):
            pass
        else:
            diagonalMaskOp = {
                "type": "diagonalOperator",
                "diagonalElements": mask_reg
            }
            grad_t = {
                "type": "gradientOperator",
                "gradType": "backward",
                "gradDirection": 2,
                "inputDimension": dimensionsWithTime
            }
            weightedTimeDerOp = {
                "type": "concatOperator",
                "operation": "composition",
                "A": diagonalMaskOp,
                "B": grad_t,
                "isMinus": False
            }
            if(motion_phase_field_coupling_norm==1):
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=(weight_v_3*10) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[weightedTimeDerOp]
                )
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=(weight_v_3*10) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[weightedTimeDerOp]
                )
            if(motion_phase_field_coupling_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=(weight_v_3*10) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[weightedTimeDerOp]
                )
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=(weight_v_3*10) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[weightedTimeDerOp]
                )
            if(motion_phase_field_coupling_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=(weight_v_3*10) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[weightedTimeDerOp]
                )
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=(weight_v_3*10) ,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[weightedTimeDerOp]
                )
        """
        
        # ||mask_reg * d_t v||: motion smooth in time, without differencing across a change of the phase field (time_switch)
        if weight_v_time > 0 and motion_phase_field_coupling_norm != 0:
            a_v = time_switch_weights(phase_field, nPx, len(frames)) if time_switch_vt else None   # phase field of this iteration
            weightedTimeOp_v = {"type": "concatOperator", "operation": "composition", "isMinus": False,
                                "A": {"type": "diagonalOperator", "diagonalElements": mask_reg},
                                "B": time_derivative_op(a_v, dimensionsWithTime)}
            prox_vt = {1: "L1IsoProxDual", 2: "L1AnisoProxDual", 3: "L2proxDual"}[motion_phase_field_coupling_norm]
            for p_vt in (0, 1):
                solver.add_dual(prox_type=prox_vt, alpha=weight_v_time, f_list=[np.zeros(nPx * len(frames), dtype=np.float32)],
                                corresponding_primals=[corresponding_primals[p_vt]], operator_dict=[weightedTimeOp_v])

        print("before third solver")
        if v_pyramid:
            # coarse to fine pyramid with warping (same terms as above, data term linearised around the current v)
            if motionTerm_v != 1 and motionTerm_v != 0:
                raise ValueError("v_pyramid only supports the optical flow term (motionTerm_v = 1)")
            pyramid_now = i < v_pyramid_outer
            v_out = v_step_pyramid(phase_field if v_pyramid_source == "s" else reconstructed_image, phase_field,
                                   motionField_0, motionField_1, mask_reg, inputDimension, len(frames),
                                   v_pyramid_levels if pyramid_now else 0, v_pyramid_warps if pyramid_now else 1, v_pyramid_source == "s")
        else:
            v_out, _ = solver.solve()
        print("after third solver")
        
        motionField_0 = np.array(v_out[0], dtype=np.float32)
        motionField_1 = np.array(v_out[1], dtype=np.float32)
        motionField_0_frames = np.split(motionField_0, len(frames))
        motionField_1_frames = np.split(motionField_1, len(frames))

        del solver

        # for testing convergence
        diffV0 = np.linalg.norm(motionField_0 - motionField_0_old)
        diffV1 = np.linalg.norm(motionField_1 - motionField_1_old)
        diffU  = np.linalg.norm(reconstructed_image - reconstructed_image_old)
        diffPhaseField = np.linalg.norm(phase_field - phase_field_old)

        print("---------------------diffU is", diffU)
        print("---------------------diffPhfield is", diffPhaseField)
        print("---------------------diffV0 is", diffV0)
        print("---------------------diffV1 is", diffV1)
        

        #tol_first_solver = (diffV0 + diffV1 + diffU)/(2*nPx*len(frames))
        #tol_third_solver = (diffV0 + diffV1 + diffU)/(2*nPx*len(frames))

        folder_name = ""
        if(motionTerm_u_upwind==1):
            folder_name = (
                f"result{name}"
                f"resizeValue{resize_value}"
                f"epsilon{epsilon}"
                f"{weight_u_1}{dataTerm_norm_list[dataTerm_norm]}"
                f"{weight_u_2}{reg_u_norm_list[reg_u_norm]}"
                f"{weight_u_3}{reg_u_t_norm_list[reg_u_t_norm]}"
                f"{weight_u_4}{motionTerm_u_norm_list[motionTerm_u_norm]}"
                "Upwind"
                f"{motionTerm_u_list[motionTerm_u]}"
                f"{weight_pf_1}{reg_phase_field_norm_list[reg_phase_field_norm]}"
                f"{weight_pf_2}{phase_field_AT_norm_list[phase_field_AT_norm]}"
                f"{motionTerm_phase_field_list[motionTerm_phase_field]}"
                f"{weight_pf_3}{motionTerm_phase_field_norm_list[motionTerm_phase_field_norm]}"
                f"{weight_pf_4}{phase_field_motion_coupling_norm_list[phase_field_motion_coupling_norm]}"
                f"{weight_v_1}{motionTerm_v_norm_list[motionTerm_v_norm]}"
                f"{motionTerm_v_list[motionTerm_v]}"
                f"{weight_v_2}{reg_v_norm_list[reg_v_norm]}"
                f"{weight_v_3}{motion_phase_field_coupling_norm_list[motion_phase_field_coupling_norm]}"
            )
        else:
            folder_name = (
                f"result{name}"
                f"resizeValue{resize_value}"
                f"epsilon{epsilon}"
                f"{weight_u_1}{dataTerm_norm_list[dataTerm_norm]}"
                f"{weight_u_2}{reg_u_norm_list[reg_u_norm]}"
                f"{weight_u_3}{reg_u_t_norm_list[reg_u_t_norm]}"
                f"{weight_u_4}{motionTerm_u_norm_list[motionTerm_u_norm]}"
                f"{motionTerm_u_list[motionTerm_u]}"
                f"{weight_pf_1}{reg_phase_field_norm_list[reg_phase_field_norm]}"
                f"{weight_pf_2}{phase_field_AT_norm_list[phase_field_AT_norm]}"
                f"{motionTerm_phase_field_list[motionTerm_phase_field]}"
                f"{weight_pf_3}{motionTerm_phase_field_norm_list[motionTerm_phase_field_norm]}"
                f"{weight_pf_4}{phase_field_motion_coupling_norm_list[phase_field_motion_coupling_norm]}"
                f"{weight_v_1}{motionTerm_v_norm_list[motionTerm_v_norm]}"
                f"{motionTerm_v_list[motionTerm_v]}"
                f"{weight_v_2}{reg_v_norm_list[reg_v_norm]}"
                f"{weight_v_3}{motion_phase_field_coupling_norm_list[motion_phase_field_coupling_norm]}"
            )

        folder_path = os.path.join(folder_name_main, folder_name)

        os.makedirs(folder_path, exist_ok=True)
        
        if i > 20 and diffV0<tol and diffV1<tol and diffU<tol:
            print(f"converged for outer iterations in the {i}-th iteration.")
            for j in range(len(frames)-1):
                result_1_v = motionField_0_frames[j].reshape(inputDimension, order="F")
                result_2_v = motionField_1_frames[j].reshape(inputDimension, order="F")
                result_1_u = reconstructed_image_frames[j].reshape(inputDimension, order="F")

                #frame_start = j * nPx
                #frame_end = (j + 1) * nPx
                #result_1_u = reconstructed_image[frame_start:frame_end].reshape(inputDimension, order="F")

                magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)
                vmax1 = np.max(np.abs(result_1_v))
                vmax2 = np.max(np.abs(result_2_v))
                vmax_mag = np.percentile(magnitude_v, 99)
                
                magnitude_u = np.sqrt(result_1_u**2)
                umax1 = np.max(np.abs(result_1_u))
                umax_mag = np.percentile(magnitude_u, 99)

                print(f"vmax1 for i = {i} is ", vmax1)
                print(f"vmax2 for i = {i} is ", vmax2)
                
                print(f"umax1 for i = {i} is ", umax1)          

                plt.figure(figsize=(7,7))
                
                plt.subplot(2,2,1)
                plt.title("magnitude of u")
                plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax1)
                #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
                plt.axis("off")
                
                plt.subplot(2,2,3)
                plt.title("v_x")
                plt.imshow(result_2_v, cmap="seismic", vmin=-vmax2, vmax=vmax2)
                plt.axis("off")

                plt.subplot(2,2,4)
                plt.title("v_y")
                plt.imshow(result_1_v, cmap="seismic", vmin=-vmax1, vmax=vmax1)
                plt.axis("off")

                plt.subplot(2,2,2)
                plt.title("magnitude of v")
                plt.imshow(magnitude_v, cmap="gray", vmin=0, vmax=vmax_mag)
                plt.axis("off")
                
                #plt.savefig(f"result{dataTerm_norm_list[dataTerm_norm]}{reg_u_norm_list[reg_u_norm]}{motionTerm_u_norm_list[motionTerm_u_norm]}{motionTerm_u_list[motionTerm_u]}{motionTerm_v_norm_list[motionTerm_v_norm]}{motionTerm_v_list[motionTerm_v]}{reg_v_norm_list[reg_v_norm]}{i+1},{j+1}.png", dpi=200, bbox_inches="tight")
                file_path = os.path.join(folder_path, f"Converged{i+1},{j+1}.png")
                plt.savefig(file_path, dpi=200, bbox_inches="tight")
                plt.close()
            break

        # visualization, starting from the second frame, because we use backward differences in the time derivatives of the image
        for j in range(0,len(frames)):
            if ((i+1) % figures_frequence == 0):
                result_1_v = motionField_0_frames[j].reshape(inputDimension, order="F")
                result_2_v = motionField_1_frames[j].reshape(inputDimension, order="F")
                result_1_u = reconstructed_image_frames[j].reshape(inputDimension, order="F")
                result_1_phase_field = phase_field_frames[j].reshape(inputDimension, order="F")
                result_1_phase_field_pore = phase_field_pore_frames[j].reshape(inputDimension, order="F")

                #frame_start = j * nPx
                #frame_end = (j + 1) * nPx
                #result_1_u = reconstructed_image[frame_start:frame_end].reshape(inputDimension, order="F")

                magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)
                vmax1 = np.max(result_1_v)
                vmin1 = np.min(result_1_v)
                vAbsMax1 = np.max(np.abs(result_1_v))
                vmax1_mag = np.percentile(np.abs(result_1_v), 99.99)
                vmax2 = np.max(result_2_v)
                vmin2 = np.min(result_2_v)
                vAbsMax2 = np.max(np.abs(result_2_v))
                vmax2_mag = np.percentile(np.abs(result_2_v), 99.99)
                vmax_mag = np.percentile(magnitude_v, 99.999)
                
                magnitude_u = np.sqrt(result_1_u**2)
                umax1 = np.max(np.abs(result_1_u))
                umax_mag = np.percentile(magnitude_u, 99.9)

                phase_field1 = np.max(np.abs(result_1_phase_field))
                phase_field2 = np.min(np.abs(result_1_phase_field))

                print(f"vmax1 for i = {i} is ", vmax1)
                print(f"vmax2 for i = {i} is ", vmax2)
                print(f"vmin1 for i = {i} is ", vmin1)
                print(f"vmin2 for i = {i} is ", vmin2)
                print(f"vmax1_mag for i = {i} is ", vmax1_mag)
                print(f"vmax2_mag for i = {i} is ", vmax2_mag)
                
                print(f"umax1 for i = {i} is ", umax1)    
                print(f"phase_field1 for i = {i} is ", phase_field1)
                print(f"max of phase_field1 for i = {i} is ", np.max(result_1_phase_field))
                print(f"min phase_field1 for i = {i} is ", np.min(result_1_phase_field))          

                plt.figure(figsize=(7,7))
                
                plt.subplot(2,3,1)
                plt.title("magnitude of u")
                plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax1)
                #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
                plt.axis("off")
                
                plt.subplot(2,3,2)
                plt.title("phase field without pores")
                plt.imshow(result_1_phase_field, cmap="gray", vmin=0.0, vmax=phase_field1)
                plt.axis("off")

                plt.subplot(2,3,3)
                plt.title("phase field of pores")
                plt.imshow(result_1_phase_field_pore, cmap="gray", vmin=0.0, vmax=1.0)
                plt.axis("off")

                plt.subplot(2,3,5)
                plt.title("v_x")
                plt.imshow(result_2_v, cmap="seismic", vmin=-vmax2_mag, vmax=vmax2_mag)
                #plt.imshow(result_2_v, cmap="seismic", vmin=-vAbsMax2, vmax=vAbsMax2)
                plt.axis("off")

                plt.subplot(2,3,6)
                plt.title("v_y")
                #plt.imshow(result_2_v, cmap="seismic", vmin=-vmax1_mag, vmax=vmax1_mag)
                plt.imshow(result_1_v, cmap="seismic", vmin=-vAbsMax1, vmax=vAbsMax1)
                plt.axis("off")

                plt.subplot(2,3,4)
                plt.title("magnitude of v")
                plt.imshow(magnitude_v, cmap="gray", vmin=0, vmax=vmax_mag)
                plt.axis("off")
                
                #plt.savefig(f"result{dataTerm_norm_list[dataTerm_norm]}{reg_u_norm_list[reg_u_norm]}{motionTerm_u_norm_list[motionTerm_u_norm]}{motionTerm_u_list[motionTerm_u]}{motionTerm_v_norm_list[motionTerm_v_norm]}{motionTerm_v_list[motionTerm_v]}{reg_v_norm_list[reg_v_norm]}{i+1},{j+1}.png", dpi=200, bbox_inches="tight")
                file_path = os.path.join(folder_path, f"{i+1},{j+1}.png")
                plt.savefig(file_path, dpi=200, bbox_inches="tight")
                plt.close()
        # visualization
        #if ((i+1) % errors_frequence == 0 or (i+1)==num_iterations):
        if ((i+1)==num_iterations):
            image_folder_path = os.path.join(folder_path, "images")
            motion_field_0_folder_path = os.path.join(folder_path, "motionField_0")
            motion_field_1_folder_path = os.path.join(folder_path, "motionField_1")
            motion_field_magnitude_folder_path = os.path.join(folder_path, "motionField_magnitude")
            phase_field_folder_path = os.path.join(folder_path, "phase_field")
            
            
            os.makedirs(image_folder_path, exist_ok=True)
            os.makedirs(motion_field_0_folder_path, exist_ok=True)
            os.makedirs(motion_field_1_folder_path, exist_ok=True)
            os.makedirs(motion_field_magnitude_folder_path, exist_ok=True)
            os.makedirs(phase_field_folder_path, exist_ok=True)
            for j in range(len(frames)):
                result_1_v = motionField_0_frames[j].reshape(inputDimension, order="F")
                result_2_v = motionField_1_frames[j].reshape(inputDimension, order="F")
                result_1_u = reconstructed_image_frames[j].reshape(inputDimension, order="F")
                result_1_phase_field = phase_field_frames[j].reshape(inputDimension, order="F")

                #frame_start = j * nPx
                #frame_end = (j + 1) * nPx
                #result_1_u = reconstructed_image[frame_start:frame_end].reshape(inputDimension, order="F")

                magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)
                vmax1 = np.max(np.abs(result_1_v))
                vmax1_mag = np.percentile(np.abs(result_1_v), 99)
                vmax2 = np.max(np.abs(result_2_v))
                vmax2_mag = np.percentile(np.abs(result_2_v), 99)
                vmax_mag = np.percentile(magnitude_v, 99)
                
                magnitude_u = np.sqrt(result_1_u**2)
                umax1 = np.max(np.abs(result_1_u))
                umax_mag = np.percentile(magnitude_u, 99)

                phase_field1 = np.max(np.abs(result_1_phase_field))
                phase_field2 = np.min(np.abs(result_1_phase_field))

                print(f"vmax1 for i = {i} is ", vmax1)
                print(f"vmax2 for i = {i} is ", vmax2)
                print(f"vmax1_mag for i = {i} is ", vmax1_mag)
                print(f"vmax2_mag for i = {i} is ", vmax2_mag)
                
                print(f"umax1 for i = {i} is ", umax1)    
                print(f"phase_field1 for i = {i} is ", phase_field1)
                print(f"max of phase_field1 for i = {i} is ", np.max(result_1_phase_field))
                print(f"min phase_field1 for i = {i} is ", np.min(result_1_phase_field))          

                plt.figure(figsize=(7,7))
                
                plt.plot()
                plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax1)
                #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
                plt.axis("off")

                image_path = os.path.join(image_folder_path, f"{i+1},{j+1}.png")
                plt.savefig(image_path, dpi=200, bbox_inches="tight")
                plt.close()
                
                plt.plot()
                plt.imshow(result_1_phase_field, cmap="gray", vmin=phase_field2, vmax=phase_field1)
                plt.axis("off")

                phase_field_path = os.path.join(phase_field_folder_path, f"{i+1},{j+1}.png")
                plt.savefig(phase_field_path, dpi=200, bbox_inches="tight")
                plt.close()

                plt.plot()
                plt.imshow(result_2_v, cmap="seismic", vmin=-vmax2, vmax=vmax2)
                plt.axis("off")

                motion_field_1_path = os.path.join(motion_field_1_folder_path, f"{i+1},{j+1}.png")
                plt.savefig(motion_field_1_path, dpi=200, bbox_inches="tight")
                plt.close()

                plt.plot()
                plt.imshow(result_1_v, cmap="seismic", vmin=-vmax1, vmax=vmax1)
                plt.axis("off")

                motion_field_0_path = os.path.join(motion_field_0_folder_path, f"{i+1},{j+1}.png")
                plt.savefig(motion_field_0_path, dpi=200, bbox_inches="tight")
                plt.close()

                plt.plot()
                plt.imshow(magnitude_v, cmap="gray", vmin=0, vmax=vmax_mag)
                plt.axis("off")

                motion_field_magnitude_path = os.path.join(motion_field_magnitude_folder_path, f"{i+1},{j+1}.png")
                plt.savefig(motion_field_magnitude_path, dpi=200, bbox_inches="tight")
                plt.close()
            """
            if groundFlowExists:
                flo_files = [f for f in os.listdir(image_folder_gt) if f.lower().endswith('.flo')]
                flo_path = os.path.join(image_folder_gt, flo_files[0])
                # use the same cropping as for the images
                gt_vx, gt_vy = read_flo_file(flo_path)
                if cut:
                    h_orig, w_orig = gt_vx.shape
                    x1 = int(w_orig * cut_width[0])
                    x2 = int(w_orig * (1 - cut_width[1]))
                    y1 = int(h_orig * cut_height[0])
                    y2 = int(h_orig * (1 - cut_height[1]))
                    gt_vx = gt_vx[y1:y2, x1:x2]
                    gt_vy = gt_vy[y1:y2, x1:x2]
                
                if(resize_value!=1.0):
                    #resize
                    gt_vx = cv2.resize(gt_vx, None, fx=resize_value, fy=resize_value)
                    gt_vy = cv2.resize(gt_vy, None, fx=resize_value, fy=resize_value)
                    
                invalid_mask = (np.abs(gt_vx) > 1e8) | (np.abs(gt_vy) > 1e8)
                valid_mask = ~invalid_mask
                

                calc_vx = motionField_1[:nPx].reshape(gt_vx.shape, order="F")
                calc_vy = motionField_0[:nPx].reshape(gt_vy.shape, order="F")

                print("gt shape:", gt_vx.shape)
                print("motionField shape:", motionField_0.shape)

                epe = ef.compute_epe(calc_vx,calc_vy,gt_vx,gt_vy,valid_mask)
                aae = ef.compute_aae(calc_vx,calc_vy,gt_vx,gt_vy,valid_mask)

                if (epe<best_epe):
                    best_epe_parameters = [weight_1, weight_2, weight_3, weight_pf_1, weight_pf_2, dataTerm_norm, reg_u_norm, motionTerm_u, motionTerm_u_norm, motionTerm_u_upwind, motionTerm_v, motionTerm_v_norm, reg_v_norm]
                    best_epe=epe
                

                if (aae<best_aae):
                    best_aae_parameters = [weight_1, weight_2, weight_3, weight_pf_1, weight_pf_2, dataTerm_norm, reg_u_norm, motionTerm_u, motionTerm_u_norm, motionTerm_u_upwind, motionTerm_v, motionTerm_v_norm, reg_v_norm]
                    best_aae=aae
                

                if (aae+epe<best_aae_epe_combined[0] + best_aae_epe_combined[1]):
                    best_aae_epe_combined_parameters = [weight_1, weight_2, weight_3, weight_pf_1, weight_pf_2, dataTerm_norm, reg_u_norm, motionTerm_u, motionTerm_u_norm, motionTerm_u_upwind, motionTerm_v, motionTerm_v_norm, reg_v_norm]
                    best_aae_epe_combined=[aae,epe]
                
                mag_calc = cv2.magnitude(calc_vx, calc_vy)
                mag_gt   = cv2.magnitude(gt_vx, gt_vy)
                
                numerator = calc_vx * gt_vx + calc_vy * gt_vy + 1.0
                denominator = np.sqrt(mag_calc**2 + 1.0) * np.sqrt(mag_gt**2 + 1.0)
                
                clipped_cos = np.clip(numerator / denominator, -1.0, 1.0)
                mean_aee_rad = np.mean(np.arccos(clipped_cos))
                mean_aee_deg = np.degrees(mean_aee_rad)
                
                print("\n========= METRICS EVALUATION (.flo) =========")
                print(f"Verglichene Datei           : {flo_files[0]}")
                print(f"sum(alid_mask)/(valid_mask.size)     : {np.sum(valid_mask)} / {valid_mask.size}")
                print(f"Average Endpoint Error (EPE): {mean_epe:.4f} Pixel")
                print(f"Average Angular Error (AAE)  : {mean_aee_deg:.4f}° Grad")
                print("=============================================")

                error_file_path = os.path.join(folder_name_main, f"errors{name}.txt")
                # Ergebnisse an die parameters.txt anhängen
                with open(error_file_path, "a", encoding="utf-8") as f:
                    f.write(f"weight for {dataTerm_norm_list[dataTerm_norm]} data term is {weight_1}\nweight for the {reg_u_norm_list[reg_u_norm]} regularizer of the image u is {weight_2}\nweight for the {motionTerm_u_norm_list[motionTerm_u_norm]} {motionTerm_u_list[motionTerm_u]} motion term of u is {weight_3}\nweight for the {motionTerm_v_norm_list[motionTerm_v_norm]} {motionTerm_v_list[motionTerm_v]} motion term for v is {weight_pf_1}\nweight for the {reg_v_norm_list[reg_v_norm]} regularizer of the motion fields v is {weight_pf_2}\nThe error values are: \n---Errors at outer iteration {i+1}--- \nEPE: {epe:.4f}\nAEE (Deg): {aae:.4f}\n\n")
                    #f.write(f"weight for the {motionTerm_v_norm_list[motionTerm_v_norm]} {motionTerm_v_list[motionTerm_v]} motion term for v is {weight_4}\nweight for the {reg_v_norm_list[reg_v_norm]} regularizer of the motion fields v is {weight_pf_2}\nThe error values are: \n---Errors at outer iteration {i+1}--- \nEPE: {epe:.4f}\nAEE (Deg): {aae:.4f}\n\n")
            """
        if groundFlowExists and syntheticData:
            for j in range(len(frames)):
                calc_vx = motionField_1_frames[j].reshape(motionfield_1_gt_list[j].shape, order="F")
                calc_vy = motionField_0_frames[j].reshape(motionfield_0_gt_list[j].shape, order="F")

                print("gt shape:", motionfield_1_gt_list[j].shape)
                print("motionField shape:", calc_vx.shape)
                """
                calc_vx = motionField_1[:nPx].reshape(gt_vx.shape, order="F")
                calc_vy = -motionField_0[:nPx].reshape(gt_vy.shape, order="F")
                """
                epe = ef.compute_epe(calc_vx,calc_vy,motionfield_1_gt_list[j],motionfield_0_gt_list[j])
                aae = ef.compute_aae(calc_vx,calc_vy,motionfield_1_gt_list[j],motionfield_0_gt_list[j])

                print("epe is ", epe)
                print("aae is ", aae)

        gc.collect()


error_file_path = os.path.join(folder_name_main, f"errors{name}.txt")
# Ergebnisse an die parameters.txt anhängen
"""
if groundFlowExists:
    with open(error_file_path, "a", encoding="utf-8") as f:
        f.write(
            f"The best epe is achieved for:\n"
            f"weight for {dataTerm_norm_list[best_epe_parameters[5]]} data term is {best_epe_parameters[0]}\n"
            f"weight for the {reg_u_norm_list[best_epe_parameters[6]]} regularizer of the image u is {best_epe_parameters[1]}\n"
            f"weight for the {motionTerm_u_norm_list[best_epe_parameters[8]]} {motionTerm_u_list[best_epe_parameters[7]]} motion term of u is {best_epe_parameters[2]}\n"
            f"weight for the {motionTerm_v_norm_list[best_epe_parameters[11]]} {motionTerm_v_list[best_epe_parameters[10]]} motion term for v is {best_epe_parameters[3]}\n"
            f"weight for the {reg_v_norm_list[best_epe_parameters[12]]} regularizer of the motion fields v is {best_epe_parameters[4]}\n"
            f"The epe is: {best_epe:.4f}\n\n"
        )
        f.write(
            f"The best aae is {best_aae:.4f} and is achieved for:\n"
            f"weight for {dataTerm_norm_list[best_aae_parameters[5]]} data term is {best_aae_parameters[0]}\n"
            f"weight for the {reg_u_norm_list[best_aae_parameters[6]]} regularizer of the image u is {best_aae_parameters[1]}\n"
            f"weight for the {motionTerm_u_norm_list[best_aae_parameters[8]]} {motionTerm_u_list[best_aae_parameters[7]]} motion term of u is {best_aae_parameters[2]}\n"
            f"weight for the {motionTerm_v_norm_list[best_aae_parameters[11]]} {motionTerm_v_list[best_aae_parameters[10]]} motion term for v is {best_aae_parameters[3]}\n"
            f"weight for the {reg_v_norm_list[best_aae_parameters[12]]} regularizer of the motion fields v is {best_aae_parameters[4]}\n"
            f"The aae is: {best_aae:.4f}\n\n"
        )
        f.write(
            f"The best combined epe and aae is {best_aae_epe_combined[0]:.4f} (epe), {best_aae_epe_combined[1]:.4f} (aae) and is achieved for:\n"
            f"weight for {dataTerm_norm_list[best_aae_epe_combined_parameters[5]]} data term is {best_aae_epe_combined_parameters[0]}\n"
            f"weight for the {reg_u_norm_list[best_aae_epe_combined_parameters[6]]} regularizer of the image u is {best_aae_epe_combined_parameters[1]}\n"
            f"weight for the {motionTerm_u_norm_list[best_aae_epe_combined_parameters[8]]} {motionTerm_u_list[best_aae_epe_combined_parameters[7]]} motion term of u is {best_aae_epe_combined_parameters[2]}\n"
            f"weight for the {motionTerm_v_norm_list[best_aae_epe_combined_parameters[11]]} {motionTerm_v_list[best_aae_epe_combined_parameters[10]]} motion term for v is {best_aae_epe_combined_parameters[3]}\n"
            f"weight for the {reg_v_norm_list[best_aae_epe_combined_parameters[12]]} regularizer of the motion fields v is {best_aae_epe_combined_parameters[4]}\n"
            f"The error values are: {best_aae_epe_combined[0]:.4f} and {best_aae_epe_combined[1]:.4f}\n\n"
        )
"""