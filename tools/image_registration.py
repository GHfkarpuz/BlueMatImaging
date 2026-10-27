import cv2
import numpy as np

def estimate_translation_phase_correlation(frame_ref, frame_cur):
    """
    Gibt (dx, dy) zurück — wie weit frame_cur verschoben ist
    relativ zu frame_ref.
    """
    # beide müssen float32 sein
    ref = frame_ref.astype(np.float32)
    cur = frame_cur.astype(np.float32)

    shift, response = cv2.phaseCorrelate(ref, cur)
    dx, dy = shift  # dx = Verschiebung in x, dy = in y
    return dx, dy


def apply_translation(frame, dx, dy):
    """
    Verschiebt frame um (-dx, -dy), also korrigiert die Verschiebung.
    """
    h, w = frame.shape
    M = np.float32([[1, 0, -dx],
                    [0, 1, -dy]])
    shifted = cv2.warpAffine(frame, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return shifted


def stabilize_frames_phase_correlation(frames):
    """
    Registriert alle Frames auf den ersten Frame.
    Gibt stabilisierte Frames + die geschätzten Shifts zurück.
    """
    reference = frames[0]
    stabilized = [reference.copy()]
    shifts = [(0.0, 0.0)]

    for i in range(1, len(frames)):
        dx, dy = estimate_translation_phase_correlation(reference, frames[i])
        corrected = apply_translation(frames[i], dx, dy)
        stabilized.append(corrected)
        shifts.append((dx, dy))
        print(f"Frame {i}: dx={dx:.3f}, dy={dy:.3f}")

    return stabilized, shifts

def stabilize_frames_ecc(frames, max_iterations=50, epsilon=1e-4):
    """
    ECC-basierte Registrierung — subpixelgenau, nur Translation.
    """
    reference = frames[0].astype(np.float32)
    stabilized = [reference.copy()]
    shifts = [(0.0, 0.0)]

    warp_mode = cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iterations, epsilon)

    for i in range(1, len(frames)):
        cur = frames[i].astype(np.float32)
        warp_matrix = np.eye(2, 3, dtype=np.float32)  # Startwert: keine Verschiebung

        try:
            _, warp_matrix = cv2.findTransformECC(
                reference, cur,
                warp_matrix, warp_mode, criteria
            )
            dx = warp_matrix[0, 2]
            dy = warp_matrix[1, 2]

            h, w = cur.shape
            corrected = cv2.warpAffine(
                cur, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            stabilized.append(corrected)
            shifts.append((dx, dy))
            print(f"Frame {i}: dx={dx:.4f}, dy={dy:.4f}")

        except cv2.error as e:
            print(f"Frame {i}: ECC failed ({e}), using phase correlation fallback")
            dx, dy = estimate_translation_phase_correlation(reference, cur)
            corrected = apply_translation(cur, dx, dy)
            stabilized.append(corrected)
            shifts.append((dx, dy))

    return stabilized, shifts
def stabilize_frames_feature_based(frames):
    orb = cv2.ORB_create(nfeatures=500)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    reference = frames[0].astype(np.uint8)
    kp_ref, des_ref = orb.detectAndCompute(reference, None)

    stabilized = [frames[0].copy()]
    shifts = [(0.0, 0.0)]

    # Absicherung: Referenz-Frame hat keine Features
    if des_ref is None or len(des_ref) < 4:
        print("Warnung: Referenz-Frame hat keine Keypoints – keine Stabilisierung möglich.")
        return frames, [(0.0, 0.0)] * len(frames)

    for i in range(1, len(frames)):
        cur = frames[i].astype(np.uint8)
        kp_cur, des_cur = orb.detectAndCompute(cur, None)

        if des_cur is None or len(des_cur) < 4:
            print(f"Frame {i}: keine Keypoints – Identity")
            stabilized.append(frames[i].copy())
            shifts.append((0.0, 0.0))
            continue

        matches = bf.match(des_ref, des_cur)

        # Absicherung: zu wenige Matches
        if len(matches) < 4:
            print(f"Frame {i}: nur {len(matches)} Matches – Identity")
            stabilized.append(frames[i].copy())
            shifts.append((0.0, 0.0))
            continue

        matches = sorted(matches, key=lambda m: m.distance)
        good    = matches[:50]

        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(
            pts_cur, pts_ref,
            method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        if M is None:
            print(f"Frame {i}: RANSAC fehlgeschlagen – Identity")
            stabilized.append(frames[i].copy())
            shifts.append((0.0, 0.0))
            continue

        dx = M[0, 2]
        dy = M[1, 2]
        h, w = cur.shape
        corrected = cv2.warpAffine(cur, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        stabilized.append(corrected.astype(np.float32))
        shifts.append((dx, dy))
        print(f"Frame {i}: dx={dx:.4f}, dy={dy:.4f}, "
              f"inliers={int(inliers.sum())}/{len(good)}")

    return stabilized, shifts