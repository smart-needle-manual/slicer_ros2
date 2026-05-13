import numpy as np


class NeedleCurvature:

    def __init__(self,
                 curv_smooth_window=7):

        if curv_smooth_window < 3:
            raise ValueError(
                "Window must be >= 3"
            )

        if curv_smooth_window % 2 == 0:
            raise ValueError(
                "Window must be odd"
            )

        self.window = curv_smooth_window

    # ==================================================
    # ARCLENGTH
    # ==================================================
    def compute_arclength(self, points):

        seg = np.linalg.norm(
            np.diff(points, axis=0),
            axis=1
        )

        s = np.insert(
            np.cumsum(seg),
            0,
            0.0
        )

        return s

    # ==================================================
    # LOCAL QUADRATIC DERIVATIVE
    # ==================================================
    def local_derivatives(self, s, xyz):

        n = len(s)

        d1 = np.zeros((n, 3))
        d2 = np.zeros((n, 3))

        half = self.window // 2

        for i in range(n):

            i0 = max(0, i - half)
            i1 = min(n, i + half + 1)

            ss = s[i0:i1] - s[i]

            for dim in range(3):

                yy = xyz[i0:i1, dim]

                deg = min(2, len(ss) - 1)

                coef = np.polyfit(ss, yy, deg)

                if deg == 2:

                    a, b, c = coef

                    d1[i, dim] = b
                    d2[i, dim] = 2 * a

                elif deg == 1:

                    a, b = coef

                    d1[i, dim] = a
                    d2[i, dim] = 0.0

        return d1, d2

    # ==================================================
    # CURVATURE
    # ==================================================
    def compute_curvature(self, points):

        pts = np.asarray(points, dtype=float)

        if len(pts) < 5:
            raise RuntimeError(
                "Not enough points"
            )

        s = self.compute_arclength(pts)

        d1, d2 = self.local_derivatives(s, pts)

        curvature = np.zeros(len(pts))

        for i in range(len(pts)):

            r1 = d1[i]
            r2 = d2[i]

            numer = np.linalg.norm(
                np.cross(r1, r2)
            )

            denom = (
                np.linalg.norm(r1) ** 3
            )

            if denom > 1e-12:
                curvature[i] = numer / denom

        return {
            "arclength": s,
            "curvature": curvature,
            "tangent": d1,
            "second_derivative": d2,
        }
