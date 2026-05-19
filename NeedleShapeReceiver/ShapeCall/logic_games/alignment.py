import numpy as np
import vtk


class NeedleAlignment:

    def __init__(
        self,
        ds=1.0,
        eps=1e-6,
        catheter_first_point_is_base=False
    ):

        self.DS = ds
        self.EPS = eps

        self.CATHETER_FIRST_POINT_IS_BASE = (
            catheter_first_point_is_base
        )

    # ==================================================
    # RESAMPLE
    # ==================================================
    def resample(self, points):

        seg = np.linalg.norm(
            np.diff(points, axis=0),
            axis=1
        )

        arc = np.insert(
            np.cumsum(seg),
            0,
            0.0
        )

        if arc[-1] < self.EPS:
            raise RuntimeError("Curve too small")

        s = np.arange(
            0.0,
            arc[-1],
            self.DS
        )

        if len(s) == 0 or abs(s[-1] - arc[-1]) > self.EPS:
            s = np.append(s, arc[-1])

        return np.vstack([
            np.interp(s, arc, points[:, 0]),
            np.interp(s, arc, points[:, 1]),
            np.interp(s, arc, points[:, 2]),
        ]).T

    # ==================================================
    # ALIGN
    # ==================================================
    def align(self, needle_pts, catheter_pts):

        needle_pts = np.asarray(
            needle_pts,
            dtype=float
        )

        catheter_pts = np.asarray(
            catheter_pts,
            dtype=float
        )

        if len(catheter_pts) < 2:
            raise RuntimeError("Catheter too small")

        # ------------------------------------------------
        # REORIENT
        # ------------------------------------------------
        if self.CATHETER_FIRST_POINT_IS_BASE:
            cat_reoriented = catheter_pts.copy()
        else:
            cat_reoriented = catheter_pts[::-1].copy()

        # ------------------------------------------------
        # RESAMPLE
        # ------------------------------------------------
        cat_resampled = self.resample(cat_reoriented)

        # ------------------------------------------------
        # LENGTHS
        # ------------------------------------------------
        L_needle = np.sum(
            np.linalg.norm(
                np.diff(needle_pts, axis=0),
                axis=1
            )
        )

        L_cat = np.sum(
            np.linalg.norm(
                np.diff(cat_resampled, axis=0),
                axis=1
            )
        )

        L_ext = L_needle - L_cat

        if L_ext < -self.EPS:
            raise RuntimeError(
                "Catheter already longer than needle"
            )

        n_ext = int(
            round(max(L_ext, 0.0) / self.DS)
        )

        # ------------------------------------------------
        # EXTENSION
        # ------------------------------------------------
        if n_ext > 0:

            base = cat_resampled[0]

            direction = (
                cat_resampled[0]
                - cat_resampled[1]
            )

            direction = (
                direction
                / np.linalg.norm(direction)
            )

            ext_pts = np.array([
                base + direction * (
                    (n_ext - i) * self.DS
                )
                for i in range(n_ext)
            ])

            cat_extended = np.vstack([
                ext_pts,
                cat_resampled
            ])

        else:

            cat_extended = cat_resampled

        # ------------------------------------------------
        # ALIGNMENT
        # ------------------------------------------------
        n_align = min(
            len(needle_pts),
            len(cat_extended)
        )

        if n_align < 3:
            raise RuntimeError(
                "Not enough alignment points"
            )

        source = vtk.vtkPoints()
        target = vtk.vtkPoints()

        for i in range(n_align):

            source.InsertNextPoint(
                *needle_pts[i]
            )

            target.InsertNextPoint(
                *cat_extended[i]
            )

        transform = vtk.vtkLandmarkTransform()

        transform.SetSourceLandmarks(source)
        transform.SetTargetLandmarks(target)

        transform.SetModeToRigidBody()

        transform.Update()

        return transform
