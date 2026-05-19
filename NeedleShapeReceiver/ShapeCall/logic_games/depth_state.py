import numpy as np


class NeedleDepthState:

    def __init__(self):
        self.skin_entry = None
        self.needle_pose_position = None
        self.needle_pose_orientation = None
        self.insertion_depth = 0.0
        self.state_initialized = False

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------
    def initialize_from_defaults(
        self,
        skin_entry,
        needle_pose_position,
        needle_pose_orientation,
        insertion_depth=0.0,
    ):
        self.skin_entry = np.asarray(skin_entry, dtype=float)
        self.needle_pose_position = np.asarray(
            needle_pose_position, dtype=float
        )
        self.needle_pose_orientation = self._normalize(
            needle_pose_orientation
        )
        self.insertion_depth = float(insertion_depth)
        self.state_initialized = True

    # --------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------
    def _normalize(self, v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n <= 1e-9:
            raise RuntimeError("Zero-length orientation vector")
        return v / n

    def _curve_arclength(self, pts):
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return 0.0
        return float(
            np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        )

    def update_pose_from_depth(self):
        # /stage/state/needle_pose convention: z = insertion_depth in mm.
        self.needle_pose_position = np.array(
            [0.0, 0.0, float(self.insertion_depth)],
            dtype=float
        )
        return self.needle_pose_position

    # --------------------------------------------------
    # ALIGNMENT
    # --------------------------------------------------
    def apply_alignment_measurement(
        self,
        reference_pts,
        updated_orientation=None,
    ):
        # Set insertion depth from fiducial arclength.
        depth = self._curve_arclength(reference_pts)
        self.insertion_depth = float(depth)

        if updated_orientation is not None:
            self.needle_pose_orientation = self._normalize(
                updated_orientation
            )

        self.update_pose_from_depth()
        return self.export_state()

    # --------------------------------------------------
    # EXPORT
    # --------------------------------------------------
    def export_state(self):
        return {
            "skin_entry": (
                None if self.skin_entry is None
                else self.skin_entry.copy()
            ),
            "needle_pose_position": (
                None if self.needle_pose_position is None
                else self.needle_pose_position.copy()
            ),
            "needle_pose_orientation": (
                None if self.needle_pose_orientation is None
                else self.needle_pose_orientation.copy()
            ),
            "insertion_depth": float(self.insertion_depth),
            "state_initialized": bool(self.state_initialized),
        }