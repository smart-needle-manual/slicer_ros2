import logging
import os
import csv
import re
import json
import subprocess
import qt
import slicer
import vtk
import numpy as np
import time

from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _
from slicer.i18n import translate

from logic_games.alignment import NeedleAlignment
from logic_games.depth_state import NeedleDepthState


# ==================================================
# MODULE
# ==================================================

class ShapeCall(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = _("ShapeCall")
        parent.categories = [translate("qSlicerAbstractCoreModule", "Needle Visualization")]
        parent.contributors = ["RB"]


# ==================================================
# WIDGET
# ==================================================

class ShapeCallWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # ---------------- UI ----------------
        self.statusLabel    = qt.QLabel("Ready")
        self.numPointsLabel = qt.QLabel("0")
        self.lastPointLabel = qt.QLabel("N/A")

        info   = qt.QGroupBox("Needle Info")
        layout = qt.QFormLayout(info)
        layout.addRow("Points:", self.numPointsLabel)
        layout.addRow("Last:",   self.lastPointLabel)
        layout.addRow("Status:", self.statusLabel)
        self.layout.addWidget(info)

        # ---------------- BUTTONS ----------------
        self.alignButton      = qt.QPushButton("Align")
        self.saveShapeButton  = qt.QPushButton("Save Shape")
        self.resetAlignButton = qt.QPushButton("Reset Alignment")
        self.exportButton     = qt.QPushButton("Export .needle")

        self.alignButton.connect("clicked(bool)",      self.onAlign)
        self.saveShapeButton.connect("clicked(bool)",  self.onSaveShape)
        self.resetAlignButton.connect("clicked(bool)", self.onResetAlignment)
        self.exportButton.connect("clicked(bool)",     self.onExportNeedle)

        alignBox    = qt.QGroupBox("Alignment")
        alignLayout = qt.QVBoxLayout(alignBox)
        alignLayout.addWidget(self.alignButton)
        alignLayout.addWidget(self.saveShapeButton)
        alignLayout.addWidget(self.resetAlignButton)
        alignLayout.addWidget(self.exportButton)
        self.layout.addWidget(alignBox)

        # ---------------- CURVATURE SOURCE ----------------
        self.csvPathEdit       = qt.QLineEdit("/home/obgynbrachy/MATLAB_dosimetry/needle_poses.csv")
        self.needleIdCombo     = qt.QComboBox()
        self.loadCsvButton     = qt.QPushButton("Load CSV")
        self.publishCurvButton = qt.QPushButton("Publish Curvatures (External)")
        self.revertFbgButton   = qt.QPushButton("Revert to FBG")

        self.loadCsvButton.connect("clicked(bool)",     self.onLoadCurvatureCSV)
        self.publishCurvButton.connect("clicked(bool)", self.onPublishCurvaturesExt)
        self.revertFbgButton.connect("clicked(bool)",   self.onRevertToFBG)

        curvBox    = qt.QGroupBox("Curvature Source")
        curvLayout = qt.QFormLayout(curvBox)
        curvLayout.addRow("CSV file:",  self.csvPathEdit)
        curvLayout.addRow("Needle ID:", self.needleIdCombo)
        curvLayout.addRow(self.loadCsvButton)
        curvLayout.addRow(self.publishCurvButton)
        curvLayout.addRow(self.revertFbgButton)
        self.layout.addWidget(curvBox)

        self.layout.addStretch(1)

        # ---------------- STATE ----------------
        # latestBodyPts  : raw body-frame points from last PoseArray, in mm
        #                  (bridge ×1000 undone by dividing at read time)
        # _latestWorldPts: body-frame points after alignment transform (mm).
        #                  In alignment-time world coords — depth offset is NOT
        #                  baked in. NeedleInsertionTransform moves both
        #                  NeedleCurveModel and NeedleFiducials live.
        self.latestBodyPts   = None   # (N,3) float64, mm, body frame
        self._latestWorldPts = None   # (N,3) float64, mm, alignment-time world frame

        # _alignMatrix: 4×4 numpy matrix extracted ONCE in onAlign().
        # Guards onDepth — transform only fires after alignment is complete.
        self._alignMatrix       = None   # np (4,4)
        self.alignmentDepth     = 0.0    # raw ROS depth at alignment time (mm)
        self.insertionAxisWorld = None   # unit vector, world frame, insertion direction
        #                                 NOTE: assumed to be in Slicer RAS (mm) frame,
        #                                 matching the frame in which csv_base is expressed.

        self._lastDepthPublishTime = 0.0
        self._renderPending        = False

        self.depthState = NeedleDepthState()

        self._curvRows = {}   # needle_id -> {'kx', 'ky', 'base', 'tangent'} from CSV

        self.aligner = NeedleAlignment(
            ds=1.0,
            eps=1e-6,
            catheter_first_point_is_base=False
        )

        self.pubNeedlePose = None
        self.subShape      = None
        self.subDepth      = None

        self.needleFid              = None
        self.model                  = None
        self.insertionTransformNode = None

        self._vtkPts      = vtk.vtkPoints()
        self._vtkPoly     = vtk.vtkPolyLine()
        self._vtkCell     = vtk.vtkCellArray()
        self._vtkPolyData = vtk.vtkPolyData()
        self._vtkPolyData.SetPoints(self._vtkPts)
        self._nPtsAllocated = 0

        self.startROS()

    # ==================================================
    # ROS SETUP
    # ==================================================
    def startROS(self):

        rosLogic = slicer.util.getModuleLogic('ROS2')
        rosNode  = rosLogic.GetDefaultROS2Node()

        self.subShape = rosNode.CreateAndAddSubscriberNode(
            "PoseArray",
            "/needle/state/current_shape"
        )
        self.subDepth = rosNode.CreateAndAddSubscriberNode(
            "Double",
            "/needle/state/insertion_depth"
        )
        self.pubNeedlePose = rosNode.CreateAndAddPublisherNode(
            "PoseStamped",
            "/stage/state/needle_pose"
        )

        self.needleFid = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "NeedleFiducials"
        )
        self.needleFid.CreateDefaultDisplayNodes()

        self.model = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "NeedleCurveModel"
        )
        self.model.CreateDefaultDisplayNodes()

        # Both nodes observe NeedleInsertionTransform.
        # Polydata and fiducial control points are written in alignment-time
        # world coords; the transform translates them along the insertion axis
        # as depth changes. Both nodes move together atomically.
        self.insertionTransformNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", "NeedleInsertionTransform"
        )
        self.needleFid.SetAndObserveTransformNodeID(self.insertionTransformNode.GetID())
        self.model.SetAndObserveTransformNodeID(self.insertionTransformNode.GetID())

        self.subShape.AddObserver('ModifiedEvent', self.onShape)
        self.subDepth.AddObserver('ModifiedEvent', self.onDepth)

        self._poseTimer = qt.QTimer()
        self._poseTimer.setInterval(100)
        self._poseTimer.connect('timeout()', self._publishPoseTick)
        self._poseTimer.start()

        self._renderTimer = qt.QTimer()
        self._renderTimer.setInterval(50)   # 50 ms = 20 Hz
        self._renderTimer.connect('timeout()', self._renderShape)
        self._renderTimer.start()

    # ==================================================
    # DEPTH CALLBACK — primary pose publish path
    # ==================================================
    def onDepth(self, caller=None, event=None):
        """
        Fires on every new /needle/state/insertion_depth message (~20 Hz).

        Guarded by _alignMatrix — transform only fires once alignment is
        complete, preventing queued pre-alignment depth messages from
        producing a burst jump on the first post-alignment tick.

        Transform driven by raw ROS depth relative to alignmentDepth — not
        by depthState's processed value — to avoid state variance accumulation.

        Pose publish still requires state_initialized since depthState must
        be fully set up before export_state() is valid.
        """
        if self._alignMatrix is None:
            return

        depthMsg = self.subDepth.GetLastMessage()
        if depthMsg is None:
            return

        ros_depth_mm = float(depthMsg)

        # Transform driven directly by raw ROS depth — authoritative for visualization.
        depth_offset = ros_depth_mm - self.alignmentDepth
        self._setInsertionTransform(depth_offset)

        # Pose publish path — only once depthState is fully initialized.
        if self.depthState.state_initialized:
            self.depthState.insertion_depth = ros_depth_mm
            self.depthState.update_pose_from_depth()
            state = self.depthState.export_state()
            self._publishPose(state)
            self._lastDepthPublishTime = time.time()

        # Always show raw ROS value — not depthState's processed value.
        self.statusLabel.setText(f"Depth: {ros_depth_mm:.2f} mm")

    # ==================================================
    # SHAPE CALLBACK — data ingestion only
    # ==================================================
    def onShape(self, caller=None, event=None):
        """
        Fires on every ModifiedEvent on subShape (~20 Hz). Lightweight:
        extracts points into numpy caches only. No VTK mutations.

        Bridge applies ×1000 assuming ROS meters; SSN publishes mm.
        Divide by 1000 at read time to recover true mm values.

        worldPts are in alignment-time world coords — NeedleInsertionTransform
        handles live translation of both NeedleCurveModel and NeedleFiducials.
        """
        msg = self.subShape.GetLastMessage()
        if msg is None:
            return

        poses = msg.GetPoses()
        if not poses:
            return

        n = len(poses)

        bodyPts = np.empty((n, 3), dtype=np.float64)
        for i, p in enumerate(poses):
            bodyPts[i, 0] = p.GetElement(0, 3) / 1000.0
            bodyPts[i, 1] = p.GetElement(1, 3) / 1000.0
            bodyPts[i, 2] = p.GetElement(2, 3) / 1000.0

        self.latestBodyPts = bodyPts

        if self._alignMatrix is not None:
            worldPts = self._applyAlignMatrix(bodyPts)
        else:
            worldPts = bodyPts

        self._latestWorldPts = worldPts
        self._renderPending  = True

    # ==================================================
    # RENDER TIMER — VTK scene flush at display rate
    # ==================================================
    def _renderShape(self):
        """
        Render timer callback (20 Hz). Flushes _latestWorldPts to the VTK
        scene if onShape has flagged new data since the last render.
        Decouples ROS ModifiedEvent rate from VTK mutation rate.
        """
        if not self._renderPending or self._latestWorldPts is None:
            return
        self._renderPending = False
        worldPts = self._latestWorldPts
        self._updateModelPolyData(worldPts)
        self.numPointsLabel.setText(str(len(worldPts)))
        if len(worldPts) > 0:
            self.lastPointLabel.setText(f"{worldPts[-1]}")

    # ==================================================
    # ALIGNMENT PIPELINE
    # ==================================================
    def onAlign(self):

        try:

            if self.latestBodyPts is None:
                raise RuntimeError("No needle data yet — wait for /needle/state/current_shape")

            needle_id = self.needleIdCombo.currentText
            if not needle_id or needle_id not in self._curvRows:
                raise RuntimeError("Load CSV and select a needle ID before aligning")

            csv_base    = np.asarray(self._curvRows[needle_id]['base'],    dtype=np.float64)
            csv_tan     = np.asarray(self._curvRows[needle_id]['tangent'], dtype=np.float64)
            csv_tan_hat = csv_tan / np.linalg.norm(csv_tan)

            # Synthesise a 2-point world-frame reference from CSV so that
            # aligner.align has a properly-scaled axis to register against.
            # Arc-length of the current needle shape gives a realistic scale.
            needle_pts = self.latestBodyPts
            diffs      = np.diff(needle_pts, axis=0)
            arc_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
            if arc_length < 1.0:
                arc_length = 100.0   # safe fallback (mm) if shape is degenerate

            # Two-point reference: [tip, base] ordering matches
            # catheter_first_point_is_base=False in the aligner.
            ref_pts = np.array([
                csv_base + csv_tan_hat * arc_length,   # [0] = tip  (far end)
                csv_base,                              # [1] = base (skin entry)
            ], dtype=np.float64)

            vtk_transform     = self.aligner.align(needle_pts, ref_pts)
            self._alignMatrix = self._extractAlignMatrix(vtk_transform)

            self.insertionAxisWorld = csv_tan_hat
            skin_entry              = csv_base

            logging.info(
                f"onAlign: base/tangent from CSV needle_id='{needle_id}', "
                f"arc_length={arc_length:.1f} mm"
            )

            if not self.depthState.state_initialized:
                self.depthState.initialize_from_defaults(
                    skin_entry=skin_entry,
                    needle_pose_position=skin_entry,
                    needle_pose_orientation=self.insertionAxisWorld,
                    insertion_depth=0.0,
                )

            state = self.depthState.apply_alignment_measurement(
                reference_pts=ref_pts,
                updated_orientation=self.insertionAxisWorld,
            )

            # Depth: authoritative starting value comes from the ROS side.
            # Fall back to 0 — never bake in a stale depthState value.
            depthMsg = self.subDepth.GetLastMessage()
            if depthMsg is not None:
                self.alignmentDepth = float(depthMsg)
                logging.info(f"alignmentDepth seeded from ROS: {self.alignmentDepth:.2f} mm")
            else:
                self.alignmentDepth = 0.0
                logging.warning("No depth message at alignment time — alignmentDepth set to 0")

            # Reset transform to identity — both nodes start at alignment-time position.
            self._setInsertionTransform(0.0)

            # Recompute worldPts immediately so model and fiducials are consistent.
            freshWorldPts        = self._applyAlignMatrix(self.latestBodyPts)
            self._latestWorldPts = freshWorldPts
            self._renderPending  = True
            self._seedNeedleFid(freshWorldPts)

            self._publishPose(state)
            self._lastDepthPublishTime = time.time()

            self.statusLabel.setText(
                f"Aligned — CSV '{needle_id}' — depth={self.alignmentDepth:.2f} mm"
            )

        except Exception as e:
            logging.exception(e)
            self.statusLabel.setText(str(e))
            slicer.util.errorDisplay(str(e))

    # ==================================================
    # SAVE SHAPE
    # ==================================================
    def onSaveShape(self):
        """
        Freeze the current NeedleFiducials state into a new static markups
        node named <needle_id>_shape. Points are read via
        GetNthControlPointPositionWorld so the depth offset from
        NeedleInsertionTransform is baked in — reflecting true physical position.
        The frozen node does NOT observe NeedleInsertionTransform and will
        not move as depth changes subsequently.
        If a node with the same name already exists it is silently replaced.
        """
        try:
            needle_id = self.needleIdCombo.currentText
            if not needle_id:
                raise RuntimeError("Select a needle ID before saving shape")

            if self.needleFid is None:
                raise RuntimeError("NeedleFiducials node missing — align first")

            n = self.needleFid.GetNumberOfControlPoints()
            if n == 0:
                raise RuntimeError("NeedleFiducials is empty — align first")

            frozen_name = f"{needle_id}_shape"

            # Remove any pre-existing node with this name.
            existing = slicer.mrmlScene.GetFirstNodeByName(frozen_name)
            if existing is not None:
                slicer.mrmlScene.RemoveNode(existing)
                logging.info(f"onSaveShape: replaced existing node '{frozen_name}'")

            frozenNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", frozen_name
            )
            frozenNode.CreateDefaultDisplayNodes()
            # No transform applied — node stays fixed in world space permanently.

            for i in range(n):
                p = [0.0, 0.0, 0.0]
                self.needleFid.GetNthControlPointPositionWorld(i, p)
                frozenNode.AddControlPoint(p[0], p[1], p[2])

            logging.info(f"onSaveShape: saved {n} points to '{frozen_name}'")
            self.statusLabel.setText(f"Shape saved as '{frozen_name}' ({n} pts)")

        except Exception as e:
            logging.exception(e)
            self.statusLabel.setText(str(e))
            slicer.util.errorDisplay(str(e))

    # ==================================================
    # RESET ALIGNMENT
    # ==================================================
    def onResetAlignment(self):
        """
        Clear all alignment state so that onAlign must be re-run before
        onDepth resumes driving NeedleInsertionTransform.
        depthState is left intact — onAlign will call
        apply_alignment_measurement again on the next alignment.
        NeedleInsertionTransform is reset to identity so both
        NeedleCurveModel and NeedleFiducials return to their
        alignment-time positions until the next Align press.
        """
        self._alignMatrix       = None
        self.insertionAxisWorld = None
        self.alignmentDepth     = 0.0

        self._setInsertionTransform(0.0)

        logging.info("onResetAlignment: alignment state cleared")
        self.statusLabel.setText("Alignment reset — select needle ID and press Align")

    # ==================================================
    # EXPORT .needle
    # ==================================================
    def onExportNeedle(self):
        """
        Resample NeedleFiducials at uniform ds=1.0 mm and write to
        ~/MATLAB_dosimetry/<needle_id>.needle as a CSV with
        header x_mm_, y_mm_, z_mm_.
        Points are read via GetNthControlPointPositionWorld so the
        current depth offset is baked in — true physical position.
        """
        try:
            needle_id = self.needleIdCombo.currentText
            if not needle_id:
                raise RuntimeError("Select a needle ID before exporting")

            if self.needleFid is None:
                raise RuntimeError("NeedleFiducials node missing — align first")

            n = self.needleFid.GetNumberOfControlPoints()
            if n < 2:
                raise RuntimeError("NeedleFiducials needs at least 2 points to export")

            # Collect world-coord points (depth offset included).
            pts = []
            for i in range(n):
                p = [0.0, 0.0, 0.0]
                self.needleFid.GetNthControlPointPositionWorld(i, p)
                pts.append(p)
            pts = np.array(pts, dtype=np.float64)

            # Cumulative arclength.
            deltas         = np.diff(pts, axis=0)
            segLengths     = np.linalg.norm(deltas, axis=1)
            s              = np.concatenate([[0.0], np.cumsum(segLengths)])
            totalLength    = s[-1]

            # Resample at uniform ds=1.0 mm.
            ds      = 1.0
            sInterp = np.arange(0.0, totalLength + ds, ds)
            xInterp = np.interp(sInterp, s, pts[:, 0])
            yInterp = np.interp(sInterp, s, pts[:, 1])
            zInterp = np.interp(sInterp, s, pts[:, 2])
            ptsInterp = np.column_stack([xInterp, yInterp, zInterp])

            # Write file.
            outPath = os.path.expanduser(
                f"~/MATLAB_dosimetry/{needle_id}.needle"
            )
            os.makedirs(os.path.dirname(outPath), exist_ok=True)

            with open(outPath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x_mm_", "y_mm_", "z_mm_"])
                for p in ptsInterp:
                    writer.writerow(p)

            logging.info(
                f"onExportNeedle: wrote {len(ptsInterp)} pts "
                f"(total {totalLength:.1f} mm) to {outPath}"
            )
            self.statusLabel.setText(
                f"Exported '{needle_id}.needle' — "
                f"{len(ptsInterp)} pts, {totalLength:.1f} mm"
            )

        except Exception as e:
            logging.exception(e)
            self.statusLabel.setText(str(e))
            slicer.util.errorDisplay(str(e))

    # ==================================================
    # CURVATURE CSV
    # ==================================================
    def onLoadCurvatureCSV(self):
        """
        Read needle_poses.csv and populate the needle ID selector.
        Detects curvature columns by name pattern Kx_aaN / Ky_aaN.
        Also stores base position (base_x/y/z) and insertion tangent
        (tan_x/y/z) for each row so that onAlign can consume them directly.
        Both csv_base and csv_tan are assumed to be in Slicer RAS (mm) frame.
        """
        csvPath = self.csvPathEdit.text.strip()
        if not os.path.isfile(csvPath):
            slicer.util.errorDisplay(f"File not found:\n{csvPath}")
            return
        try:
            self._curvRows = {}
            with open(csvPath, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                kx_cols = sorted(
                    [c for c in fieldnames if re.match(r'Kx_aa\d+', c, re.IGNORECASE)],
                    key=lambda c: int(re.search(r'\d+', c).group())
                )
                ky_cols = sorted(
                    [c for c in fieldnames if re.match(r'Ky_aa\d+', c, re.IGNORECASE)],
                    key=lambda c: int(re.search(r'\d+', c).group())
                )
                if not kx_cols or not ky_cols:
                    raise ValueError("No Kx_aaN / Ky_aaN columns found in CSV.")
                if len(kx_cols) != len(ky_cols):
                    raise ValueError(
                        f"Mismatched curvature columns: {len(kx_cols)} Kx vs {len(ky_cols)} Ky."
                    )

                # Validate that pose columns are present.
                required_pose_cols = {"base_x", "base_y", "base_z", "tan_x", "tan_y", "tan_z"}
                missing = required_pose_cols - set(fieldnames)
                if missing:
                    raise ValueError(
                        f"CSV is missing required pose columns: {sorted(missing)}"
                    )

                for row in reader:
                    needle_id = row["needle_id"].strip()
                    self._curvRows[needle_id] = {
                        'kx':      [float(row[c]) for c in kx_cols],
                        'ky':      [float(row[c]) for c in ky_cols],
                        'base':    [float(row["base_x"]),
                                    float(row["base_y"]),
                                    float(row["base_z"])],
                        'tangent': [float(row["tan_x"]),
                                    float(row["tan_y"]),
                                    float(row["tan_z"])],
                    }

            self.needleIdCombo.clear()
            for nid in self._curvRows:
                self.needleIdCombo.addItem(nid)
            self.statusLabel.setText(
                f"Loaded {len(self._curvRows)} rows — {len(kx_cols)} AAs per row"
            )
        except Exception as e:
            logging.exception(e)
            slicer.util.errorDisplay(str(e))

    def onPublishCurvaturesExt(self):
        """
        1. Call /needle/curvatures/use_external to latch external mode.
        2. Publish one Float64MultiArray to /needle/state/curvatures_in.

        Data layout: [Kx_AA1, Ky_AA1, Kx_AA2, Ky_AA2, ...] — matches the
        ravel('F') convention that sub_curvatures_ext_callback expects on the
        ROS side (reshape to (2, numAAs) order='F' recovers X-row, Y-row).
        Re-press to update if the selected needle ID changes.
        """
        needle_id = self.needleIdCombo.currentText
        if not needle_id or needle_id not in self._curvRows:
            slicer.util.errorDisplay("Load CSV first and select a needle ID.")
            return
        try:
            # Step 1 — latch mode switch
            self._callTriggerService("/needle/curvatures/use_external")

            # Step 2 — pack curvature data
            kx   = self._curvRows[needle_id]['kx']
            ky   = self._curvRows[needle_id]['ky']
            data = []
            for x, y in zip(kx, ky):
                data.append(x)
                data.append(y)

            # Step 3 — publish via ros2 topic pub --once
            msg_str = json.dumps({"data": data})
            result  = self._ros2_cmd([
                "topic", "pub", "--once",
                "/needle/state/curvatures_in",
                "std_msgs/msg/Float64MultiArray",
                f"'{msg_str}'",
            ])
            if result.returncode != 0:
                raise RuntimeError(
                    f"Topic publish failed:\n{result.stderr}"
                )

            self.statusLabel.setText(
                f"Curvatures published: '{needle_id}' ({len(kx)} AAs) — external mode latched"
            )

        except Exception as e:
            logging.exception(e)
            slicer.util.errorDisplay(str(e))

    def onRevertToFBG(self):
        """
        Call /needle/curvatures/use_fbg to hand control back to the
        FBG pipeline. Use after Slicer-driven curvature session ends,
        or after an unclean disconnect.
        """
        try:
            self._callTriggerService("/needle/curvatures/use_fbg")
            self.statusLabel.setText("Curvature source reverted to FBG pipeline")
        except Exception as e:
            logging.exception(e)
            slicer.util.errorDisplay(str(e))

    # ==================================================
    # KEEPALIVE TIMER
    # ==================================================
    def _publishPoseTick(self):
        """
        Keepalive: fires every 100 ms but publishes only if onDepth has been
        silent for > 80 ms. Suppressed whenever onDepth is actively firing.
        """
        if not self.depthState.state_initialized:
            return
        if time.time() - self._lastDepthPublishTime < 0.08:
            return
        state = self.depthState.export_state()
        self._publishPose(state)

    # ==================================================
    # HELPERS
    # ==================================================
    def _seedNeedleFid(self, worldPts):
        """
        Write worldPts into NeedleFiducials at alignment time, in
        alignment-time world coords. Called once by onAlign.
        NeedleInsertionTransform then moves NeedleFiducials live
        in sync with NeedleCurveModel as depth changes.
        """
        self.needleFid.RemoveAllControlPoints()
        for pt in worldPts:
            self.needleFid.AddControlPoint(float(pt[0]), float(pt[1]), float(pt[2]))

    def _ros2_cmd(self, ros2_args):
        """
        Wrap a ros2 CLI invocation in a clean bash shell that sources the
        ROS 2 setup before executing. This bypasses any PYTHONPATH/PYTHONHOME
        conflicts introduced by Slicer's embedded Python environment.

        ros2_args : list of str, e.g.
            ['service', 'call', '/my/srv', 'std_srvs/srv/Trigger', '{}']
        Returns a subprocess.CompletedProcess.
        """
        ros2_setup = "/opt/ros/humble/setup.bash"
        cmd_str    = " ".join(ros2_args)
        bash_cmd   = (
            "unset PYTHONPATH PYTHONHOME PYTHONSTARTUP "
            "PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE && "
            f"source {ros2_setup} && "
            f"ros2 {cmd_str}"
        )
        return subprocess.run(
            ["bash", "-c", bash_cmd],
            capture_output=True, text=True, timeout=10.0,
        )

    def _callTriggerService(self, service_name):
        """
        Call a std_srvs/srv/Trigger service via subprocess.
        Sources /opt/ros/humble/setup.bash inside bash to ensure the ros2
        CLI uses the system Python that ROS 2 was built against, not Slicer's.
        Raises RuntimeError on non-zero exit or timeout.
        """
        result = self._ros2_cmd([
            "service", "call",
            service_name,
            "std_srvs/srv/Trigger",
            "'{}'",
        ])
        if result.returncode != 0:
            raise RuntimeError(
                f"Service call to {service_name} failed:\n{result.stderr}"
            )
        logging.info(f"Service {service_name}: {result.stdout.strip()}")

    def _extractAlignMatrix(self, vtk_transform):
        """
        Extract a vtkAbstractTransform's 4×4 matrix into numpy.
        Called ONCE at alignment time — never in any hot path.
        """
        m = vtk.vtkMatrix4x4()
        vtk_transform.GetMatrix(m)
        return np.array(
            [[m.GetElement(i, j) for j in range(4)] for i in range(4)],
            dtype=np.float64
        )

    def _applyAlignMatrix(self, pts):
        """
        Apply the cached (4,4) alignment matrix to an (N,3) point array.
        Returns alignment-time world coords — depth offset NOT included.
        NeedleInsertionTransform handles live translation of both nodes.
        """
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
        return (self._alignMatrix @ pts_h.T).T[:, :3]

    def _updateModelPolyData(self, worldPts):
        """
        Update model polydata in-place. VTK objects pre-allocated at setup();
        only mutates contents and calls Modified().
        """
        n = len(worldPts)

        if n != self._nPtsAllocated:
            self._vtkPts.SetNumberOfPoints(n)
            self._vtkPoly.GetPointIds().SetNumberOfIds(n)
            for i in range(n):
                self._vtkPoly.GetPointIds().SetId(i, i)
            self._nPtsAllocated = n

        for i in range(n):
            self._vtkPts.SetPoint(
                i,
                float(worldPts[i, 0]),
                float(worldPts[i, 1]),
                float(worldPts[i, 2])
            )

        self._vtkPts.Modified()

        self._vtkCell.Reset()
        self._vtkCell.InsertNextCell(self._vtkPoly)

        self._vtkPolyData.SetLines(self._vtkCell)
        self._vtkPolyData.Modified()

        if self.model.GetPolyData() is not self._vtkPolyData:
            self.model.SetAndObservePolyData(self._vtkPolyData)

    def _setInsertionTransform(self, depth_offset_mm):
        """
        Translate NeedleInsertionTransform by depth_offset_mm along insertionAxisWorld.
        Both NeedleCurveModel and NeedleFiducials observe this node — Slicer
        propagates the translation to both simultaneously.
        Called by: onDepth (every depth message), onAlign (reset to 0),
        onResetAlignment (reset to 0).
        """
        if self.insertionTransformNode is None or self.insertionAxisWorld is None:
            return

        t = depth_offset_mm * self.insertionAxisWorld

        m = vtk.vtkMatrix4x4()
        m.Identity()
        m.SetElement(0, 3, t[0])
        m.SetElement(1, 3, t[1])
        m.SetElement(2, 3, t[2])

        self.insertionTransformNode.SetMatrixTransformToParent(m)

    # ==================================================
    # PUBLISH — single path
    # ==================================================
    def _publishPose(self, state):
        """
        Sole publish function. Called by: onDepth, _publishPoseTick, onAlign.
        Bridge divides by 1000 to produce ROS meters — pre-multiply by 1000
        to express in bridge's expected Slicer units (mm).
        """
        if self.pubNeedlePose is None:
            return

        pose_position = state["needle_pose_position"]
        if pose_position is None:
            return

        pose_msg = self.pubNeedlePose.GetBlankMessage()
        pose_mat = pose_msg.GetPose()
        pose_mat.Identity()

        pose_mat.SetElement(0, 3, float(pose_position[0]) * 1000.0)
        pose_mat.SetElement(1, 3, float(pose_position[1]) * 1000.0)
        pose_mat.SetElement(2, 3, float(pose_position[2]) * 1000.0)

        self.pubNeedlePose.Publish(pose_msg)