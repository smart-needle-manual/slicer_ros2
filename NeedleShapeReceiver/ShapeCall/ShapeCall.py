import logging
import qt
import slicer
import vtk
import numpy as np
import time

from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _
from slicer.i18n import translate

from logic_games.alignment import NeedleAlignment
from logic_games.curvature import NeedleCurvature
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
        self.statusLabel = qt.QLabel("Ready")
        self.numPointsLabel = qt.QLabel("0")
        self.lastPointLabel = qt.QLabel("N/A")

        info = qt.QGroupBox("Needle Info")
        layout = qt.QFormLayout(info)
        layout.addRow("Points:", self.numPointsLabel)
        layout.addRow("Last:", self.lastPointLabel)
        layout.addRow("Status:", self.statusLabel)

        self.layout.addWidget(info)

        # ---------------- NODE SELECTOR ----------------
        self.catheterSelector = slicer.qMRMLNodeComboBox()
        self.catheterSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.catheterSelector.setMRMLScene(slicer.mrmlScene)
        self.catheterSelector.noneEnabled = False

        # ---------------- ALIGN BUTTON ----------------
        self.alignButton = qt.QPushButton("Align (Full Base/Tip Model)")
        self.alignButton.connect("clicked(bool)", self.onAlign)

        alignBox = qt.QGroupBox("Alignment")
        alignLayout = qt.QFormLayout(alignBox)
        alignLayout.addRow("Catheter:", self.catheterSelector)
        alignLayout.addRow(self.alignButton)

        self.layout.addWidget(alignBox)
        self.layout.addStretch(1)

        self.curvatureButton = qt.QPushButton("Compute Curvature Table")
        self.curvatureButton.connect("clicked(bool)", self.onComputeCurvature)
        alignLayout.addRow(self.curvatureButton)

        # ---------------- STATE ----------------
        self.latestNeedle = None        # raw body-frame points (mm)
        self.alignTransform = None      # vtkLandmarkTransform (body -> world)
        self.alignmentDepth = 0.0       # insertion_depth at alignment time (mm)
        self.insertionAxisWorld = None  # unit insertion axis in world frame
        self.lastUpdate = 0
        self.throttle = 0.15

        self.depthState = NeedleDepthState()

        self.aligner = NeedleAlignment(
            ds=1.0,
            eps=1e-6,
            catheter_first_point_is_base=False
        )
        self.curvatureCalculator = NeedleCurvature(
            curv_smooth_window=7
        )

        self.pubNeedlePose = None
        self.subShape = None
        self.subDepth = None
        self.needleFid = None
        self.model = None
        self.insertionTransformNode = None

        self.startROS()

    # ==================================================
    # ROS
    # ==================================================
    def startROS(self):

        rosLogic = slicer.util.getModuleLogic('ROS2')
        rosNode = rosLogic.GetDefaultROS2Node()

        self.subShape = rosNode.CreateAndAddSubscriberNode(
            "PoseArray",
            "/needle/state/current_shape"
        )

        self.subDepth = rosNode.CreateAndAddSubscriberNode(
            "Double",
            "/needle/state/insertion_depth"
        )

        # Publish to needle_pose_in — the TopicRepeater's input topic.
        # The repeater (wait_for_input=True) wakes on the first message here
        # and then continuously re-broadcasts to /stage/state/needle_pose.
        # Nothing else publishes to needle_pose, so there is no competition.
        self.pubNeedlePose = rosNode.CreateAndAddPublisherNode(
            "PoseStamped",
            "/stage/state/needle_pose_in"
        )

        self.needleFid = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode",
            "NeedleFiducials"
        )
        self.needleFid.CreateDefaultDisplayNodes()

        self.model = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode",
            "NeedleCurveModel"
        )
        self.model.CreateDefaultDisplayNodes()

        # Transform node that carries the cumulative insertion translation.
        # Both needleFid and model observe it so Slicer handles the math.
        self.insertionTransformNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode",
            "NeedleInsertionTransform"
        )
        self.needleFid.SetAndObserveTransformNodeID(
            self.insertionTransformNode.GetID()
        )
        self.model.SetAndObserveTransformNodeID(
            self.insertionTransformNode.GetID()
        )

        self.subShape.AddObserver('ModifiedEvent', self.onROS)

        # Publish needle_pose_in at 10 Hz so the TopicRepeater continuously
        # receives the latest depth after alignment and re-broadcasts it.
        # Before alignment depthState.state_initialized is False so nothing
        # is sent — the one-shot seed in sim_needle.launch.py covers that gap.
        self._poseTimer = qt.QTimer()
        self._poseTimer.setInterval(100)  # 100 ms = 10 Hz
        self._poseTimer.connect('timeout()', self._publishPoseTick)
        self._poseTimer.start()

    # ==================================================
    # FAST STREAM
    # ==================================================
    def onROS(self, caller=None, event=None):

        now = time.time()
        if now - self.lastUpdate < self.throttle:
            return
        self.lastUpdate = now

        msg = self.subShape.GetLastMessage()
        if msg is None:
            return

        poses = msg.GetPoses()
        if not poses:
            return

        # --- Piece 1: ROS body-frame points, meters -> mm ---
        bodyPts = []
        for p in poses:
            bodyPts.append([
                p.GetElement(0, 3) / 1000.0,
                p.GetElement(1, 3) / 1000.0,
                p.GetElement(2, 3) / 1000.0,
            ])

        self.latestNeedle = np.array(bodyPts, dtype=float)

        # --- Piece 2: apply alignment transform to get world-frame points ---
        if self.alignTransform is not None:
            worldPts = self._apply_vtk_transform(
                self.alignTransform,
                self.latestNeedle
            )
        else:
            worldPts = self.latestNeedle

        # --- Piece 3: ROS insertion_depth drives transform and needle_pose ---
        if self.depthState.state_initialized:
            depthMsg = self.subDepth.GetLastMessage()
            if depthMsg is not None:
                ros_depth_mm = float(depthMsg)
                self.depthState.insertion_depth = ros_depth_mm
                self.depthState.update_pose_from_depth()
                state = self.depthState.export_state()
                self.publishNeedleState(state)
                self.statusLabel.setText(
                    f"Depth: {ros_depth_mm:.2f} mm"
                )

            # Update the Slicer insertion transform with the cumulative
            # depth offset since alignment. All visualization follows.
            depth_offset = (
                self.depthState.insertion_depth - self.alignmentDepth
            )
            self._set_insertion_transform(depth_offset)

        # Write world-frame (alignment) points into the nodes once per frame.
        # NeedleInsertionTransform carries all positional updates.
        pts = vtk.vtkPoints()
        poly = vtk.vtkPolyLine()
        poly.GetPointIds().SetNumberOfIds(len(worldPts))

        self.needleFid.RemoveAllControlPoints()

        for i, pt in enumerate(worldPts):
            pts.InsertNextPoint(pt[0], pt[1], pt[2])
            poly.GetPointIds().SetId(i, i)
            self.needleFid.AddControlPoint(pt[0], pt[1], pt[2])

        cell = vtk.vtkCellArray()
        cell.InsertNextCell(poly)

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(pts)
        polyData.SetLines(cell)

        self.model.SetAndObservePolyData(polyData)

        self.numPointsLabel.setText(str(len(worldPts)))
        self.lastPointLabel.setText(f"{worldPts[-1]}")

    # ==================================================
    # ALIGNMENT PIPELINE
    # ==================================================
    def onAlign(self):

        try:

            if self.latestNeedle is None:
                raise RuntimeError("No needle data yet")

            needle_pts = self.latestNeedle
            catNode = self.catheterSelector.currentNode()

            if catNode is None:
                raise RuntimeError("Select catheter")

            cat_pts = []
            for i in range(catNode.GetNumberOfControlPoints()):
                p = [0, 0, 0]
                catNode.GetNthControlPointPositionWorld(i, p)
                cat_pts.append(p)

            cat_pts = np.asarray(cat_pts, dtype=float)

            if len(cat_pts) < 2:
                raise RuntimeError("Catheter/reference list too small")

            # Compute and store the alignment transform (body -> world).
            vtk_transform = self.aligner.align(needle_pts, cat_pts)
            self.alignTransform = vtk_transform

            ref_axis = (
                np.asarray(cat_pts[-1], dtype=float)
                - np.asarray(cat_pts[0], dtype=float)
            )

            # Store normalized world-frame insertion axis for translation.
            self.insertionAxisWorld = ref_axis / np.linalg.norm(ref_axis)

            if not self.depthState.state_initialized:
                self.depthState.initialize_from_defaults(
                    skin_entry=np.asarray(cat_pts[0], dtype=float),
                    needle_pose_position=np.asarray(cat_pts[0], dtype=float),
                    needle_pose_orientation=ref_axis,
                    insertion_depth=0.0,
                )

            # Set insertion_depth from fiducial arclength and reset the
            # shape arclength baseline for post-alignment tracking.
            state = self.depthState.apply_alignment_measurement(
                reference_pts=cat_pts,
                updated_orientation=ref_axis,
            )

            # Seed alignmentDepth from the live ROS depth so depth_offset
            # starts at zero on the first post-alignment frame. Fall back to
            # fiducial arclength only if the topic has no message yet.
            depthMsg = self.subDepth.GetLastMessage()
            if depthMsg is not None:
                self.alignmentDepth = float(depthMsg)
            else:
                self.alignmentDepth = state["insertion_depth"]

            # Reset the insertion transform to identity at alignment time.
            self._set_insertion_transform(0.0)

            self.publishNeedleState(state)
            self.statusLabel.setText(
                f"Alignment complete; depth={state['insertion_depth']:.2f} mm"
            )

        except Exception as e:
            logging.exception(e)
            self.statusLabel.setText(str(e))
            slicer.util.errorDisplay(str(e))

    # ==================================================
    # CURVATURE
    # ==================================================
    def onComputeCurvature(self):

        try:

            needleNode = self.needleFid

            if needleNode is None:
                raise RuntimeError("NeedleFiducials missing")

            pts = []
            for i in range(needleNode.GetNumberOfControlPoints()):
                p = [0, 0, 0]
                needleNode.GetNthControlPointPositionWorld(i, p)
                pts.append(p)

            result = self.curvatureCalculator.compute_curvature(pts)

            s = result["arclength"]
            k = result["curvature"]
            t = result["tangent"]

            tableNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLTableNode",
                "NeedleCurvatureTable"
            )
            table = tableNode.GetTable()

            arr_s = vtk.vtkDoubleArray(); arr_s.SetName("s")
            arr_k = vtk.vtkDoubleArray(); arr_k.SetName("curvature")
            arr_tx = vtk.vtkDoubleArray(); arr_tx.SetName("tx")
            arr_ty = vtk.vtkDoubleArray(); arr_ty.SetName("ty")
            arr_tz = vtk.vtkDoubleArray(); arr_tz.SetName("tz")

            for i in range(len(s)):
                arr_s.InsertNextValue(s[i])
                arr_k.InsertNextValue(k[i])
                arr_tx.InsertNextValue(t[i, 0])
                arr_ty.InsertNextValue(t[i, 1])
                arr_tz.InsertNextValue(t[i, 2])

            table.AddColumn(arr_s)
            table.AddColumn(arr_k)
            table.AddColumn(arr_tx)
            table.AddColumn(arr_ty)
            table.AddColumn(arr_tz)

            self.statusLabel.setText("Curvature table created")

        except Exception as e:
            logging.exception(e)
            slicer.util.errorDisplay(str(e))
            self.statusLabel.setText(str(e))

    # ==================================================
    # HELPERS
    # ==================================================
    def _apply_vtk_transform(self, transform, pts):
        """Apply a vtkAbstractTransform to an (N,3) numpy array."""
        out = np.zeros_like(pts)
        for i, pt in enumerate(pts):
            p_out = [0.0, 0.0, 0.0]
            transform.TransformPoint(
                [float(pt[0]), float(pt[1]), float(pt[2])],
                p_out
            )
            out[i] = p_out
        return out

    def _set_insertion_transform(self, depth_offset_mm):
        """Update NeedleInsertionTransform with a translation along the
        insertion axis by depth_offset_mm (mm, Slicer world frame)."""
        if self.insertionTransformNode is None:
            return
        if self.insertionAxisWorld is None:
            return

        t = -depth_offset_mm * self.insertionAxisWorld

        m = vtk.vtkMatrix4x4()
        m.Identity()
        m.SetElement(0, 3, t[0])
        m.SetElement(1, 3, t[1])
        m.SetElement(2, 3, t[2])

        self.insertionTransformNode.SetMatrixTransformToParent(m)

    # ==================================================
    # PUBLISH
    # ==================================================
    def publishNeedleState(self, state):

        if self.pubNeedlePose is None:
            return

        pose_position = state["needle_pose_position"]
        if pose_position is None:
            return

        pose_msg = self.pubNeedlePose.GetBlankMessage()
        pose_mat = pose_msg.GetPose()
        pose_mat.Identity()

        # Internal units are mm; ROS expects meters, so multiply by 1000.
        pose_mat.SetElement(0, 3, float(pose_position[0]) * 1000.0)
        pose_mat.SetElement(1, 3, float(pose_position[1]) * 1000.0)
        pose_mat.SetElement(2, 3, float(pose_position[2]) * 1000.0)

        self.pubNeedlePose.Publish(pose_msg)

    def _publishPoseTick(self):
        """Timer callback: publish current needle pose at 10 Hz to needle_pose_in.
        The TopicRepeater latches each value and re-broadcasts to needle_pose,
        ensuring ShapeSensingNeedleNode always sees the latest depth."""
        if not self.depthState.state_initialized:
            return
        state = self.depthState.export_state()
        self.publishNeedleState(state)