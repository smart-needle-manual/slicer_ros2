import logging
import qt
import slicer
import vtk
import numpy as np
import time

from qt import QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy
from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _
from slicer.i18n import translate


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

        # ---------------- STATE ----------------
        self.latestNeedle = None
        self.lastUpdate = 0
        self.throttle = 0.15

        # EXACT PARAMETER FROM YOUR SCRIPT
        self.DS = 1.0
        self.EPS = 1e-6
        self.CATheter_FIRST_POINT_IS_BASE = False

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

        self.subShape.AddObserver('ModifiedEvent', self.onROS)

    # ==================================================
    # FAST STREAM (NO HEAVY COMPUTATION)
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

        pts = vtk.vtkPoints()
        poly = vtk.vtkPolyLine()
        poly.GetPointIds().SetNumberOfIds(len(poses))

        npPts = []

        self.needleFid.RemoveAllControlPoints()

        for i, p in enumerate(poses):

            x = p.GetElement(0, 3) / 1000.0
            y = p.GetElement(1, 3) / 1000.0
            z = p.GetElement(2, 3) / 1000.0

            pts.InsertNextPoint(x, y, z)
            poly.GetPointIds().SetId(i, i)

            self.needleFid.AddControlPoint(x, y, z)
            npPts.append([x, y, z])

        self.latestNeedle = np.array(npPts)

        cell = vtk.vtkCellArray()
        cell.InsertNextCell(poly)

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(pts)
        polyData.SetLines(cell)

        self.model.SetAndObservePolyData(polyData)

        last = npPts[-1]
        self.numPointsLabel.setText(str(len(npPts)))
        self.lastPointLabel.setText(f"{last}")

    # ==================================================
    # EXACT ALIGNMENT PIPELINE (YOUR ORIGINAL LOGIC)
    # ==================================================
    def onAlign(self):

        try:
            if self.latestNeedle is None:
                raise RuntimeError("No needle data yet")

            needle_pts = self.latestNeedle

            catNode = self.catheterSelector.currentNode()
            if catNode is None:
                raise RuntimeError("Select catheter")

            # ---------------- GET CATETER POINTS ----------------
            cat_pts = []
            for i in range(catNode.GetNumberOfControlPoints()):
                p = [0, 0, 0]
                catNode.GetNthControlPointPositionWorld(i, p)
                cat_pts.append(p)

            cat_pts = np.array(cat_pts, dtype=float)

            if len(cat_pts) < 2:
                raise RuntimeError("Catheter too small")

            # ==================================================
            # STEP 2: REORIENT BASE -> TIP
            # ==================================================
            if self.CATheter_FIRST_POINT_IS_BASE:
                cat_reoriented = cat_pts.copy()
            else:
                cat_reoriented = cat_pts[::-1].copy()

            # ==================================================
            # STEP 3: RESAMPLE
            # ==================================================
            def resample(points, ds):
                seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
                arc = np.insert(np.cumsum(seg), 0, 0.0)

                if arc[-1] < self.EPS:
                    raise RuntimeError("Curve too small")

                s = np.arange(0.0, arc[-1], ds)
                if len(s) == 0 or abs(s[-1] - arc[-1]) > self.EPS:
                    s = np.append(s, arc[-1])

                return np.vstack([
                    np.interp(s, arc, points[:, 0]),
                    np.interp(s, arc, points[:, 1]),
                    np.interp(s, arc, points[:, 2]),
                ]).T

            cat_resampled = resample(cat_reoriented, self.DS)

            # ==================================================
            # STEP 4: LENGTHS
            # ==================================================
            L_needle = np.sum(np.linalg.norm(np.diff(needle_pts, axis=0), axis=1))
            L_cat = np.sum(np.linalg.norm(np.diff(cat_resampled, axis=0), axis=1))

            L_ext = L_needle - L_cat

            if L_ext < -self.EPS:
                raise RuntimeError("Catheter already longer than needle")

            n_ext = int(round(max(L_ext, 0.0) / self.DS))

            # ==================================================
            # STEP 5: EXTENSION BEFORE BASE
            # ==================================================
            if n_ext > 0:
                base = cat_resampled[0]
                direction = (cat_resampled[0] - cat_resampled[1])
                direction = direction / np.linalg.norm(direction)

                ext_pts = np.array([
                    base + direction * ((n_ext - i) * self.DS)
                    for i in range(n_ext)
                ])

                cat_extended = np.vstack([ext_pts, cat_resampled])
            else:
                cat_extended = cat_resampled

            # ==================================================
            # STEP 6: ALIGNMENT
            # ==================================================
            n_align = min(len(needle_pts), len(cat_extended))
            if n_align < 3:
                raise RuntimeError("Not enough alignment points")

            source = vtk.vtkPoints()
            target = vtk.vtkPoints()

            for i in range(n_align):
                source.InsertNextPoint(*needle_pts[i])
                target.InsertNextPoint(*cat_extended[i])

            t = vtk.vtkLandmarkTransform()
            t.SetSourceLandmarks(source)
            t.SetTargetLandmarks(target)
            t.SetModeToRigidBody()
            t.Update()

            transform = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode",
                "Needle_to_Catheter"
            )

            transform.SetMatrixTransformToParent(t.GetMatrix())

            needleNode = slicer.util.getNode("NeedleFiducials")
            needleNode.SetAndObserveTransformNodeID(transform.GetID())

            self.statusLabel.setText("Alignment complete")

        except Exception as e:
            logging.exception(e)
            self.statusLabel.setText(str(e))
            slicer.util.errorDisplay(str(e))
