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

#Custom helper alignment.py
from logic.alignment import NeedleAlignment

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
        self.aligner = NeedleAlignment(
            ds=1.0,
            eps=1e-6,
            catheter_first_point_is_base=False
        )

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

            cat_pts = []

            for i in range(
                catNode.GetNumberOfControlPoints()
            ):

                p = [0, 0, 0]

                catNode.GetNthControlPointPositionWorld(
                    i,
                    p
                )

                cat_pts.append(p)

            vtk_transform = self.aligner.align(
                needle_pts,
                cat_pts
            )

            transformNode = (
                slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLinearTransformNode",
                    "Needle_to_Catheter"
                )
            )

            transformNode.SetMatrixTransformToParent(
                vtk_transform.GetMatrix()
            )

            self.needleFid.SetAndObserveTransformNodeID(
                transformNode.GetID()
            )

            self.statusLabel.setText(
                "Alignment complete"
            )

        except Exception as e:

            logging.exception(e)

            self.statusLabel.setText(str(e))

            slicer.util.errorDisplay(str(e))
