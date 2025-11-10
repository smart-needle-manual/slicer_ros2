import logging
import ctk
import qt
from qt import QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
import time
import vtk


#
# ShapeCall Module
#

class ShapeCall(ScriptedLoadableModule):
    """Scripted Loadable Module for real-time needle visualization via ROS2."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ShapeCall")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Needle Visualization")]
        self.parent.dependencies = []
        self.parent.contributors = ["Rajdeep Banerjee (Brigham & Women's Hospital)"]
        self.parent.helpText = _("This module subscribes to incoming FBG needle shape data using Slicer-ROS.")
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = _(
            "Originally developed by Jean-Christophe Fillion-Robin, Kitware Inc. and Steve Pieper, Isomics, Inc. "
            "Project is funded by NIH 1R01EB036015-01 with following contributors: "
            "Junichi Tokuda, Mariana Bernardes, Rajdeep Banerjee, Robert Cormack (BWH); "
            "Iulian Iordachita, Jacynthe Francoeur, Yinsong Ma, Dimitri Lezcano, Jin Seob Kim (JHU)."
        )


#
# ShapeCall Widget
#

class ShapeCallWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Info panel for needle shape
        infoGroupBox = qt.QGroupBox("Needle Shape Info")
        infoLayout = qt.QFormLayout(infoGroupBox)

        self.numPointsLabel = qt.QLabel("Number of points: 0")
        self.lastPointLabel = qt.QLabel("Last point: (N/A, N/A, N/A)")

        # Needle param file label
        self.paramFileLabel = qt.QLabel(
            "Needle param: 3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json"
        )

        # Needle info table
        self.dsInfoTable = QTableWidget()
        self.dsInfoTable.setRowCount(6)
        self.dsInfoTable.setColumnCount(1)
        self.dsInfoTable.setVerticalHeaderLabels(
            ["ds (mm)", "Length (mm)", "AA1 (mm)", "AA2 (mm)", "AA3 (mm)", "AA4 (mm)"]
        )
        self.dsInfoTable.setItem(0, 0, QTableWidgetItem("1.0"))
        self.dsInfoTable.setItem(1, 0, QTableWidgetItem("200"))
        self.dsInfoTable.setItem(2, 0, QTableWidgetItem("35"))
        self.dsInfoTable.setItem(3, 0, QTableWidgetItem("65"))
        self.dsInfoTable.setItem(4, 0, QTableWidgetItem("100"))
        self.dsInfoTable.setItem(5, 0, QTableWidgetItem("135"))
        self.dsInfoTable.setEditTriggers(QTableWidget.NoEditTriggers)
        self.dsInfoTable.resizeColumnsToContents()
        self.dsInfoTable.resizeRowsToContents()
        self.dsInfoTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.dsInfoTable.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        infoLayout.addRow(self.numPointsLabel)
        infoLayout.addRow(self.lastPointLabel)
        infoLayout.addRow(self.paramFileLabel)
        infoLayout.addRow("Needle Info:", self.dsInfoTable)
        self.layout.addWidget(infoGroupBox)

        # Vertical spacer
        self.layout.addStretch(1)

        # Start ROS2 subscription automatically
        self.startSubscriber()

    #### BEGIN ROS2 ADDITION ####
    def startSubscriber(self):
        """Set up ROS2 subscription and visualization nodes."""
        rosLogic = slicer.util.getModuleLogic('ROS2')
        rosNode = rosLogic.GetDefaultROS2Node()

        topic_name = '/needle/state/current_shape'
        msg_type = 'PoseArray'

        self.subShape = rosNode.CreateAndAddSubscriberNode(msg_type, topic_name)
        if self.subShape is None:
            raise RuntimeError(f"Failed to create subscriber for {topic_name}")
        print(f"Subscribed to {topic_name}")

        # Create fiducial node for reference points
        self.fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "NeedleFiducials")
        if self.fid.GetDisplayNode() is None:
            self.fid.CreateDefaultDisplayNodes()
        self.fid.GetDisplayNode().SetPointLabelsVisibility(False)
        self.fid.GetDisplayNode().SetSelectedColor(1, 0, 0)  # Red fiducials

        # Create MRML model node for the actual curve visualization
        self.curveModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "NeedleCurveModel")
        if self.curveModel.GetDisplayNode() is None:
            self.curveModel.CreateDefaultDisplayNodes()

        modelDisplay = self.curveModel.GetDisplayNode()
        modelDisplay.SetColor(0.0, 1.0, 0.0)  # Green curve
        modelDisplay.SetLineWidth(3.0)
        modelDisplay.SetVisibility(True)

        self.THROTTLE_INTERVAL = 2
        self.lastUpdateTime = 0

        def updateFiducialsAndModel(caller=None, event=None):
            currentTime = time.time()
            if currentTime - self.lastUpdateTime < self.THROTTLE_INTERVAL:
                return
            self.lastUpdateTime = currentTime

            poseArray = self.subShape.GetLastMessage()
            if poseArray is None:
                return
            poses = poseArray.GetPoses()
            if not poses:
                return

            # Prepare VTK structures
            points = vtk.vtkPoints()
            polyLine = vtk.vtkPolyLine()
            polyLine.GetPointIds().SetNumberOfIds(len(poses))

            self.fid.RemoveAllControlPoints()

            for i, pose in enumerate(poses):
                x = pose.GetElement(0, 3) / 1000.0
                y = pose.GetElement(1, 3) / 1000.0
                z = pose.GetElement(2, 3) / 1000.0
                points.InsertNextPoint(x, y, z)
                polyLine.GetPointIds().SetId(i, i)
                self.fid.AddControlPoint(x, y, z)

            # Construct PolyData for the curve
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(polyLine)
            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(cells)
            polyData.Modified()

            # Update the model node
            self.curveModel.SetAndObservePolyData(polyData)

            # Update info labels
            numPoints = len(poses)
            lastPose = poses[-1]
            lastPoint = (
                lastPose.GetElement(0, 3) / 1000.0,
                lastPose.GetElement(1, 3) / 1000.0,
                lastPose.GetElement(2, 3) / 1000.0
            )

            def updateLabels():
                self.numPointsLabel.setText(f"Number of points: {numPoints}")
                self.lastPointLabel.setText(
                    f"Last point: ({lastPoint[0]:.2f}, {lastPoint[1]:.2f}, {lastPoint[2]:.2f})"
                )

            qt.QTimer.singleShot(0, updateLabels)

        # Add observer for incoming ROS messages
        self.subShape.AddObserver('ModifiedEvent', updateFiducialsAndModel)
        print("Throttled model + fiducial visualization active.")
    #### END ROS2 ADDITION ####
