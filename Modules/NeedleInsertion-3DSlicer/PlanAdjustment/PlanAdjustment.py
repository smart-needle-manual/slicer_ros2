import logging
import os
import sys
import subprocess
from typing import Annotated, Optional

import vtk
import numpy as np
from matplotlib import pyplot as plt

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLMarkupsFiducialNode
from slicer import vtkMRMLSegmentationNode

# Try to import path_planning functions, install dependencies if needed
try:
    # Add the path_planning directory to Python path
    current_dir = os.path.dirname(__file__)
    path_planning_dir = os.path.join(current_dir, "path_planning")
    if os.path.exists(path_planning_dir):
        sys.path.insert(0, path_planning_dir)

    from path_planning.val import load_reg
    from path_planning.elastic import find_entry
    from path_planning.elastic import plan_path
    PATH_PLANNING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import path_planning functions: {e}")
    PATH_PLANNING_AVAILABLE = False


def install_required_packages():
    """Install required packages for path planning functionality."""
    try:
        # Common packages that might be needed for machine learning models
        required_packages = ['scikit-learn', 'joblib', 'pandas']
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logging.info(f"Package {package} already installed")
            except ImportError:
                logging.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logging.info(f"Successfully installed {package}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to install required packages: {e}")
        return False


#
# PlanAdjustment
#


class PlanAdjustment(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PlanAdjustment")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#PlanAdjustment">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # PlanAdjustment1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PlanAdjustment",
        sampleName="PlanAdjustment1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "PlanAdjustment1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="PlanAdjustment1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="PlanAdjustment1",
    )

    # PlanAdjustment2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PlanAdjustment",
        sampleName="PlanAdjustment2",
        thumbnailFileName=os.path.join(iconsPath, "PlanAdjustment2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="PlanAdjustment2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="PlanAdjustment2",
    )


#
# PlanAdjustmentParameterNode
#


@parameterNodeWrapper
class PlanAdjustmentParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    markupFiducials: vtkMRMLMarkupsFiducialNode
    roiRLDimension: Annotated[float, WithinRange(0, 200)] = 100
    roiAPDimension: Annotated[float, WithinRange(0, 200)] = 100
    outputVolume: vtkMRMLScalarVolumeNode
    autoUpdate: bool = True
    segmentation: vtkMRMLSegmentationNode
    outputFiducials: vtkMRMLMarkupsFiducialNode
    pathFiducials: vtkMRMLMarkupsFiducialNode
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# PlanAdjustmentWidget
#


class PlanAdjustmentWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PlanAdjustment.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PlanAdjustmentLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.showROICheckBox.connect("toggled(bool)", self.onShowROIToggled)
        self.ui.updateNeedleEntryButton.connect("clicked(bool)", self.onUpdateNeedleEntryButton)

        # Connect input controls for automatic processing
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)
        self.ui.markupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMarkupChanged)
        self.ui.outputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)
        self.ui.outputFiducialSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputFiducialChanged)
        self.ui.pathFiducialSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onPathFiducialChanged)
        # Use sliderReleased signal to only crop when user stops sliding
        self.ui.roiRLSlider.connect("sliderReleased()", self.onInputChanged)
        self.ui.roiAPSlider.connect("sliderReleased()", self.onInputChanged)
        
        # Track markup node observer
        self._markupObserverTag = None
        self._currentMarkupNode = None

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        # Clean up markup observer
        self._removeMarkupObserver()
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.onInputChanged)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[PlanAdjustmentParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.onInputChanged)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.onInputChanged)
            # Update markup observer when parameter node changes
            self._updateMarkupObserver()


    def resampleLabelmapVolume(self, segmentationNode, voxelSpacing=1.0):
        """
        Resample a segmentation's labelmap to specified voxel spacing using nearest neighbor interpolation.
        
        Args:
            segmentationNode: The segmentation node to resample
            voxelSpacing: Voxel spacing in mm . Default is 1.0.
            
        Returns:
            tuple: (gridLabels, resampledNode) where gridLabels is the numpy array and 
                   resampledNode is the temporary node (caller should clean up)
                   Returns (None, None) if resampling fails
        """
        if not segmentationNode:
            logging.warning("No segmentation node provided")
            return None, None
            
        # Export segmentation to labelmap
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        success = slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            segmentationNode, labelmapVolumeNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
        )
        
        if not success:
            logging.warning("Failed to export segmentation to labelmap")
            slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
            return None, None
        
        # Convert labelmap to scalar volume for CropInterpolated (which requires scalar volumes)
        scalarVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        scalarVolumeNode.SetAndObserveImageData(labelmapVolumeNode.GetImageData())
        scalarVolumeNode.CopyOrientation(labelmapVolumeNode)
        
        # Create resampled labelmap with specified voxel spacing
        resampledScalarNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", 
                                                                 f"ResampledScalar_{voxelSpacing}mm")

        roiNodeName = f"{self._parameterNode.inputVolume.GetName()}_ROI"
        roiNode = slicer.mrmlScene.GetFirstNodeByName(roiNodeName)

        min_spacing = min(labelmapVolumeNode.GetSpacing())
        scaling = voxelSpacing / min_spacing


        cropVolumeLogic = slicer.modules.cropvolume.logic()
        success = cropVolumeLogic.CropInterpolated(roiNode, scalarVolumeNode, resampledScalarNode, True, scaling, slicer.vtkMRMLCropVolumeParametersNode.InterpolationNearestNeighbor, 0.0)

        # Clean up original labelmap and temporary scalar volume
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
        slicer.mrmlScene.RemoveNode(scalarVolumeNode)

        if success == 0: # success
            # Convert the resampled scalar volume back to labelmap volume for proper handling
            resampledLabelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", 
                                                                       f"ResampledLabelmap_{voxelSpacing}mm")
            resampledLabelmapNode.SetAndObserveImageData(resampledScalarNode.GetImageData())
            resampledLabelmapNode.CopyOrientation(resampledScalarNode)

            # Clean up temporary scalar node
            slicer.mrmlScene.RemoveNode(resampledScalarNode)
            
            # Get the resampled volume data as numpy array
            volumeArray = slicer.util.arrayFromVolume(resampledScalarNode)

            # Get volume dimensions and spacing for logging
            dimensions = resampledLabelmapNode.GetImageData().GetDimensions()
            spacing = resampledLabelmapNode.GetSpacing()

            logging.info(f"Resampled labelmap dimensions: {dimensions}")
            logging.info(f"Resampled labelmap spacing: {spacing}")
            logging.info(f"Resampled labelmap array shape: {volumeArray.shape}")
            logging.info(f"Label range: {volumeArray.min()} to {volumeArray.max()}")
            logging.info(f"Unique labels: {np.unique(volumeArray)}")

            return volumeArray, resampledLabelmapNode
        else:
            logging.error("Failed to resample labelmap volume")
            slicer.mrmlScene.RemoveNode(resampledScalarNode)
            return None, None


    def onInputChanged(self, caller=None, event=None) -> None:
        """Called when any input parameter changes - update ROI and optionally crop."""
        if (self._parameterNode and
            self._parameterNode.inputVolume and
            self._parameterNode.markupFiducials and
            self._parameterNode.markupFiducials.GetNumberOfControlPoints() > 0):

            try:
                # Always update ROI visualization (regardless of Auto Update setting)
                self.updateROI()
                # Only crop if auto update is enabled and output volume is selected
                if (self._parameterNode.autoUpdate and self._parameterNode.outputVolume):
                    self.logic.process(
                        self._parameterNode.inputVolume,
                        self._parameterNode.markupFiducials,
                        self._parameterNode.outputVolume,
                        self._parameterNode.roiRLDimension,
                        self._parameterNode.roiAPDimension,
                        False
                    )
            except Exception as e:
                logging.warning(f"Automatic processing failed: {str(e)}")

    def onShowROIToggled(self, checked) -> None:
        """Called when Show ROI checkbox is toggled."""
        if not self._parameterNode or not self._parameterNode.inputVolume:
            return

        roiNodeName = f"{self._parameterNode.inputVolume.GetName()}_ROI"
        roiNode = slicer.mrmlScene.GetFirstNodeByName(roiNodeName)

        if roiNode:
            displayNode = roiNode.GetDisplayNode()
            if not displayNode:
                roiNode.CreateDefaultDisplayNodes()
                displayNode = roiNode.GetDisplayNode()

            if displayNode:
                displayNode.SetVisibility(checked)

    def onOutputFiducialChanged(self) -> None:
        """Called when output fiducial selector changes - set color to blue."""
        outputFiducialNode = self.ui.outputFiducialSelector.currentNode()
        if outputFiducialNode:
            displayNode = outputFiducialNode.GetDisplayNode()
            if not displayNode:
                outputFiducialNode.CreateDefaultDisplayNodes()
                displayNode = outputFiducialNode.GetDisplayNode()

            if displayNode:
                # Set color to blue (RGB: 0, 0, 1)
                displayNode.SetSelectedColor(0, 0, 1)
                displayNode.SetColor(0, 0, 1)

    def onPathFiducialChanged(self) -> None:
        """Called when output fiducial selector changes - set color to blue."""
        pathFiducialNode = self.ui.pathFiducialSelector.currentNode()
        if pathFiducialNode:
            displayNode = pathFiducialNode.GetDisplayNode()
            if not displayNode:
                pathFiducialNode.CreateDefaultDisplayNodes()
                displayNode = pathFiducialNode.GetDisplayNode()

            if displayNode:
                # Set color to blue (RGB: 0, 0, 1)
                displayNode.SetSelectedColor(0, 0, 1)
                displayNode.SetColor(0, 0, 1)

    def onMarkupChanged(self) -> None:
        """Called when markup selector changes - update observers and trigger processing."""
        # Update markup observer
        self._updateMarkupObserver()
        # Call regular input change handler
        self.onInputChanged()

    def _updateMarkupObserver(self) -> None:
        """Update observer for current markup node."""
        # Remove old observer
        self._removeMarkupObserver()

        # Add observer to new markup node
        if self._parameterNode and self._parameterNode.markupFiducials:
            markupNode = self._parameterNode.markupFiducials
            self._currentMarkupNode = markupNode
            # Observe point modifications, additions, and removals
            self._markupObserverTag = self.addObserver(markupNode, vtk.vtkCommand.ModifiedEvent, self.onMarkupModified)

    def _removeMarkupObserver(self) -> None:
        """Remove observer from current markup node."""
        if self._currentMarkupNode and self._markupObserverTag:
            self.removeObserver(self._currentMarkupNode, vtk.vtkCommand.ModifiedEvent, self._markupObserverTag)
        self._currentMarkupNode = None
        self._markupObserverTag = None

    def onMarkupModified(self, caller=None, event=None) -> None:
        """Called when markup node is modified - update ROI and optionally crop."""
        # Always update ROI when markups change
        self.updateROI()

        # Only crop if auto update is enabled and output volume is selected
        if (self._parameterNode and
            self._parameterNode.autoUpdate and
            self._parameterNode.outputVolume and
            self._parameterNode.markupFiducials and
            self._parameterNode.markupFiducials.GetNumberOfControlPoints() > 0):

            try:
                self.logic.process(
                    self._parameterNode.inputVolume,
                    self._parameterNode.markupFiducials,
                    self._parameterNode.outputVolume,
                    self._parameterNode.roiRLDimension,
                    self._parameterNode.roiAPDimension,
                    False
                )
            except Exception as e:
                logging.warning(f"Automatic processing failed after markup modification: {str(e)}")

    def onUpdateNeedleEntryButton(self) -> None:
        """Update needle entry point based on current parameters."""
        if not self._parameterNode or not self._parameterNode.outputFiducials:
            slicer.util.warningDisplay("Please select an output fiducial list first.")
            return

        if not self._parameterNode or not self._parameterNode.pathFiducials:
            slicer.util.warningDisplay("Please select a path fiducial list first.")
            return
        
        if not self._parameterNode.markupFiducials or self._parameterNode.markupFiducials.GetNumberOfControlPoints() == 0:
            slicer.util.warningDisplay("Please add at least one input fiducial point.")
            return

        def xyz_to_zyx(arr_xyz) -> list:
            """
            Convert XYZ → ZYX convention.
            Mapping: [x, y, z] -> [-z, -y, x]
            Returns list(s), not NumPy arrays.
            """
            arr_xyz = np.atleast_2d(np.asarray(arr_xyz, dtype=float))
            zyx = np.stack([
                arr_xyz[:, 2],   # -z
                arr_xyz[:, 1],   # -y
                arr_xyz[:, 0],    #  x
            ], axis=1)
            return zyx[0].tolist() if zyx.shape[0] == 1 else zyx.tolist()

        def zyx_to_xyz(arr_zyx) -> list:
            """
            Convert ZYX → XYZ convention.
            Mapping: [z, y, x] -> [x, -y, -z]
            Returns list(s), not NumPy arrays.
            """
            arr_zyx = np.atleast_2d(np.asarray(arr_zyx, dtype=float))
            x = arr_zyx[:, 2]
            y = arr_zyx[:, 1]
            z = arr_zyx[:, 0]
            result = np.stack([x, y, z], axis=1)
            return result[0].tolist() if result.shape[0] == 1 else result.tolist()

        def vtk4x4_to_numpy(m: vtk.vtkMatrix4x4) -> np.ndarray:
            """Copy vtkMatrix4x4 to a (4,4) NumPy array."""
            M = np.empty((4,4), dtype=float)
            for r in range(4):
                for c in range(4):
                    M[r, c] = m.GetElement(r, c)
            return M

        def ijk_to_ras_point(point_ijk, ijkToRas_vtk: vtk.vtkMatrix4x4) -> list:
            """
            Convert a single IJK point → RAS.
            Returns a Python list [x, y, z].
            """
            point4 = [point_ijk[0], point_ijk[1], point_ijk[2], 1.0]
            ras4 = [0, 0, 0, 0]
            ijkToRas_vtk.MultiplyPoint(point4, ras4)
            return ras4[:3]  # plain list

        def ras_to_ijk_point(point_ras, rasToIjk_vtk: vtk.vtkMatrix4x4) -> list:
            """
            Convert a single RAS point → IJK.
            Returns a Python list [i, j, k].
            """
            point4 = [point_ras[0], point_ras[1], point_ras[2], 1.0]
            ijk4 = [0, 0, 0, 0]
            rasToIjk_vtk.MultiplyPoint(point4, ijk4)
            return ijk4[:3]

        def ijk_to_ras_batch(points_ijk, ijkToRas_vtk: vtk.vtkMatrix4x4) -> list:
            """
            Convert multiple IJK points → RAS.
            Input: (N,3) array-like
            Returns list of lists [[x,y,z], ...].
            """
            pts = np.atleast_2d(np.asarray(points_ijk, dtype=float))
            M = np.array([[ijkToRas_vtk.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)
            pts4 = np.hstack([pts, np.ones((pts.shape[0], 1))])
            ras = (pts4 @ M.T)[:, :3]
            return ras.tolist()

        def ras_to_ijk_batch(points_ras, rasToIjk_vtk: vtk.vtkMatrix4x4) -> list:
            """
            Convert multiple RAS points → IJK.
            Input: (N,3) array-like
            Returns list of lists [[i,j,k], ...].
            """
            pts = np.atleast_2d(np.asarray(points_ras, dtype=float))
            M = np.array([[rasToIjk_vtk.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)
            pts4 = np.hstack([pts, np.ones((pts.shape[0], 1))])
            ijk = (pts4 @ M.T)[:, :3]
            return ijk.tolist()
        
        def vtkmatrix4x4_to_numpy(matrix: vtk.vtkMatrix4x4) -> np.ndarray:
            """Convert a vtkMatrix4x4 to a (4,4) NumPy array."""
            return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)

        def np_array_to_vtk_table(np_array: np.ndarray) -> vtk.vtkTable:
            n_rows, n_cols = np_array.shape
            table = vtk.vtkTable()
            for j in range(n_cols):
                a = vtk.vtkDoubleArray()
                a.SetName(f"c{j+1}")
                a.SetNumberOfValues(n_rows)
                for i in range(n_rows):
                    a.SetValue(i, float(np_array[i, j]))
                table.AddColumn(a)

        # Get target and entry positions
        targetPos = [0, 0, 0]
        self._parameterNode.markupFiducials.GetNthControlPointPosition(0, targetPos)

        entryPos = [0, 0, 0]
        self._parameterNode.markupFiducials.GetNthControlPointPosition(1, entryPos)

        # Resample labelmap volume with 1mm voxel spacing using nearest neighbor interpolation
        gridLabels_ijk = None
        entry_ijk = None
        target_ijk = None
        resampledLabelmapNode = None

        if self._parameterNode.segmentation:
            # Use the new resampling function
            gridLabels_ijk, resampledLabelmapNode = self.resampleLabelmapVolume(
                self._parameterNode.segmentation, voxelSpacing=1.0
            )
            if gridLabels_ijk is not None and resampledLabelmapNode is not None:
                # Convert target and entry positions to IJK coordinates of resampled volume
                rasToIjk = vtk.vtkMatrix4x4()
                resampledLabelmapNode.GetRASToIJKMatrix(rasToIjk)
                # ras to ijk conversions (target / entry in RAS → IJK)
                target_ijk = ras_to_ijk_point(targetPos, rasToIjk)
                entry_ijk  = ras_to_ijk_point(entryPos,  rasToIjk)    
           
                logging.info(f"Target position (RAS): {targetPos}")
                logging.info(f"Target position (IJK): {target_ijk}")
                logging.info(f"Entry position (RAS): {entryPos}")
                logging.info(f"Entry position (IJK): {entry_ijk}")
        else:
            logging.warning("No segmentation node specified")

        # Get info for Lydia:
        # i = needle/robot horiontal direction  (patient L-R)
        # j = needle/robot vertical direction   (patient A-P)
        # k = needle/robot insertion direction  (patient I-S)
        print('*************** For Lydia ******************')
        sag_slice = gridLabels_ijk[:, :, int(entry_ijk[0])]
        plt.imshow(np.fliplr(sag_slice), origin="lower")
        plt.xlabel("A -> P")
        plt.ylabel("I -> S")
        plt.savefig(os.path.join(path_planning_dir, 'debug', 'gridlabels_ijk.png'))
        ijkToRas = vtk.vtkMatrix4x4()
        resampledLabelmapNode.GetIJKToRASMatrix(ijkToRas)
        print(ijkToRas)
        print(f"Target position (RAS): {targetPos}")
        print(f"Target position (IJK): {target_ijk}")
        print(f"Entry position (RAS): {entryPos}")
        print(f"Entry position (IJK): {entry_ijk}")
        # Export to csv
        np_ijkToRas = vtkmatrix4x4_to_numpy(ijkToRas)
        slice_array = np.array(sag_slice)
        np.savetxt(os.path.join(path_planning_dir, 'debug', 'ijkToRas.csv'), np_ijkToRas, delimiter=",")
        np.savetxt(os.path.join(path_planning_dir, 'debug', 'gridlabels_ijk.csv'), slice_array, delimiter=",")
        np.savetxt(os.path.join(path_planning_dir, 'debug', 'entry_ijk.csv'), np.array(entry_ijk), delimiter=",")
        np.savetxt(os.path.join(path_planning_dir, 'debug', 'target_ijk.csv'), np.array(target_ijk), delimiter=",")
        np.savetxt(os.path.join(path_planning_dir, 'debug', 'entry_ras.csv'), np.array(entryPos), delimiter=",")
        np.savetxt(os.path.join(path_planning_dir, 'debug', 'target_ras.csv'), np.array(targetPos), delimiter=",")
        
        # DEBUG in Slicer:
        #table_grid = np_array_to_vtk_table(slice_array)
        #tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        #tableNode.SetName("Gridlabels_ijk")
        #tableNode.SetAndObserveTable(table_grid)
        #slicer.util.saveNode(tableNode, os.path.join(path_planning_dir, 'debug', 'gridlabels_ijk.csv'))
        print('*****************************************')
        

        # Rotate coordinate system from XYZ to YZX for external function compatibility
        if gridLabels_ijk is not None and entry_ijk is not None and target_ijk is not None:
            # gridLabels: numpy array from slicer (ZYX order) -> transpose to YZX
            gridLabels_rotated = np.transpose(gridLabels_ijk, axes=(0, 1, 2))
            # entry: [x, y, z] -> [z, -y, -x]
            entry_rotated = np.array(xyz_to_zyx(entry_ijk))
            # target: [x, y, z] -> [z, y, x]
            target_rotated = np.array(xyz_to_zyx(target_ijk))
            logging.info(f"Original entry indexes (XYZ): {entry_ijk}")
            logging.info(f"Rotated entry indexes (YZX): {entry_rotated}")
            logging.info(f"Original target indexes (XYZ): {target_ijk}")
            logging.info(f"Rotated target indexes (YZX): {target_rotated}")

            # Update variables for use with external function
            gridLabels = gridLabels_rotated
            entry = entry_rotated
            target = target_rotated

        # Call path planning function if available
        if PATH_PLANNING_AVAILABLE and gridLabels is not None and entry is not None and target is not None:
            try:
                # Install required packages on first run
                install_required_packages()

                # Load the regressor model
                reg = load_reg()

                # Set default search_samples if not defined
                search_samples = 20  # You can adjust this value as needed
                # entry = [5,5,5]
                # target = [80,5,5]
                print('Entry (grid) = %s' %entry)
                print('Target (grid) = %s' %target)
                print(f"GridLabels shape: {gridLabels.shape}")
                print(f"GridLabels min: {gridLabels.min()}, max: {gridLabels.max()}")

                # Find the optimal entry point
                plt.clf() 
                plt.imshow(gridLabels[:, :, int(entry[2])])
                plt.savefig(os.path.join(path_planning_dir, 'debug', 'geo_img.png'))
                best_start_point, tolerance_count, best_index = find_entry(gridLabels, reg, entry, target, search_dist=8, search_samples=search_samples)
                print('Adjusted Entry (grid) = %s' %best_start_point)


                # Convert back from YZX to XYZ coordinates for Slicer
                if best_start_point is not None:
                    # best_start_point is in YZX format, convert back to XYZ entry: [z, -y, -x] -> [x, y, z]
                    best_start_xyz = zyx_to_xyz(best_start_point)
                    # Convert from IJK coordinates back to RAS coordinates using the resampled volume transform
                    if resampledLabelmapNode is not None:
                        # Get IJK to RAS transform from the resampled volume we already have
                        ijkToRas = vtk.vtkMatrix4x4()
                        resampledLabelmapNode.GetIJKToRASMatrix(ijkToRas)
                        # Transform optimized entry point back to RAS
                        entryPos = ijk_to_ras_point(best_start_xyz, ijkToRas)
                        logging.info(f"Updated entry position to optimized location: {entryPos}")
                    # Get path
                    planned_path = plan_path(gridLabels, reg, best_start_point, target, tolerance_range=1)
                    logging.info(f"Path planning completed: best_start_point={best_start_point}, tolerance_count={tolerance_count}, best_index={best_index}")
                    path_ijk = zyx_to_xyz(planned_path)
                    path_ras = ijk_to_ras_batch(path_ijk, ijkToRas)



            except Exception as e:
                logging.error(f"Path planning failed: {e}")
                logging.info("Using original entry position")
        else:
            if not PATH_PLANNING_AVAILABLE:
                logging.warning("Path planning functions not available - using original entry position")
            else:
                logging.warning("Grid data not available - using original entry position")

        # Clean up temporary resampled volume
        if resampledLabelmapNode is not None:
            slicer.mrmlScene.RemoveNode(resampledLabelmapNode)

        # Add or update the needle entry point in output fiducials
        # Ensure outputFiducials has exactly 2 points:
        # index 0 -> "TARGET" (targetPos), index 1 -> "ENTRY" (entryPos)
        outputFiducials = self._parameterNode.outputFiducials
        if not outputFiducials:
            raise RuntimeError("outputFiducials node is not set")
        # Make sure we only have the two we want, in the correct order
        outputFiducials.RemoveAllControlPoints()
        # Add TARGET at index 0
        outputFiducials.AddControlPoint(targetPos)
        outputFiducials.SetNthControlPointLabel(0, "TARGET")
        # Add ENTRY at index 1
        outputFiducials.AddControlPoint(entryPos)
        outputFiducials.SetNthControlPointLabel(1, "ENTRY")

        # Add path output from path
        pathFiducials = self._parameterNode.pathFiducials
        pathFiducials.RemoveAllControlPoints()

        displayNode = pathFiducials.GetDisplayNode()
        displayNode.SetTextScale(0.0)          
        label = ''
        for i, (x, y, z) in enumerate(path_ras):
            pathFiducials.AddControlPoint(vtk.vtkVector3d(float(x), float(y), float(z)), label)



    def updateROI(self) -> None:
        """Update ROI node based on current parameters."""
        if not self._parameterNode or not self._parameterNode.inputVolume or not self._parameterNode.markupFiducials:
            return

        markupNode = self._parameterNode.markupFiducials
        numPoints = markupNode.GetNumberOfControlPoints()

        if numPoints < 1:
            return

        # Get the first fiducial point in RAS coordinates (center of ROI)
        fiducial1_RAS = [0, 0, 0]
        markupNode.GetNthControlPointPosition(0, fiducial1_RAS)

        # Get input volume bounds for SI dimension
        bounds_RAS = [0, 0, 0, 0, 0, 0]
        self._parameterNode.inputVolume.GetRASBounds(bounds_RAS)
        si_dimension = bounds_RAS[5] - bounds_RAS[4]

        # Create or update ROI node
        roiNodeName = f"{self._parameterNode.inputVolume.GetName()}_ROI"
        roiNode = slicer.mrmlScene.GetFirstNodeByName(roiNodeName)
        if not roiNode:
            roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", roiNodeName)

        # Update ROI position and size
        roiNode.SetCenterWorld(fiducial1_RAS)
        roiNode.SetSizeWorld([self._parameterNode.roiRLDimension, self._parameterNode.roiAPDimension, si_dimension])

        # Create or get transform node for ROI orientation
        transformNodeName = f"{self._parameterNode.inputVolume.GetName()}_ROI_Transform"
        transformNode = slicer.mrmlScene.GetFirstNodeByName(transformNodeName)
        if not transformNode:
            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", transformNodeName)

        # Apply transform to ROI
        roiNode.SetAndObserveTransformNodeID(transformNode.GetID())

        # If we have a second fiducial, align ROI axes to needle path using transform
        if numPoints >= 2:

            # Get second fiducial point
            fiducial2_RAS = [0, 0, 0]
            markupNode.GetNthControlPointPosition(1, fiducial2_RAS)

            # Calculate needle path vector (from first to second fiducial)
            needlePath = np.array(fiducial2_RAS) - np.array(fiducial1_RAS)
            needlePath = needlePath / np.linalg.norm(needlePath)  # normalize

            # Define anatomical AP direction (anterior-posterior)
            ap_direction = np.array([0, 1, 0])  # Y+ is anterior in RAS

            # Compute first axis (closest to RL) as cross product of needle path and AP
            first_axis = np.cross(needlePath, ap_direction)
            first_axis = first_axis / np.linalg.norm(first_axis)  # normalize

            # Compute second axis as cross product of first axis and needle path
            second_axis = np.cross(first_axis, needlePath)
            second_axis = second_axis / np.linalg.norm(second_axis)  # normalize

            # Third axis is the needle path (closest to SI)
            third_axis = needlePath

            # Create orientation matrix (columns are the axes)
            orientation_matrix = np.column_stack([first_axis, second_axis, third_axis])

            # Create VTK transform matrix
            vtkMatrix = vtk.vtkMatrix4x4()
            vtkMatrix.Identity()
            for i in range(3):
                for j in range(3):
                    vtkMatrix.SetElement(i, j, orientation_matrix[i, j])

            # Set transform matrix
            transformNode.SetMatrixTransformToParent(vtkMatrix)

            logging.info(f"ROI aligned to needle path from {fiducial1_RAS} to {fiducial2_RAS}")
        else:
            # Reset to identity transform if only one fiducial
            identityMatrix = vtk.vtkMatrix4x4()
            identityMatrix.Identity()
            transformNode.SetMatrixTransformToParent(identityMatrix)

#
# PlanAdjustmentLogic
#


class PlanAdjustmentLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return PlanAdjustmentParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                markupFiducials: vtkMRMLMarkupsFiducialNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                roiRLDimension: float,
                roiAPDimension: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be processed
        :param markupFiducials: fiducial markups for needle planning
        :param outputVolume: processing result
        :param roiRLDimension: Right-Left dimension for ROI (0-200mm)
        :param roiAPDimension: Anterior-Posterior dimension for ROI (0-200mm)
        :param invert: if True then processing is inverted
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        if not markupFiducials or markupFiducials.GetNumberOfControlPoints() == 0:
            raise ValueError("At least one fiducial point is required")

        import time

        startTime = time.time()
        logging.info("Processing started")
        logging.info(f"ROI RL dimension: {roiRLDimension}mm")
        logging.info(f"ROI AP dimension: {roiAPDimension}mm")
        logging.info(f"Using markup fiducials: {markupFiducials.GetName()}")

        # Get the first fiducial point in RAS coordinates
        fiducialPos_RAS = [0, 0, 0]
        markupFiducials.GetNthControlPointPosition(0, fiducialPos_RAS)
        logging.info(f"Target position (RAS): {fiducialPos_RAS}")

        # Get input volume bounds in RAS coordinates
        bounds_RAS = [0, 0, 0, 0, 0, 0]
        inputVolume.GetRASBounds(bounds_RAS)

        # Create or get existing ROI node
        roiNodeName = f"{inputVolume.GetName()}_ROI"
        roiNode = slicer.mrmlScene.GetFirstNodeByName(roiNodeName)
        if not roiNode:
            roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", roiNodeName)

        # Set ROI center to fiducial position
        center = fiducialPos_RAS
        center[2] = 0.5*(bounds_RAS[4] + bounds_RAS[5])
        roiNode.SetCenterWorld(fiducialPos_RAS)

        # Calculate ROI size: RL and AP dimensions from sliders, SI dimension from input volume
        si_dimension = bounds_RAS[5] - bounds_RAS[4]  # Full SI extent
        roiNode.SetSizeWorld([roiRLDimension, roiAPDimension, si_dimension])

        logging.info(f"ROI center: {fiducialPos_RAS}")
        logging.info(f"ROI size: [{roiRLDimension}, {roiAPDimension}, {si_dimension}]")

        # Use Slicer's crop volume functionality

        cropVolumeLogic = slicer.modules.cropvolume.logic()

        # We are using CropInterpolated instead of ResampleScalarVolume because the latter
        # does not working..

        success = cropVolumeLogic.CropVoxelBased(roiNode, inputVolume, outputVolume)

        if not success:
        # Fallback: use the interpolated approach
            cropVolumeLogic.CropInterpolated(roiNode, inputVolume, outputVolume, True, 1.0, slicer.vtkMRMLCropVolumeParametersNode.InterpolationLinear, 0.0)

        if showResult:
            # Show the result in slice viewers
            slicer.util.setSliceViewerLayers(background=outputVolume)   
        
        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")



#
# PlanAdjustmentTest
#


class PlanAdjustmentTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_PlanAdjustment1()

    def test_PlanAdjustment1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("PlanAdjustment1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = PlanAdjustmentLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")