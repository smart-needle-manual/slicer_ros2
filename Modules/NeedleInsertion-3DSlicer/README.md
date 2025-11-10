# Dependencies install (in 3D Slicer):
1. Install dependencies from within 3D Slicer python console:
```
slicer.util.pip_install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
slicer.util.pip_install('scikit-image')
slicer.util.pip_install('scikit-learn')
slicer.util.pip_install('tqdm')
slicer.util.pip_install('pynrrd')
```
