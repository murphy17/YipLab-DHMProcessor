# YipLab-DHMProcessor


This repository contains a CUDA implementation of digital holographic microscopy numerical 
reconstruction. The workflow is wrapped via a C++ object, ```DHMProcessor```.


Thus far, the workflow continuously monitors a folder (possibly residing on a network server) and
loads images into memory as they appear, generates a stack of reconstructions at specified
Z-distances, and optionally saves the volume to disk. Space is provided in the code for additional
processing -- i.e. to generate a depth-map. Depth-map saving is implemented. 


Input images must be 1024x1024 8-bit TIFF. The output depth map (presently a placeholder) is
1024x1024 floating-point TIFF. The output volume is a TIFF stack of 1024x10248-bit slices.  


Average runtimes, generating 100 1024x1024 slices per frame without saving, are
250ms/frame on Jetson TX2 and 36ms/frame on Titan X. The slowest operation on the Jetson TX2
is (NVIDIA's) inverse FFT; the Titan X spends ~6ms/frame idling while the wavefront stack transfers.
Using separate CPU threads to perform network file transfers concurrently with GPU processing
effectively eliminates any wait time to load and store TIFF images. (This does not apply when
choosing to save the entire image stack -- the workflow runtime is negligible in comparison.) 
Half-precision arithmetic was explored on the Jetson TX2; while the multiplication and modulus
operations sped up by ~30%, the 16-bit FFT was slower by the same amount (16-bit arithmetic
additionally requires the data be re-scaled in a few places to prevent saturation.) CPU and GPU
operations are not interleaved frequently enough in this workflow for the Jetson TX2's unified
memory to provide any meaningful speedup.


### Features
- Double-buffering for overlapped filter stack transfer + processing
- Single-quadrant storage + transfer of filter stack (implicitly mirrored during multiply)
- Asynchronous image load + store in separate CPU threads
- Slice masking (speedup -- can skip processing of certain slices in next frame based on current frame)
- Optional reconstructed image-stack save
- C++ exceptions


### Todo
- Wavefront stack transfer speedup (transfer upper triangle of quadrant)
- Extra frequency-domain processing (denoising)
- Extra spatial-domain processing (defringing)
- Autofocus implementation (object segmentation <-> arg-max object sharpness)
- MicroManager integration (direct image transfer, plugin GUI)


### Files
- README.md
	* This file
- .project, .cproject
	* Eclipse / NSight project files (contain necessary compiler flags)
- include/common.h, include/helper_string.h
	* Not my code, used for CUDA error handling
- include/util.h
	* Timers, printing, visualization functions to aid development
- test/run.sh
	* Shell script showing how to use command-line arguments
- thirdparty/strnatcmp/
	* Natural string sort, from https://github.com/sourcefrog/natsort
- thirdparty/TinyTIFF/
	* Fast TIFF I/O, from https://github.com/jkriege2/TinyTIFF
- src/DHMProcessor.cu, src/DHMProcessor.cuh
	* DHMProcessor object and header -- all functionality encapsulated in here
- src/ImageQueue.cpp, src/ImageQueue.hpp
	* A simple blocking queue type, used here for performing concurrent file I/O in separate threads
- src/main.cu
	* Main executable file -- parses command line arguments, shows how to use the object
	
	
### Dependencies
- CUDA
- CUFFT
- OpenCV (with CUDA support)
- Boost
- C++11