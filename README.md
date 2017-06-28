### YipLab-DHMProcessor

This repository contains a ...

# Features
- Asynchronous image load + store in separate CPU threads
- Double-buffering for overlapped filter stack transfer + processing
- Optional reconstructed image-stack save
- Single-quadrant storage + transfer of filter stack (implicitly mirrored during multiply)
- ...

# TODO
- Extra frequency-domain processing (denoising)
- Extra spatial-domain processing (defringing)
- Autofocus implementation
- MicroManager integration (direct image transfer)
- MicroManager integration (plugin)

# Files:
- README.md
	This file
- .project, .cproject
	Eclipse / NSight project files
- include/common.h, include/helper_string.h
	Not my code, used for CUDA error handling
- include/util.h
	Timers, printing, visualization functions to aid development
- test/run.sh
	Shell script showing how to use command-line arguments
- thirdparty/strnatcmp/
	Natural string sort, from https://github.com/sourcefrog/natsort
- thirdparty/TinyTIFF/
	Fast TIFF I/O, from https://github.com/jkriege2/TinyTIFF
- src/DHMProcessor.cu, src/DHMProcessor.cuh
	DHMProcessor object and header -- all functionality encapsulated in here
- src/ImageQueue.cpp, src/ImageQueue.hpp
	A simple blocking queue type, used here for performing concurrent file I/O in separate threads
- src/main.cu
	Main executable file -- parses command line arguments, shows how to use the object