# YipLab-DHMProcessor

Features:
- Asynchronous image load + store in separate CPU threads
- Double-buffering for overlapped filter stack transfer + processing
- Optional reconstructed image-stack save
- Single-quadrant storage + transfer of filter stack (implicitly mirrored during multiply)
- ...

TODO:
- Extra frequency-domain processing
- Autofocus implementation
- MicroManager integration (direct image transfer)
- MicroManager integration (plugin)

Files:
- include/common.h, include/helper_string.h
	Not my code, used for CUDA error handling
- include/DHMCommon.h