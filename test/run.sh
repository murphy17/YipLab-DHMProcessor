# Example command line invocation of DHMProcessor
# usage: ./DHMProcessor /path/to/input/dir /path/to/output/dir z_min delta_z num_steps num_frames save_volume
# z_min, delta_z in mm
# num_frames = 0 to only process files currently in folder, -1 to monitor realtime
# save_volume = 0 or 1

../Debug/DHMProcessor "./in" "./out" 30 1 20 0 1