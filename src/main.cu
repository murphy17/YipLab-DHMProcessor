/*
 * main.cpp
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#include "DHMProcessor.cuh"

using namespace YipLab;

////////////////////////////////////////////////////////////////////////////////
// Example usage
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    using namespace std;

    string input_dir, output_dir;
    bool save_volume;
    int num_slices, max_frames;
    float delta_z, z_init;

    if (argc < 7)
    {
        input_dir = "/mnt/image_store/Murphy_Michael/dhm_in/spheres";
        output_dir = "/mnt/image_store/Murphy_Michael/dhm_out/spheres";
        z_init = 30.0f;
        delta_z = 1.0f;
        num_slices = 50;
        max_frames = 100;
        save_volume = false;
    }
    else
    {
        input_dir = string(argv[1]);
        output_dir = string(argv[2]);
        z_init = stof(string(argv[3]));
        delta_z = stof(string(argv[4]));
        num_slices = stoi(string(argv[5]));
        max_frames = stoi(string(argv[6]));
        save_volume = stoi(string(argv[7]));
    }

    float delta_x = 5.32f / 1024.f;
    float delta_y = 6.66f / 1280.f;
    float lambda0 = 0.000488f;

    DHMProcessor dhm(num_slices, delta_z, z_init, delta_x, delta_y, lambda0);

    // inputs must be 8-bit single TIFF images, of size 1024x1024
    dhm.process_folder(input_dir, output_dir, save_volume, max_frames);
}


