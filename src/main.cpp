/*
 * main.cpp
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#include "DHMProcessor.cuh"

// TODO: move try-catch entirely inside DHMProcessor
// exceptions not working at all, error message is gibberish

int main(int argc, char* argv[])
{
    std::string input_dir = argc == 1 ? "../test/input" : std::string(argv[1]);

    try
    {
        DHMProcessor dhm("../test/output");

        // dhm.set_callback()

        CUDA_TIMER( dhm.process_folder(input_dir) );
    }
    catch (DHMException &e)
    {
        std::cerr << e.what() << std::endl; // this yields gibberish
    }
}




