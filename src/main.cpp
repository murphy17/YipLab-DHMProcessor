/*
 * main.cpp
 *
 *  Created on: Jun 15, 2017
 *      Author: michaelmurphy
 */

#include "DHMProcessor.cuh"

// TODO: move try-catch entirely inside DHMProcessor
// exceptions not working at all, error message is gibberish

int main()
{
    try {
        DHMProcessor dhm("../test/output");

        // dhm.set_callback()

        dhm.process_folder("../test/input");

    } catch (DHMException &e) {
        std::cerr << e.what() << std::endl;
    }
}




