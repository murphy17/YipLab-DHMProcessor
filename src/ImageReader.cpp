#include "ImageReader.hpp"

// TODO: complementary ImageWriter for storage

namespace YipLab {

ImageReader::ImageReader(const fs::path input_path, const int buffer_size)
{
    this->input_path = input_path;

    _queue = new BoundedBuffer<Image>(buffer_size);
    std::atomic_store(&_stop, 0);
    this->latest_frame = -1;
}

void ImageReader::run() // max_frames
{
    // watch the folder
    _thread = std::thread([&](){
        try {
            while (std::atomic_load(&_stop) == 0)
            {
                //////////////////////////////////////////////////////////////////////////////////////////////////////
                // MicroManager acquisition code would live here.
                // (MM API docs at https://valelab4.ucsf.edu/~MM/doc/MMCore/html/class_c_m_m_core.html)
                // Basically, just comment everything out inside this while loop, add a call that asks MM for a new
                // image, wrap the returned data pointer in an OpenCV Mat (then wrap that in my Image structure, with
                // some descriptor string in the "path" field). Then just do _queue->push_front.
                // (Alternatively, MM appears to use an event system -- that would be a bit more involved to use)
                //////////////////////////////////////////////////////////////////////////////////////////////////////

                std::vector<fs::path> new_files;

                // I think the iterator has problems with folder being modified
                for (auto &i : boost::make_iterator_range(fs::directory_iterator(input_path), {})) {
                    fs::path this_path = i.path();
                    if (std::atoi(this_path.stem().c_str()) > latest_frame)
                    {
                        new_files.push_back(this_path);
                    }
                }

                if (new_files.size() > 0)
                {
                    std::sort(new_files.begin(), new_files.end(),
                              [](const fs::path &a, const fs::path &b) -> bool
                              {
                                  return strnatcmp(a.stem().c_str(), b.stem().c_str()) < 0;
                              });

                    for (fs::path p : new_files)
                    {
                        cv::Mat mat;
                        int num_tries = 0;
                        bool f_exists = true;

                        do {
                            f_exists &= fs::exists(p); // make sure file hasn't subsequently vanished
                            if (!f_exists) break;
                            mat = cv::imread(p.string(), CV_LOAD_IMAGE_GRAYSCALE);
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        } while ( mat.rows == 0 && ++num_tries < MAX_N_TRIES );

                        if ( !f_exists || num_tries == MAX_N_TRIES )
                        {
                            std::cout << p.string() << " read timed out" << std::endl;
                        }
                        else
                        {
                            Image img = { mat, p.string() };
                            // push to image queue; will sleep if full
                            _queue->push_front(img);
                        }
                    }

                    latest_frame = std::atoi(new_files.back().stem().c_str());
                }

                // poll for new files every 100ms
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (...) {
            _ex = std::current_exception();
        }
    });
}

void ImageReader::get(Image *dst)
{
    _queue->pop_back(dst);
}

void ImageReader::stop()
{
    _stop++;
    _queue->clear();
}

ImageReader::~ImageReader()
{
    if (_thread.joinable()) _thread.join();

    delete _queue;
}

}
