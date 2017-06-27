#include "ImageQueue.hpp"

// TODO: complementary ImageWriter for storage
// TODO: exception handling

namespace YipLab {

ImageReader::ImageReader(const fs::path input_path, const int buffer_size)
{
    this->input_path = input_path;

    _queue = new BoundedBuffer<Image>(buffer_size);
    this->latest_frame = -1;
}

void ImageReader::start()
{
    // watch the folder
    _thread = std::thread([&](){
        try {
            while ( !_queue->is_stopped() )
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
                    bool is_new = std::atoi(this_path.stem().c_str()) > latest_frame;
                    bool is_valid = this_path.stem().string()[0] != '.';
                    bool is_tiff = this_path.extension() == ".tif" || this_path.extension() == ".tiff";
                    if (is_new && is_valid && is_tiff)
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
                            mat = cv::imread(p.string(), CV_LOAD_IMAGE_GRAYSCALE); // TIFF
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        } while ( mat.rows == 0 && ++num_tries < MAX_N_TRIES );

                        if ( !f_exists || num_tries == MAX_N_TRIES )
                        {
                            std::cout << p.string() << " read timed out" << std::endl;
                        }
                        else
                        {
                            Image img = { mat, p.string() };

                            // push to image queue; will sleep if full, return false if queue stopped
                            if ( !_queue->push_front(img) ) break;

                            latest_frame = std::atoi(p.stem().c_str());
                        }
                    }
                }

                // poll for new files every 100ms
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (...) {
            std::unique_lock<std::mutex> l(_mutex);
            _ex = std::current_exception();
            l.unlock();
        }
    });
}

void ImageReader::get(Image *dst)
{
    std::unique_lock<std::mutex> l(_mutex);
    if (_ex) std::rethrow_exception(_ex);
    l.unlock();

    _queue->pop_back(dst);
}

void ImageReader::finish()
{
    std::unique_lock<std::mutex> l(_mutex);
    if (_ex) std::rethrow_exception(_ex);
    l.unlock();

    _queue->stop();
}

ImageReader::~ImageReader()
{
    std::unique_lock<std::mutex> l(_mutex);
    if (_ex) std::rethrow_exception(_ex);
    l.unlock();

    if (_thread.joinable()) _thread.join();

    delete _queue;
}

ImageWriter::ImageWriter(const fs::path output_path, const int buffer_size)
{
    this->output_path = output_path;

    _queue = new BoundedBuffer<Image>(buffer_size);
}

void ImageWriter::start()
{
    // consume images in queue
    _thread = std::thread([&](){
        try {
            while (true)
            {
                Image img;

                if ( _queue->pop_back(&img) )
                {
                    // have to use TinyTIFF here! OpenCV C++ doesn't support FP32 TIFF
                    int bit_depth;
                    switch (img.mat.type())
                    {
                        case CV_8U:  bit_depth = 8;  break;
                        case CV_16U: bit_depth = 16; break;
                        case CV_32F: bit_depth = 32; break;
                    }

                    TinyTIFFFile* tif = TinyTIFFWriter_open(img.str.c_str(), bit_depth, img.mat.rows, img.mat.cols);

                    if (tif)
                    {
                        if (bit_depth == 32)
                            TinyTIFFWriter_writeImage(tif, (float *)img.mat.data);
                        else
                            TinyTIFFWriter_writeImage(tif, img.mat.data);
                    }
                    else // no exception handling in TinyTIFF
                    {
                        throw std::runtime_error("TIFF write error");
                    }

                    TinyTIFFWriter_close(tif);

//                    cv::Mat mat(img.mat.rows, img.mat.cols, CV_8U);
//                    cv::normalize(img.mat, img.mat, 1.0, 0.0, cv::NORM_MINMAX, -1);
//                    img.mat.convertTo(mat, CV_8U, 255.f);
//
//                    cv::imwrite(img.str, img.mat);
                }
                else // pop_back returns false if stop command has been triggered
                {
                    break;
                }
            }
        } catch (...) {
            std::unique_lock<std::mutex> l(_mutex);
            _ex = std::current_exception();
            l.unlock();
        }
    });
}

void ImageWriter::write(const Image src) // reference?
{
    std::unique_lock<std::mutex> l(_mutex);
    if (_ex) std::rethrow_exception(_ex);
    l.unlock();

    _queue->push_front(src);
}

void ImageWriter::finish()
{
    std::unique_lock<std::mutex> l(_mutex);
    if (_ex) std::rethrow_exception(_ex);
    l.unlock();

    // this is a bad solution for a few reasons, but it works
    // -- spin-lock until all images consumed
    while ( !_queue->is_empty() )
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    _queue->stop();
}

ImageWriter::~ImageWriter()
{
    std::unique_lock<std::mutex> l(_mutex);
    if (_ex) std::rethrow_exception(_ex);
    l.unlock();

    if (_thread.joinable()) _thread.join();

    delete _queue;
}

}
