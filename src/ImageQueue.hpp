#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/call_traits.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <string>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <chrono>
#include <exception>

#include "strnatcmp/strnatcmp.h"
#include "TinyTIFF/tinytiffwriter.h"

namespace YipLab {

namespace fs = boost::filesystem;

typedef struct
{
    cv::Mat mat;
    std::string str;
} Image;

// this implements a fixed-size blocking queue, and is largely copied from a Boost example
// plus some hacks that work for our application
template <class T>
class BoundedBuffer {
public:

    typedef boost::circular_buffer<T> container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename boost::call_traits<value_type>::param_type param_type;

    explicit BoundedBuffer(size_type capacity) : m_unread(0), m_container(capacity) {}

    bool push_front(param_type item) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_not_full.wait(lock, std::bind(&BoundedBuffer<value_type>::is_not_full, this));
        if (!m_stop)
        {
            m_container.push_front(item);
            ++m_unread;
            lock.unlock();
            m_not_empty.notify_one();
            return true;
        }
        else
        {
            lock.unlock();
            return false;
        }
    }

    bool pop_back(value_type* pItem) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_not_empty.wait(lock, std::bind(&BoundedBuffer<value_type>::is_not_empty, this));
        if (!m_stop)
        {
            *pItem = m_container[--m_unread];
            lock.unlock();
            m_not_full.notify_one();
            return true;
        }
        else
        {
            lock.unlock();
            return false;
        }
    }

    void stop() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_stop = true;
        lock.unlock();
        m_not_empty.notify_one();
        m_not_full.notify_one();
    }

    bool is_empty() {
        std::unique_lock<std::mutex> lock(m_mutex);
        bool empty = m_unread == 0;
        lock.unlock();
        return empty;
    }

    bool is_stopped() {
        std::unique_lock<std::mutex> lock(m_mutex);
        bool stop = m_stop;
        lock.unlock();
        return stop;
    }

private:
    BoundedBuffer(const BoundedBuffer&);              // Disabled copy constructor
    BoundedBuffer& operator = (const BoundedBuffer&); // Disabled assign operator

    bool is_not_empty() const { return m_unread > 0 || m_stop; }
    bool is_not_full() const { return m_unread < m_container.capacity() || m_stop; }

    size_type m_unread;
    container_type m_container;
    std::mutex m_mutex;
    std::condition_variable m_not_empty;
    std::condition_variable m_not_full;
    bool m_stop = false;
};

class ImageReader
{
public:
    ImageReader(const fs::path, const int);
    void start();
    void get(Image *);
    void finish();
    ~ImageReader();

private:
    fs::path input_path;
    //int latest_frame;
    std::string latest_name;
    BoundedBuffer<Image> *_queue = nullptr;
    std::thread _thread;
    std::exception_ptr _ex = nullptr;
    std::mutex _mutex;

    const static int MAX_N_TRIES = 30;
};

class ImageWriter
{
public:
    ImageWriter(const fs::path, const int);
    void start();
    void write(const Image);
    void finish();
    ~ImageWriter();

private:
    fs::path output_path;
    BoundedBuffer<Image> *_queue = nullptr;
    std::thread _thread;
    std::exception_ptr _ex = nullptr;
    std::mutex _mutex;

    const static int MAX_N_TRIES = 30;
};

}
