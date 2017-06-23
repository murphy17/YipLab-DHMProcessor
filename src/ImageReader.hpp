#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/call_traits.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <string>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <functional>
#include <deque>
#include <chrono>
#include <exception>
#include <unordered_set>

//#include "inotify.h"
//#include "FileSystemEvent.h"
#include "strnatcmp/strnatcmp.h"
#include "TinyTIFF/tinytiffreader.h"

namespace YipLab {

namespace fs = boost::filesystem;

typedef struct
{
    cv::Mat mat;
    std::string str;
} Image;

template <class T>
class BoundedBuffer {
public:

    typedef boost::circular_buffer<T> container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename boost::call_traits<value_type>::param_type param_type;

    explicit BoundedBuffer(size_type capacity) : m_unread(0), m_container(capacity) {}

    void push_front(param_type item) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_not_full.wait(lock, std::bind(&BoundedBuffer<value_type>::is_not_full, this));
        m_container.push_front(item);
        ++m_unread;
        lock.unlock();
        m_not_empty.notify_one();
    }

    void pop_back(value_type* pItem) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_not_empty.wait(lock, std::bind(&BoundedBuffer<value_type>::is_not_empty, this));
        *pItem = m_container[--m_unread];
        lock.unlock();
        m_not_full.notify_one();
    }

private:
    BoundedBuffer(const BoundedBuffer&);              // Disabled copy constructor
    BoundedBuffer& operator = (const BoundedBuffer&); // Disabled assign operator

    bool is_not_empty() const { return m_unread > 0; }
    bool is_not_full() const { return m_unread < m_container.capacity(); }

    size_type m_unread;
    container_type m_container;
    std::mutex m_mutex;
    std::condition_variable m_not_empty;
    std::condition_variable m_not_full;
};

class ImageReader
{
public:
    ImageReader(const fs::path, const int);
    void run();
    void get(Image *);
    void stop();
    ~ImageReader();

private:
    fs::path input_path;
    int latest_frame;
    BoundedBuffer<Image> *_queue = nullptr;
    std::thread _thread;
    std::atomic<int> _stop;
    std::exception_ptr _ex = nullptr;

    const static int MAX_N_TRIES = 30;
};

}
