#pragma once

#include <chrono>
#include <iostream>
#include <fstream>

class Timer
{
private:
  std::chrono::system_clock::time_point m_start;
  std::string m_message;
  static std::ofstream *m_fileStream;
public:
  Timer() : m_start(std::chrono::system_clock::now()), m_message("")
  {
  }

  void start(std::string message);
  void end();
  static void setFileStream(std::ofstream *stream)
  {
    m_fileStream = stream;
  }

  ~Timer()
  {
  }
};

