#include "Timer.h"

std::ofstream *Timer::m_fileStream = nullptr;

void Timer::start(std::string message)
{
  if (!m_fileStream)
  {
    std::cout << "Timer started: \"" << message.c_str() << "\"" << std::endl;
  }
  m_message = message;
  m_start = std::chrono::steady_clock::now();
}

void Timer::end()
{
  double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - m_start).count();
  if (m_fileStream)
  {
    (*m_fileStream) << m_message.c_str() << ", " << duration << std::endl;
  }
  else
  {
    std::cout << "Timer ended: \"" << m_message.c_str() << "\". Duration = " << duration << " seconds." << std::endl;
  }
}