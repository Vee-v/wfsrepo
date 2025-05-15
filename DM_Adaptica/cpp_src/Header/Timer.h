/***************************************************************************/
// Timer.h
// =======
// High Resolution Timer.
// This timer is able to measure the elapsed time with 1 micro-second accuracy
// in both Windows, Linux and Unix system 
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
//
// MODIFIED BY: Riccardo Marogna for ADAPTICA 2009
//
/***************************************************************************/


#ifndef TIMER_H_DEF
#define TIMER_H_DEF

////////////////////////////////////////
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
///////////////////////////////////////

class Timer
{
    
public:
    
    Timer();                                    // default constructor
    ~Timer();                                   // default destructor

    void    start();                             // start timer
    void    stop();                              // stop the timer
    int     getElapsedTimeInSec();                  // get elapsed time in second
    double  getElapsedTimeInMilliSec();          // get elapsed time in milli-second
    double  getElapsedTimeInMicroSec();          // get elapsed time in micro-second
    void    wait(int millis);

private:
    
    double startTimeInMicroSec;                 // starting time in micro-second
    double endTimeInMicroSec;                   // ending time in micro-second
    int    stopped;                             // stop flag 
    
#ifdef WIN32
    LARGE_INTEGER frequency;                    // ticks per second
    LARGE_INTEGER startCount;                   //
    LARGE_INTEGER endCount;                     //
#else
    timeval startCount;                         //
    timeval endCount;                           //
#endif
    
};

#endif // TIMER_H_DEF

