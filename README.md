# Data-over-audio
Attempt to experiment with SDR concepts using audio as the channel

# Build
## Windows 11
First download Build Tools for Visual Studio 2022
```
winget install -e --id Kitware.CMake
cmake -S . -B build -G "Visual Studio 17 2022"
# install fftw3 
# 1. download here 32 bit: http://fftw.org/install/windows.html
# 1. unzip in downloads
# 1. open dev powershell for vscode in project directory top level
# 1. run `lib /def:<path-to-file>`
cmake --build build
```

## Ubuntu 24.04
```
sudo apt install make gcc libasound2-dev libfftw3-dev
make
```

## TX Architecture

Linux Processes
    DSP task

1. On Demand Tasks 
    1. Channel coding
    1. Data to Symbol coversion
    1. Upscale and filter
    1. Send to front end loop
1. Free Running Loop
    1. Mix IQ w/ Lo and add
    1. Send samples to be played

## RX Architecture

Linux Process
    DSP Task

1. Free Running Loop
    1. Capture Audio Samples 
    1. Filter 
    1. Coarse Freq Sync
    1. Time Sync
    1. Fine Freq Sync
    1. Symbol to Data
    1. Frame Detect / Sync
1. On Demand Processes
    1. Channel Decoding
