# Data-over-audio
Attempt to experiment with SDR concepts using audio as the channel

## TX Architecture

Linux Processes
    DSP task

1. One Demand Tasks 
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
