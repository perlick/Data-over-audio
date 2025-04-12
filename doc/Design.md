# Capital 'D' Design Doc

# Overview

# Goals and non-Goals

# Milestones

# Technical Design

Most of the DSP logic is written to follow http://pysdr.org.

## TX DSP Chain

The TX Chain is made up of two parts, 
1. **The Data Modulation Function (DMF)**: reads a packet of data and modulates it into IQ. 
1. **The Front-End Loop (FEL)**: acts as the "SDR" front-end device. The free running loop accepts a stream of IQ and plays a signal from the speaker device.

### Data Modulation Function

The DMF (Data Modulation Function) performs the following DSP tasks.
1. Channel coding
1. Data to IQ Symbol coversion
1. Upscale and filter
1. Send to buffer

Note that this function is run on demand. Its' input is a full L2 packet of data. It will modulate this packet and write the corresponding IQ stream to its' outpu buffer. The DMF outputs directly to the FEL. Because the FEL is constantly reading the DMF's output buffer, it may become empty and the FEL is designed to function in that case.  

### Front-End Loop

The Front-End Loop (FEL) is responsible for simulating the DSP device in a typical RF set up. It will perform the following tasks. 

1. Read IQ samples from the DMF output buffer
1. Mix IQ w/ Lo
1. Add I and Q parts together
1. Send samples to be played

It will read DSP from the DMF output buffer (when avialable) and continuously supply audio samples to the system speakers. The system audio buffer should never underflow as this could cause loss of phase sync. 

## Note on the Local Oscillator 

In the SDR setup described on pysdr.org, a physical Lo is used for signal mixing, then addition of I and Q happens. These particular hardware singal processing steps are not going to be possible on a consumer device so we must simulate them in digital. All we can send the speakers is a 1D stream of samples. These samples will be similar to the RF signal right before it reaches the antenna element. Thus, the signal processing steps to be sumilated are

1. Lo (and 90 deg phas rotation)
1. Mixers
1. Adder

The Lo will need to produce a digital signal representing a sine wave. So, it will need some concept of time so that it can produce the correct frequency. Fortunately, the Lo does not need any concept of the global time. The FEL and system speakers agree that the audio stream should be played at a certain rate. The FEL can tune it's Lo to produce a sine wave based on that sample rate. The RX DSP Chain is designed to handle some phase drift or frequency offset if the speakers are not running exactly at the requested sample rate.

## RX DSP Chain

## Code Modules

1. DSP Modulation library
1. System audio managers
1. System network drivers
1. CLI interface

# Testability, Monitoring, Alerting, and Debugability
