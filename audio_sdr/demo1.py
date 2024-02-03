import sys

import port
import data_endpoint
import symbolizer
import repeater
import splitter
import oscillator
import mixer
import player
import sampler
import scaler
import filter
import decimator


OUTPUT_SAMPLE_RATE_HZ = 44.1e3
CARRIER_FREQ = 6.5e3
TX_LO_TUNE = CARRIER_FREQ
SYMBOL_RATE_HZ = 1

INPUT_SAMPLE_RATE = 44.1e3

if __name__ == "__main__" and sys.argv[1] == "tx":
    ds_out = port.Port()
    data_endpoint.HexDataInput(data_string=None, port_out=ds_out)

    sym_out = port.Port()
    symbolizer.Symbolizer(ds_out, sym_out, "QAM")

    rep_out = port.Port()
    repeater.Repeater(sym_out, rep_out, SYMBOL_RATE_HZ, OUTPUT_SAMPLE_RATE_HZ)

    spt_i_out = port.Port()
    spt_q_out = port.Port()
    splitter.IQSplitter(rep_out, spt_i_out, spt_q_out)

    lo_i_out = port.Port()
    oscillator.LocalOscillator(lo_i_out, TX_LO_TUNE, OUTPUT_SAMPLE_RATE_HZ, "sin")

    lo_q_out = port.Port()
    oscillator.LocalOscillator(lo_q_out, TX_LO_TUNE, OUTPUT_SAMPLE_RATE_HZ, "cos")

    mxr_i_out = port.Port()
    mixer.Mixer(spt_i_out, lo_i_out, mxr_i_out)

    mxr_q_out = port.Port()
    mixer.Mixer(spt_q_out, lo_q_out, mxr_q_out)

    adr_out = port.Port()
    adr = mixer.Adder(mxr_i_out, mxr_q_out, adr_out)

    player.WAVPlayer(adr_out, OUTPUT_SAMPLE_RATE_HZ, "demo1_out.wav")

if __name__ == "__main__" and sys.argv[1] == "rx":

    smpr_out = port.Port()
    sampler.WAVSampler(smpr_out, "Record-2024-0131-121827.wav")

    sclr_out = port.Port()
    batch_size = (INPUT_SAMPLE_RATE//CARRIER_FREQ)*25 # should give 25 wavelengths of carrier
    scaler.Scaler(smpr_out, sclr_out, batch_size)

    spltr_1_out = port.Port()
    spltr_2_out = port.Port()
    splitter.RFSplitter(sclr_out, spltr_1_out, spltr_2_out)

    lo_i_out = port.Port()
    oscillator.LocalOscillator(lo_i_out, TX_LO_TUNE, OUTPUT_SAMPLE_RATE_HZ, "sin")

    lo_q_out = port.Port()
    oscillator.LocalOscillator(lo_q_out, TX_LO_TUNE, OUTPUT_SAMPLE_RATE_HZ, "cos")

    mxr_i_out = port.Port()
    mixer.Mixer(spltr_1_out, lo_i_out, mxr_i_out)

    mxr_q_out = port.Port()
    mixer.Mixer(spltr_2_out, lo_q_out, mxr_q_out)

    lpf_i_out = port.Port()
    filter.ButterLowpassFilter(mxr_i_out, lpf_i_out, CARRIER_FREQ, INPUT_SAMPLE_RATE, batch_size)

    lpf_q_out = port.Port()
    filter.ButterLowpassFilter(mxr_q_out, lpf_q_out, CARRIER_FREQ, INPUT_SAMPLE_RATE, batch_size)

    iqj_out = port.Port()
    mixer.IQJoiner(mxr_i_out, mxr_q_out, iqj_out)

    dec_out = port.Port()
    decimator.UnalignedDecimator(iqj_o ut, dec_out, SYMBOL_RATE_HZ, INPUT_SAMPLE_RATE)

    dsm_out = port.Port()
    symbolizer.DeSymbolizer(dec_out, dec_out, "QAM")

    data_endpoint.HexDataOutput(dsm_out)