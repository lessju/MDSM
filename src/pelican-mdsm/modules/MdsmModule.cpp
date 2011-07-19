// MDSM stuff
#include "MdsmModule.h"
#include "DedispersedSeries.h"
#include "dedispersion_manager.h"

// Pelican stuff
#include "pelican/data/DataBlob.h"
#include "WeightedSpectrumDataSet.h"
// C++ stuff
#include <iostream>
#include <complex>
#include "math.h"
#include <cstdlib>

#include <QString>

using namespace pelican;

// Constructor
MdsmModule::MdsmModule(const ConfigNode& config)
  : AbstractModule(config), _bufferMean(0), _bufferRMS(0), _samples(0), _gettime(0), _counter(0), _iteration(0)
{
    // Configure MDSM Module
    QString _filepath = config.getOption("observationfile", "filepath");
    _createOutputBlob = (bool) config.getOption("createOutputBlob", "value", "0").toUInt();
    _invertChannels = (bool) config.getOption("invertChannels", "value", "1").toUInt();

    // Initialise noise and rms arrays
    for (unsigned i=0; i<MDSM_STAGES; ++i){
      _means[i]=NULL;
      _rmss[i]=NULL;
    }

    // Process Survey parameters (through observation file)
    _survey = processSurveyParameters(_filepath);

    // Start up MDSM
    _input_buffer = initialiseMDSM(_survey);
}

// Destructor
MdsmModule::~MdsmModule()
{
    tearDownMDSM();
}

// Run MDSM
void MdsmModule::run(DataBlob* incoming, DedispersedTimeSeriesF32* dedispersedData)
{

    WeightedSpectrumDataSet* weightedData;
    SpectrumDataSet<float>* streamData;

    float blobMean = 0.0; // Not set at this point
    float blobRMS = 0.0; // Not set at this point


    // Get the mean and rms per chunk from the datablob

    if (( weightedData = (WeightedSpectrumDataSet*) dynamic_cast<WeightedSpectrumDataSet*>(incoming))){
      streamData = weightedData -> dataSet();
      blobMean = weightedData -> mean();
      blobRMS = weightedData -> rms();
      // If you are in the very first block, initialise everything
      
      if (_means[0]==NULL){
        unsigned chunksPerBlock = (_survey -> nsamp + _survey -> maxshift)
          /streamData -> nTimeBlocks();
        
        _survey -> samplesPerChunk = streamData -> nTimeBlocks();

        /*
        std::cout << ">>>>>>> MdsmModule: chunksPerBlock: " << chunksPerBlock 
                  << " _survey -> nchans:   " << _survey -> nchans
                  << std::endl; 
        */
        for (unsigned i=0; i < MDSM_STAGES; ++i) {
          // allocate memory for means and rmss of as many chunks fit in an MDSM block
          _means[i] = (float*) malloc((chunksPerBlock+1) * sizeof(float));
          _rmss[i] = (float*) malloc((chunksPerBlock+1) * sizeof(float));
        }
      }

      // Compute where we are in the array of means and rmss,
      // depending on whether we have a maxhift bit at the front
      
      // index is the place to write the mean and rms given the
      // current chunk. It takes into account the fact that for
      // buffers after the first, the first values are just copied
      // from the maxshift part of the previous buffer

      unsigned index = ( _counter == 0 ) ? _samples / streamData -> nTimeBlocks() 
        : _samples / streamData -> nTimeBlocks() +
        floor(_survey -> maxshift/(float) streamData -> nTimeBlocks());

      _means[_counter%MDSM_STAGES][index] = 
        weightedData -> mean() * _survey -> nchans;

      _rmss[_counter%MDSM_STAGES][index] = 
        weightedData -> rms() * _survey -> nchans;
    } 
    else
      {
          throw "MDSM:  No useful datablob";
      }
    
    unsigned nSamples, nSubbands, nChannels, reqSamp, copySamp;
    nSamples = streamData -> nTimeBlocks();
    nSubbands = streamData -> nSubbands();
    nChannels = (nSamples == 0) ? 0 : streamData->nChannels();
    float *data;


    // We need the timestamp of the first packet in the first blob (assuming that we don't
    // lose any packets), and the sampling time. This will give each sample a unique timestamp
    // _blockRate currently contains the number of time samples per chunk... not very useful
    if (_gettime == 0) {
        _timestamp = streamData -> getLofarTimestamp();
        _blockRate = streamData -> getBlockRate();

        if (_counter > 0)
            _timestamp = streamData -> getLofarTimestamp() - _blockRate * _survey -> maxshift;
    }

    // Calculate number of required samples
    reqSamp = _counter == 0 ? _survey -> nsamp + _survey -> maxshift : _survey -> nsamp;
    unsigned blobsPerBlock = ceil(reqSamp / (float) nSamples);

    // Check to see whether all samples will fit in memory
    copySamp = nSamples <= reqSamp - _samples ? nSamples : reqSamp - _samples;
    // Check if we reached the end of the stream, in which case we clear the MDSM buffers
    if (nSamples == 0) {
        reqSamp = 0;
        copySamp = 0;
    }

    // NOTE: we always care about the first Stokes parameter (XX)
    for(unsigned t = 0; t < copySamp; t++) {
        for (unsigned s = 0; s < nSubbands; s++) {
            data = streamData -> spectrumData(t, (_invertChannels) ? nSubbands - 1 - s : s, 0);
            for (unsigned c = 0; c < nChannels; c++)
//                _input_buffer[(_samples + t) * nSubbands * nChannels
//                              + s * nChannels + c] = data[(_invertChannels) ? nChannels - 1 - c : c];

                  // Corner turn first...
                if (_counter == 0)
                    _input_buffer[s * nChannels * (_survey -> nsamp + _survey -> maxshift)
                                  + c * (_survey -> nsamp + _survey -> maxshift)
                                  + (_samples + t)] = data[(_invertChannels) ? nChannels - 1 - c : c];
                else
                    _input_buffer[s * nChannels * (_survey -> nsamp + _survey -> maxshift)
                                  + c * (_survey -> nsamp + _survey -> maxshift)
                                  + _survey -> maxshift + _samples + t] = data[(_invertChannels) ? nChannels - 1 - c : c];

        }
    }

    if (blobMean != 0.0){
      _bufferMean += blobMean;
      _bufferRMS += blobRMS;
    }
    _samples += copySamp;
    _gettime = _samples;

    // We have enough samples to pass to MDSM
    if (_samples == reqSamp || nSamples == 0) {
        // Copy this chunk and get previous output
        unsigned int numSamp;
        unsigned samples;
        
        numSamp = (_counter == 0) ? _samples - _survey -> maxshift : _samples;
        
        std::cout<< " Sending data to MDSM" 
                 << " Average Mean is: " << _bufferMean/blobsPerBlock
                 << " First Mean is: " << _means[_counter%MDSM_STAGES][0] 
                 << " RMS is : " <<  _bufferRMS/blobsPerBlock
                 << " Chunks in Block : " <<  blobsPerBlock
                 << std::endl;
        float *outputBuffer = next_chunk(numSamp, samples, _timestamp, _blockRate, 
                                         _means[_counter%MDSM_STAGES], _rmss[_counter%MDSM_STAGES] );
        
        _bufferMean = 0;
        _bufferRMS = 0;

        // Copy remaining samples (if any) to MDSM input buffer
        _gettime = _samples;
        if (!start_processing(_samples))  return;

        if (outputBuffer != NULL && _createOutputBlob) {
        
            // Output available, create data blob
            dedispersedData -> resize(_survey -> tdms);
            if (_survey -> useBruteForce) {
                // All DMs have same number of samples
                DedispersedSeries<float>* data;
                for (unsigned d = 0; d < _survey -> tdms; d++) {
                    data  = dedispersedData -> samples(d);
                    data -> resize(samples);
                    data -> setDmValue(_survey -> lowdm + _survey -> dmstep * d);
                    memcpy(data -> ptr(), &outputBuffer[d * samples], samples * sizeof(float));
                }
            }
            else {
                // Number of samples differs among passes
                DedispersedSeries<float>* data;
                unsigned totdms = 0, shift = 0; 
                for (unsigned thread = 0; thread < _survey -> num_threads; thread++) {
                    for(unsigned pass = 0; pass < _survey -> num_passes; pass++) {
                        unsigned ndms = (_survey -> pass_parameters[pass].ncalls / _survey -> num_threads) 
                                       * _survey -> pass_parameters[pass].calldms;

                        float startdm = _survey -> pass_parameters[pass].lowdm + _survey -> pass_parameters[pass].sub_dmstep 
                                        * (_survey -> pass_parameters[pass].ncalls / _survey -> num_threads) * thread;
                        float dmstep  = _survey -> pass_parameters[pass].dmstep;

                        unsigned nsamp = samples / _survey -> pass_parameters[pass].binsize;
                        for(unsigned dm = 0; dm < ndms; dm++) {
                            data = dedispersedData -> samples(totdms + dm);
                            data -> resize(nsamp);
                            data -> setDmValue(startdm + dm * dmstep);
                            memcpy(data -> ptr(), &outputBuffer[shift], nsamp * sizeof(float));
                            shift += nsamp;
                        }
                        totdms += ndms;
                    }   
                }
            }
        }
        else
            dedispersedData -> resize(0);

        _counter++;
        _samples = 0;
        _gettime = _samples;

        // take the means and rmss from the maxshift part of the
        // previous block and copy them to the start of the next block
        if (_means[0] != NULL){
          memcpy(_means[_counter%MDSM_STAGES], 
                 _means[(_counter-1)%MDSM_STAGES]+_survey -> nsamp / nSamples,
                 (_survey -> maxshift / nSamples) * sizeof(float));
          memcpy(_rmss[_counter%MDSM_STAGES], 
                 _rmss[(_counter-1)%MDSM_STAGES]+_survey -> nsamp / nSamples,
                 (_survey -> maxshift / nSamples) * sizeof(float));
        }

        // Copy remaining samples of last chunk into next block for
        // processing

        for(unsigned t = copySamp; t < nSamples; t++) {
            for (unsigned s = 0; s < nSubbands; s++) {
                data = streamData -> spectrumData(t, (_invertChannels) ? nSubbands - 1 - s : s, 0);
                for(unsigned c = 0 ; c < nChannels ; ++c)
//                      _input_buffer[(t - copySamp) * nSubbands * nChannels
//                                    + s * nChannels + c] = data[(_invertChannels) ? nChannels - 1 - c : c];

                  // Corner turn first...
                    _input_buffer[s * nChannels * (_survey -> nsamp + _survey -> maxshift)
                                + c * (_survey -> nsamp + _survey -> maxshift)
                                + _survey -> maxshift + _samples + t] = data[(_invertChannels) ? nChannels - 1 - c : c];

            }
        }
        _samples += nSamples - copySamp;
    }
    else
        dedispersedData -> resize(0);

    _iteration++;
}
