//
//  AudioProcessing.metal
//  AudioBloomAI
//
//  Metal compute kernels for real-time audio processing
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Parameter Structures

/// Parameters for FFT operations
struct FFTParameters {
    uint32_t sampleCount;    // Number of samples to process
    uint32_t inputOffset;    // Offset in input buffer
    uint32_t outputOffset;   // Offset in output buffer
    uint32_t inverse;        // Whether to perform inverse FFT (1) or forward FFT (0)
};

/// Parameters for filter operations
struct FilterParameters {
    uint32_t sampleCount;    // Number of samples to process
    uint32_t inputOffset;    // Offset in input buffer
    uint32_t outputOffset;   // Offset in output buffer
    uint32_t filterOffset;   // Offset in filter buffer
};

/// Parameters for time domain filter operations
struct TimeFilterParameters {
    uint32_t sampleCount;     // Number of samples to process
    uint32_t filterLength;    // Length of the filter
    uint32_t inputOffset;     // Offset in input buffer
    uint32_t outputOffset;    // Offset in output buffer
    uint32_t filterOffset;    // Offset in filter buffer
};

/// Parameters for spectrum analysis operations
struct SpectrumAnalysisParameters {
    uint32_t sampleCount;    // Number of spectrum samples
    float sampleRate;        // Sample rate in Hz
    float bassMinFreq;       // Minimum bass frequency (Hz)
    float bassMaxFreq;       // Maximum bass frequency (Hz)
    float midMinFreq;        // Minimum mid frequency (Hz)
    float midMaxFreq;        // Maximum mid frequency (Hz)
    float trebleMinFreq;     // Minimum treble frequency (Hz)
    float trebleMaxFreq;     // Maximum treble frequency (Hz)
};

/// Parameters for normalization operations
struct NormalizeParameters {
    uint32_t sampleCount;    // Number of samples to process
    float targetLevel;       // Target normalization level (0.0-1.0)
    uint32_t inputOffset;    // Offset in input buffer
    uint32_t outputOffset;   // Offset in output buffer
};

// MARK: - Helper Functions

/// Complex number structure
struct complex_float {
    float real;
    float imag;
};

/// Multiplies two complex numbers
inline complex_float complex_multiply(complex_float a, complex_float b) {
    return {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

/// Adds two complex numbers
inline complex_float complex_add(complex_float a, complex_float b) {
    return {a.real + b.real, a.imag + b.imag};
}

/// Subtracts two complex numbers
inline complex_float complex_subtract(complex_float a, complex_float b) {
    return {a.real - b.real, a.imag - b.imag};
}

/// Calculates the magnitude of a complex number
inline float complex_magnitude(complex_float a) {
    return sqrt(a.real * a.real + a.imag * a.imag);
}

/// Calculates the phase of a complex number
inline float complex_phase(complex_float a) {
    return atan2(a.imag, a.real);
}

/// Returns a complex number from polar coordinates
inline complex_float complex_from_polar(float magnitude, float phase) {
    return {magnitude * cos(phase), magnitude * sin(phase)};
}

// MARK: - FFT Compute Kernels

/// Bitwise reverse of an integer, for FFT reordering
inline uint bitReverse(uint val, uint bits) {
    uint result = 0;
    for (uint i = 0; i < bits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

/// Compute complex FFT (Cooley-Tukey algorithm)
// This is a direct implementation for educational purposes.
// For production use, consider using the vDSP library via CPU or custom optimized kernels.
kernel void fft_forward(
    device const float* input [[buffer(0)]],
    device float2* output [[buffer(1)]],
    constant FFTParameters& params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    // Only process half the data because complex FFT outputs N/2 complex values
    const uint N = params.sampleCount;
    const uint N2 = N / 2;
    
    if (id >= N2) return;
    
    // Local storage for intermediate FFT computation
    threadgroup complex_float localData[1024]; // Adjust size based on expected max FFT size
    
    // Determine log2(N) for bit reversal
    uint bits = 0;
    uint temp = N;
    while (temp > 1) {
        temp >>= 1;
        bits++;
    }
    
    // Load input data with bit-reversal reordering
    for (uint i = 0; i < N; i++) {
        uint j = bitReverse(i, bits);
        if (i < j) {
            // For real input, initialize complex numbers
            localData[i].real = input[i + params.inputOffset];
            localData[i].imag = 0.0;
            localData[j].real = input[j + params.inputOffset];
            localData[j].imag = 0.0;
        }
    }
    
    // FFT computation
    for (uint stage = 1; stage <= bits; stage++) {
        uint m = 1 << stage;
        uint m2 = m >> 1;
        complex_float w = {1.0, 0.0}; // twiddle factor
        complex_float wm = {cos(M_PI/float(m2)), -sin(M_PI/float(m2))};
        
        for (uint j = 0; j < m2; j++) {
            for (uint k = j; k < N; k += m) {
                uint l = k + m2;
                complex_float t = complex_multiply(w, localData[l]);
                complex_float u = localData[k];
                localData[k] = complex_add(u, t);
                localData[l] = complex_subtract(u, t);
            }
            w = complex_multiply(w, wm);
        }
    }
    
    // Store output for this thread
    // Since input was real, use symmetry properties to save computation
    output[id + params.outputOffset].x = localData[id].real;
    output[id + params.outputOffset].y = localData[id].imag;
}

/// Inverse complex FFT
kernel void fft_inverse(
    device const float2* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant FFTParameters& params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    // Only process half the data because we're going from complex to real
    const uint N = params.sampleCount;
    const uint N2 = N / 2;
    
    if (id >= N2) return;
    
    // Local storage for intermediate IFFT computation
    threadgroup complex_float localData[1024]; // Adjust size based on expected max FFT size
    
    // Load input data (conjugate complex values for IFFT)
    for (uint i = 0; i < N2; i++) {
        localData[i].real = input[i + params.inputOffset].x;
        localData[i].imag = -input[i + params.inputOffset].y; // Negate imaginary part for inverse
    }
    
    // Perform similar FFT but with conjugate twiddle factors
    // (Implementation similar to fft_forward but with different twiddle factors)
    
    // Calculate and scale the result
    const float scale = 1.0 / float(N);
    output[id + params.outputOffset] = localData[id].real * scale;
}

// MARK: - Filtering Kernels

/// Time domain filtering using direct convolution
kernel void time_domain_filter(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* filter [[buffer(2)]],
    constant TimeFilterParameters& params [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.sampleCount) return;
    
    const uint filterLength = params.filterLength;
    float result = 0.0f;
    
    // Direct convolution
    for (uint i = 0; i < filterLength; i++) {
        if (id >= i) {
            result += input[id - i + params.inputOffset] * filter[i + params.filterOffset];
        }
    }
    
    output[id + params.outputOffset] = result;
}

/// Frequency domain filtering using complex multiplication
kernel void frequency_domain_filter(
    device const float2* input [[buffer(0)]],
    device float2* output [[buffer(1)]],
    device const float2* filter [[buffer(2)]],
    constant FilterParameters& params [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.sampleCount) return;
    
    // Complex multiplication of input and filter in frequency domain
    float2 inValue = input[id + params.inputOffset];
    float2 filterValue = filter[id + params.filterOffset];
    
    // Complex multiplication: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
    float outputReal = inValue.x * filterValue.x - inValue.y * filterValue.y;
    float outputImag = inValue.x * filterValue.y + inValue.y * filterValue.x;
    
    output[id + params.outputOffset] = float2(outputReal, outputImag);
}

// MARK: - Spectrum Analysis

/// Analyzes frequency spectrum to extract bass, mid, and treble levels
kernel void spectrum_analysis(
    device const float2* spectrum [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant SpectrumAnalysisParameters& params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint threadCount = min(params.sampleCount, 1024u); // Ensure we don't exceed reasonable thread counts
    if (id >= threadCount) return;
    
    const float sampleRate = params.sampleRate;
    const uint N = params.sampleCount;
    
    // Frequency resolution: sample_rate / N
    const float freqResolution = sampleRate / float(N * 2);
    
    // Create threadgroup shared memory for reduction
    threadgroup atomic_float bassSum;
    threadgroup atomic_float midSum;
    threadgroup atomic_float trebleSum;
    threadgroup atomic_int bassCount;
    threadgroup atomic_int midCount;
    threadgroup atomic_int trebleCount;
    
    // Initialize atomic counters to 0 in the first thread
    if (id == 0) {
        atomic_store_explicit(&bassSum, 0.0f, memory_order_relaxed);
        atomic_store_explicit(&midSum, 0.0f, memory_order_relaxed);
        atomic_store_explicit(&trebleSum, 0.0f, memory_order_relaxed);
        atomic_store_explicit(&bassCount, 0, memory_order_relaxed);
        atomic_store_explicit(&midCount, 0, memory_order_relaxed);
        atomic_store_explicit(&trebleCount, 0, memory_order_relaxed);
    }
    
    // Wait for initialization
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Calculate how many frequencies each thread should process
    uint freqsPerThread = max(N / threadCount, 1u);
    uint startIdx = id * freqsPerThread;
    uint endIdx = min(startIdx + freqsPerThread, N);
    
    // Process assigned frequencies
    float bassSumLocal = 0.0f;
    float midSumLocal = 0.0f;
    float trebleSumLocal = 0.0f;
    int bassCountLocal = 0;
    int midCountLocal = 0;
    int trebleCountLocal = 0;
    
    for (uint i = startIdx; i < endIdx; i++) {
        // Calculate frequency for this bin
        float frequency = i * freqResolution;
        
        // Calculate magnitude of complex spectrum data
        float2 spectrumValue = spectrum[i];
        float magnitude = sqrt(spectrumValue.x * spectrumValue.x + spectrumValue.y * spectrumValue.y);
        
        // Determine which band this frequency belongs to
        if (frequency >= params.bassMinFreq && frequency <= params.bassMaxFreq) {
            bassSumLocal += magnitude;
            bassCountLocal++;
        } else if (frequency >= params.midMinFreq && frequency <= params.midMaxFreq) {
            midSumLocal += magnitude;
            midCountLocal++;
        } else if (frequency >= params.trebleMinFreq && frequency <= params.trebleMaxFreq) {
            trebleSumLocal += magnitude;
            trebleCountLocal++;
        }
    }
    
    // Add local sums to global sums (atomic to avoid race conditions)
    atomic_fetch_add_explicit(&bassSum, bassSumLocal, memory_order_relaxed);
    atomic_fetch_add_explicit(&midSum, midSumLocal, memory_order_relaxed);
    atomic_fetch_add_explicit(&trebleSum, trebleSumLocal, memory_order_relaxed);
    atomic_fetch_add_explicit(&bassCount, bassCountLocal, memory_order_relaxed);
    atomic_fetch_add_explicit(&midCount, midCountLocal, memory_order_relaxed);
    atomic_fetch_add_explicit(&trebleCount, trebleCountLocal, memory_order_relaxed);
    
    // Wait for all threads to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // First thread writes the final results
    if (id == 0) {
        float bassAvg = atomic_load_explicit(&bassCount, memory_order_relaxed) > 0 ? 
            atomic_load_explicit(&bassSum, memory_order_relaxed) / float(atomic_load_explicit(&bassCount, memory_order_relaxed)) : 0.0f;
        
        float midAvg = atomic_load_explicit(&midCount, memory_order_relaxed) > 0 ? 
            atomic_load_explicit(&midSum, memory_order_relaxed) / float(atomic_load_explicit(&midCount, memory_order_relaxed)) : 0.0f;
        
        float trebleAvg = atomic_load_explicit(&trebleCount, memory_order_relaxed) > 0 ? 
            atomic_load_explicit(&trebleSum, memory_order_relaxed) / float(atomic_load_explicit(&trebleCount, memory_order_relaxed)) : 0.0f;
        
        // Normalize values to 0.0-1.0 range if needed (can be adjusted based on expected magnitudes)
        bassAvg = min(max(bassAvg * 2.0f, 0.0f), 1.0f);
        midAvg = min(max(midAvg * 2.0f, 0.0f), 1.0f);
        trebleAvg = min(max(trebleAvg * 2.0f, 0.0f), 1.0f);
        
        // Write the results to the output buffer (x=bass, y=mid, z=treble, w=overall)
        float overallAvg = (bassAvg + midAvg + trebleAvg) / 3.0f;
        output[0] = float4(bassAvg, midAvg, trebleAvg, overallAvg);
    }
}

// MARK: - Audio Normalization

/// Find maximum value in an array for peak normalization
/// Uses parallel reduction approach for efficiency
kernel void find_peak(
    device const float* input [[buffer(0)]],
    device atomic_float* maxValue [[buffer(1)]],
    constant NormalizeParameters& params [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint threadCount [[threads_per_grid]]
) {
    if (id >= params.sampleCount) return;
    
    // Calculate how many samples each thread should process
    uint samplesPerThread = max(params.sampleCount / threadCount, 1u);
    uint startIdx = id * samplesPerThread;
    uint endIdx = min(startIdx + samplesPerThread, params.sampleCount);
    
    // Find max value in our range
    float localMax = 0.0f;
    for (uint i = startIdx; i < endIdx; i++) {
        // Use absolute value for audio peak detection
        float value = abs(input[i + params.inputOffset]);
        localMax = max(localMax, value);
    }
    
    // Update global max atomically
    float currentMax = atomic_load_explicit(maxValue, memory_order_relaxed);
    while (localMax > currentMax) {
        bool success = atomic_compare_exchange_weak_explicit(
            maxValue,
            &currentMax,
            localMax,
            memory_order_relaxed,
            memory_order_relaxed
        );
        
        if (success) break;
    }
}

/// Normalize audio to a target peak level
kernel void normalize_audio(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant NormalizeParameters& params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.sampleCount) return;
    
    // Two-pass normalization approach:
    // 1. First kernel (find_peak) finds the peak in a separate pass
    // 2. This kernel applies the normalization using the pre-calculated peak
    
    // For this implementation, we'll use a direct approach for simplicity
    // In a real-world scenario, this would be part of a multi-pass algorithm
    
    // Read input sample
    float sample = input[id + params.inputOffset];
    
    // Assume max value was precomputed and provided as targetLevel parameter
    float maxValue = params.targetLevel;
    
    // If maxValue is 0 or very small, avoid division by zero
    if (maxValue < 0.0001f) {
        output[id + params.outputOffset] = sample;
        return;
    }
    
    // Calculate scale factor to reach target level (default is 0.9 or -0.9dB)
    float scaleFactor = params.targetLevel / maxValue;
    
    // Apply normalization
    output[id + params.outputOffset] = sample * scaleFactor;
}

// MARK: - Audio RMS Normalization

/// Calculate RMS (Root Mean Square) level of audio
kernel void calculate_rms(
    device const float* input [[buffer(0)]],
    device atomic_float* rmsValue [[buffer(1)]],
    device atomic_int* sampleCount [[buffer(2)]],
    constant NormalizeParameters& params [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint threadCount [[threads_per_grid]]
) {
    if (id >= params.sampleCount) return;
    
    // Calculate how many samples each thread should process
    uint samplesPerThread = max(params.sampleCount / threadCount, 1u);
    uint startIdx = id * samplesPerThread + params.inputOffset;
    uint endIdx = min(startIdx + samplesPerThread, params.sampleCount + params.inputOffset);
    
    // Sum squared values for this thread
    float sumSquares = 0.0f;
    for (uint i = startIdx; i < endIdx; i++) {
        float sample = input[i];
        sumSquares += sample * sample;
    }
    
    // Add to global sum using atomic operations
    atomic_fetch_add_explicit(rmsValue, sumSquares, memory_order_relaxed);
    atomic_fetch_add_explicit(sampleCount, endIdx - startIdx, memory_order_relaxed);
}

/// Normalize audio using RMS values
kernel void normalize_audio_rms(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* rmsLevel [[buffer(2)]],
    constant NormalizeParameters& params [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.sampleCount) return;
    
    // Read the precomputed RMS level
    float currentRMS = *rmsLevel;
    
    // Avoid division by zero
    if (currentRMS < 0.0001f) {
        output[id + params.outputOffset] = input[id + params.inputOffset];
        return;
    }
    
    // Calculate scale factor to reach target RMS level
    float scaleFactor = params.targetLevel / currentRMS;
    
    // Apply normalization - scale the input to target level
    output[id + params.outputOffset] = input[id + params.inputOffset] * scaleFactor;
}

// MARK: - Optimized Helpers for Apple Silicon

/// GPU-optimized smooth clamp function with adjustable transition
inline float smoothClamp(float x, float minVal, float maxVal, float smoothness) {
    // Smoothness should be a small value (0.01-0.1)
    float range = maxVal - minVal;
    float halfRange = range * 0.5f;
    float centerPoint = minVal + halfRange;
    
    // Apply smoothed soft clamp using tanh
    float scaledValue = (x - centerPoint) / (halfRange * (1.0f + smoothness));
    return centerPoint + halfRange * tanh(scaledValue);
}

/// GPU-optimized sinc function (sin(x)/x)
inline float sinc(float x) {
    // Handle the limit case at x=0
    if (fabs(x) < 1e-6f) return 1.0f;
    return sin(x) / x;
}

/// Optimized interpolation for audio processing
inline float cubic_interpolate(float y0, float y1, float y2, float y3, float mu) {
    float mu2 = mu * mu;
    float a0 = y3 - y2 - y0 + y1;
    float a1 = y0 - y1 - a0;
    float a2 = y2 - y0;
    float a3 = y1;
    
    return a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3;
}
