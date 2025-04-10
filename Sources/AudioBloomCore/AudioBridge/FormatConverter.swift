
//
// FormatConverter.swift
// Audio format conversion utilities for AudioBridge
//

import Foundation
import AVFoundation
import Accelerate

/// Format converter for audio processing
@available(macOS 15.0, *)
class FormatConverter: Sendable {
    private let standardFrequencySize = 512
    
    /// Converts audio data to ML-processable format
    /// - Parameter audioData: The audio data to convert
    /// - Returns: ML-processable audio data
    /// - Throws: Error if conversion fails
    func convertForProcessing(_ audioData: AudioData) throws -> [Float] {
        // Handle empty data case
        guard !audioData.frequencyData.isEmpty else {
            return createEmptyProcessableData()
        }
        
        // Normalize frequency data to standard size
        var processableData = normalizeFrequencyData(audioData.frequencyData)
        
        // Add amplitude data at the end
        processableData.append(audioData.levels.left)
        processableData.append(audioData.levels.right)
        
        return processableData
    }
    
    /// Creates empty processable data
    /// - Returns: Empty data array with zeros
    private func createEmptyProcessableData() -> [Float] {
        var result = [Float](repeating: 0, count: standardFrequencySize)
        // Add zero amplitude
        result.append(0)
        result.append(0)
        return result
    }
    
    /// Normalizes frequency data to standard size using efficient algorithms
    /// - Parameter frequencyData: Original frequency data
    /// - Returns: Normalized data
    private func normalizeFrequencyData(_ frequencyData: [Float]) -> [Float] {
        let originalCount = frequencyData.count
        
        // If already the right size, return a copy
        if originalCount == standardFrequencySize {
            return frequencyData
        }
        
        // Create output buffer of standard size
        var result = [Float](repeating: 0, count: standardFrequencySize)
        
        // Use Accelerate framework for efficient resampling
        if originalCount < standardFrequencySize {
            // Upsample (interpolate)
            vDSP_vgenp(
                frequencyData,
                1,
                &result,
                1,
                vDSP_Length(standardFrequencySize),
                vDSP_Length(originalCount)
            )
        } else {
            // Downsample
            let stride = Double(originalCount) / Double(standardFrequencySize)
            
            for i in 0..<standardFrequencySize {
                let originalIndex = Int(Double(i) * stride)
                if originalIndex < originalCount {
                    result[i] = frequencyData[originalIndex]
                }
            }
            
            // Apply smoothing to avoid aliasing
            smoothData(&result)
        }
        
        // Normalize values to [0.0, 1.0] range for ML processing
        normalizeAmplitude(&result)
        
        return result
    }
    
    /// Smooths data using a simple moving average filter
    /// - Parameter data: Data to smooth (modified in place)
    private func smoothData(_ data: inout [Float]) {
        let windowSize = 3
        let halfWindow = windowSize / 2
        let count = data.count
        
        // Create a temporary buffer
        var smoothed = data
        
        // Apply moving average
        for i in halfWindow..<(count - halfWindow) {
            var sum: Float = 0
            for j in -halfWindow...halfWindow {
                sum += data[i + j]
            }
            smoothed[i] = sum / Float(windowSize)
        }
        
        // Copy smoothed data back
        data = smoothed
    }
    
    /// Normalizes amplitude values to [0.0, 1.0] range
    /// - Parameter data: Data to normalize (modified in place)
    private func normalizeAmplitude(_ data: inout [Float]) {
        // Find the min and max values
        var min: Float = 0
        var max: Float = 0
        
        vDSP_minv(data, 1, &min, vDSP_Length(data.count))
        vDSP_maxv(data, 1, &max, vDSP_Length(data.count))
        
        // Skip normalization if the range is too small
        let range = max - min
        if range < 1e-6 {
            // Just center around 0.5 if all values are the same
            var value: Float = 0.5
            vDSP_vfill(&value, &data, 1, vDSP_Length(data.count))
            return
        }
        
        // Apply normalization
        var a = Float(1.0 / range)
        var b = Float(-min * a)
        
        vDSP_vsmsa(data, 1, &a, &b, &data, 1, vDSP_Length(data.count))
    }
    
    /// Converts PCM buffer to frequency data using FFT
    /// - Parameter buffer: The PCM buffer to analyze
    /// - Returns: Frequency spectrum data
    /// - Throws: Error if conversion fails
    func convertBufferToFrequencyData(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        guard let channelData = buffer.floatChannelData,
              buffer.frameLength > 0 else {
            throw AudioBridge.AudioBridgeError.dataConversionFailed
        }
        
        // Extract sample data from buffer
        let count = Int(buffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: count))
        
        // Prepare FFT
        let fftSize = 1024 // Use power of 2 for efficiency
        let fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.FORWARD
        )
        
        // Prepare input data (apply windowing)
        var windowedInput = [Float](repeating: 0, count: fftSize)
        var window = [Float](repeating: 0, count: fftSize)
        
        // Create Hann window
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        
        // Copy input data (zero-pad if needed)
        let sampleCount = min(count, fftSize)
        for i in 0..<sampleCount {
            windowedInput[i] = samples[i] * window[i]
        }
        
        // Prepare for FFT
        var realInput = windowedInput
        var imagInput = [Float](repeating: 0, count: fftSize)
        var realOutput = [Float](repeating: 0, count: fftSize)
        var imagOutput = [Float](repeating: 0, count: fftSize)
        
        // Perform FFT
        vDSP_DFT_Execute(
            fftSetup!,
            realInput, imagInput,
            &realOutput, &imagOutput
        )
        
        // Calculate magnitude
        var magnitude = [Float](repeating: 0, count: fftSize/2)
        for i in 0..<fftSize/2 {
            let real = realOutput[i]
            let imag = imagOutput[i]
            magnitude[i] = sqrt(real * real + imag * imag)
        }
        
        // Scale the magnitudes
        var scale = Float(1.0 / Float(fftSize))
        vDSP_vsmul(magnitude, 1, &scale, &magnitude, 1, vDSP_Length(fftSize/2))
        
        // Apply logarithmic scaling for better visualization
        var scaledMagnitude = [Float](repeating: 0, count: fftSize/2)
        for i in 0..<fftSize/2 {
            // Convert to dB with some scaling and clamping
            let logValue = 10.0 * log10f(magnitude[i] + 1e-9)
            // Normalize to 0.0-1.0 range
            let normalizedValue = (logValue + 90.0) / 90.0
            scaledMagnitude[i] = min(max(normalizedValue, 0.0), 1.0)
        }
        
        // Destroy FFT setup
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
        
        return scaledMagnitude
    }
    
    /// Creates a PCM buffer from processable data
    /// - Parameters:
    ///   - data: The processable data
    ///   - format: The audio format
    /// - Returns: A PCM buffer
    /// - Throws: Error if conversion fails
    func createBufferFromProcessableData(_ data: [Float], format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        // Extract audio data (excluding level information at the end)
        let audioSampleCount = data.count - 2
        
        guard audioSampleCount > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audioSampleCount)) else {
            throw AudioBridge.AudioBridgeError.dataConversionFailed
        }
        
        // Set frame length
        buffer.frameLength = AVAudioFrameCount(audioSampleCount)
        
        // Copy data to buffer
        if let channelData = buffer.floatChannelData {
            for i in 0..<audioSampleCount {
                channelData[0][i] = data[i]
            }
        }
        
        return buffer
    }
}

