// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import Foundation
import AVFoundation
import Accelerate

@available(macOS 15.0, *)
public actor AudioFeatureExtractor {
    private let fftSetup: FFTSetup
    private let sampleRate: Double
    private let bufferSize: Int
    private let windowType: WindowType
    private var lastMagnitudes: [Float]?
    
    public enum WindowType {
        case hann
        case hamming
        case blackman
        case none
    }
    
    public init(sampleRate: Double, bufferSize: Int, windowType: WindowType = .hann) {
        self.sampleRate = sampleRate
        self.bufferSize = bufferSize
        self.windowType = windowType
        
        // Ensure bufferSize is a power of 2 for FFT
        let log2n = vDSP_Length(log2(Float(bufferSize)))
        self.fftSetup = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self)!
    }
    
    deinit {
        // No explicit cleanup needed as of Swift 6
    }
    
    // Main feature extraction function
    public func extractFeatures(buffer: AVAudioPCMBuffer) async -> [AudioFeature] {
        var features: [AudioFeature] = []
        
        // Convert buffer to array of floats
        guard let floatData = convertBufferToFloatArray(buffer) else {
            return []
        }
        
        // Apply window function
        let windowedData = applyWindow(to: floatData)
        
        // Perform FFT and get magnitude spectrum
        let magnitudes = performFFT(on: windowedData)
        
        // Extract features from the magnitude spectrum
        if let dominantFrequency = calculateDominantFrequency(magnitudes: magnitudes) {
            features.append(AudioFeature(type: .dominantFrequency, value: dominantFrequency))
        }
        
        // Calculate spectral centroid
        if let spectralCentroid = calculateSpectralCentroid(magnitudes: magnitudes) {
            features.append(AudioFeature(type: .spectralCentroid, value: spectralCentroid))
        }
        
        // Calculate spectral flux
        if let spectralFlux = calculateSpectralFlux(magnitudes: magnitudes) {
            features.append(AudioFeature(type: .spectralFlux, value: spectralFlux))
        }
        
        // Update last magnitudes for next calculation
        self.lastMagnitudes = magnitudes
        
        return features
    }
    
    // MARK: - Private Helper Methods
    
    private func convertBufferToFloatArray(_ buffer: AVAudioPCMBuffer) -> [Float]? {
        guard let floatChannelData = buffer.floatChannelData?[0] else { return nil }
        let frameCount = Int(buffer.frameLength)
        
        // Ensure we don't exceed the buffer size
        let count = min(frameCount, bufferSize)
        
        var floatArray = [Float](repeating: 0, count: count)
        for i in 0..<count {
            floatArray[i] = floatChannelData[i]
        }
        
        return floatArray
    }
    
    private func applyWindow(to data: [Float]) -> [Float] {
        var windowedData = [Float](repeating: 0, count: data.count)
        
        switch windowType {
        case .hann:
            vDSP.multiply(data, vDSP.window.hanningPeriodicNormalized(count: data.count), result: &windowedData)
        case .hamming:
            vDSP.multiply(data, vDSP.window.hammingNormalized(count: data.count), result: &windowedData)
        case .blackman:
            vDSP.multiply(data, vDSP.window.blackmanNormalized(count: data.count), result: &windowedData)
        case .none:
            windowedData = data
        }
        
        return windowedData
    }
    
    private func performFFT(on data: [Float]) -> [Float] {
        let halfSize = bufferSize / 2
        
        // Prepare real and imaginary parts
        let realp = UnsafeMutablePointer<Float>.allocate(capacity: halfSize)
        let imagp = UnsafeMutablePointer<Float>.allocate(capacity: halfSize)
        defer {
            realp.deallocate()
            imagp.deallocate()
        }
        
        // Create split complex buffer
        var splitComplex = DSPSplitComplex(realp: realp, imagp: imagp)
        
        // Convert input to split complex format
        var paddedData = data
        if paddedData.count < bufferSize {
            paddedData.append(contentsOf: [Float](repeating: 0, count: bufferSize - paddedData.count))
        }
        
        paddedData.withUnsafeBytes { dataPtr in
            vDSP.convert(interleavedComplexVector: dataPtr.bindMemory(to: DSPComplex.self),
                        toSplitComplexVector: &splitComplex,
                        count: vDSP_Length(halfSize))
        }
        
        // Perform forward FFT
        fftSetup.forward(input: splitComplex, output: &splitComplex)
        
        // Calculate magnitude spectrum
        var magnitudes = [Float](repeating: 0, count: halfSize)
        vDSP.absolute(splitComplex, result: &magnitudes)
        
        // Scale magnitudes by 2/bufferSize for proper normalization
        var scaleFactor = Float(2.0 / Float(bufferSize))
        vDSP.multiply(magnitudes, scaleFactor, result: &magnitudes)
        
        return magnitudes
    }
    
    private func calculateDominantFrequency(magnitudes: [Float]) -> Double? {
        guard !magnitudes.isEmpty else { return nil }
        
        // Find the index of the maximum magnitude
        if let maxIndex = magnitudes.indices.max(by: { magnitudes[$0] < magnitudes[$1] }) {
            // Convert index to frequency
            let frequency = Double(maxIndex) * sampleRate / Double(bufferSize)
            return frequency
        }
        
        return nil
    }
    
    private func calculateSpectralCentroid(magnitudes: [Float]) -> Double? {
        guard !magnitudes.isEmpty else { return nil }
        
        // Calculate weighted average of frequencies present in the signal
        var weightedSum: Float = 0.0
        var magnitudeSum: Float = 0.0
        
        for i in 0..<magnitudes.count {
            let frequency = Float(i) * Float(sampleRate) / Float(bufferSize)
            weightedSum += frequency * magnitudes[i]
            magnitudeSum += magnitudes[i]
        }
        
        guard magnitudeSum > 0 else { return nil }
        
        let spectralCentroid = Double(weightedSum / magnitudeSum)
        return spectralCentroid
    }
    
    private func calculateSpectralFlux(magnitudes: [Float]) -> Double? {
        guard let lastMagnitudes = lastMagnitudes, !magnitudes.isEmpty else { 
            // First calculation, no previous magnitudes to compare
            return nil 
        }
        
        // Calculate the difference between current and previous magnitudes
        let minLength = min(magnitudes.count, lastMagnitudes.count)
        var sumOfSquaredDifferences: Float = 0.0
        
        for i in 0..<minLength {
            let diff = magnitudes[i] - lastMagnitudes[i]
            // Only consider positive changes (increases in energy)
            let positiveDiff = max(0, diff)
            sumOfSquaredDifferences += positiveDiff * positiveDiff
        }
        
        // Normalize by the number of frequency bins
        let spectralFlux = Double(sqrt(sumOfSquaredDifferences) / Float(minLength))
        return spectralFlux
    }
}

