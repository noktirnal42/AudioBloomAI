import XCTest
import Accelerate
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Base class for FFT processor tests that provides shared setup, teardown, and utility methods
class FFTProcessorBaseTests: XCTestCase {

    // MARK: - Test Variables

    /// Audio pipeline for context in tests
    var pipelineCore: AudioPipelineCore!

    /// FFT processing node under test
    var fftNode: FFTProcessingNode!

    /// Input buffer ID for testing
    var inputBufferId: AudioBufferID!

    /// Output buffer ID for testing
    var outputBufferId: AudioBufferID!

    /// Sample rate for test audio
    let sampleRate: Double = 48000

    /// Default FFT size
    let defaultFFTSize = 2048

    // MARK: - Setup and Teardown

    override func setUp() async throws {
        try await super.setUp()

        // Initialize audio pipeline
        let config = AudioPipelineConfiguration(
            enableMetalCompute: true,
            defaultFormat: AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!,
            bufferSize: defaultFFTSize,
            maxProcessingLoad: 0.8
        )

        do {
            pipelineCore = try AudioPipelineCore(configuration: config)
        } catch {
            // If Metal is not available, try again with Metal disabled
            pipelineCore = try AudioPipelineCore(
                configuration: AudioPipelineConfiguration(enableMetalCompute: false)
            )
        }

        // Initialize FFT node with default settings
        fftNode = FFTProcessingNode(
            name: "Test FFT Node",
            fftSize: defaultFFTSize,
            windowFunction: .hann
        )

        // Configure stream
        try await pipelineCore.configureStream(
            format: AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!,
            bufferSize: defaultFFTSize,
            channels: 1
        )

        // Add node to pipeline
        try pipelineCore.addNode(fftNode, connections: [])

        // Allocate input and output buffers
        inputBufferId = try pipelineCore.allocateBuffer(
            size: defaultFFTSize * MemoryLayout<Float>.stride,
            type: .cpu
        )

        outputBufferId = try pipelineCore.allocateBuffer(
            size: defaultFFTSize / 2 * MemoryLayout<Float>.stride,
            type: .cpu
        )

        // Start the stream
        try await pipelineCore.startStream()
    }

    override func tearDown() async throws {
        // Release buffers
        if let inputBufferId = inputBufferId {
            pipelineCore.releaseBuffer(id: inputBufferId)
        }

        if let outputBufferId = outputBufferId {
            pipelineCore.releaseBuffer(id: outputBufferId)
        }

        // Stop the stream and clean up
        pipelineCore.stopStream()
        pipelineCore = nil
        fftNode = nil

        try await super.tearDown()
    }

    // MARK: - Test Data Generation

    /// Generates a sine wave with the specified parameters
    /// - Parameters:
    ///   - frequency: The frequency of the sine wave in Hz
    ///   - amplitude: The amplitude of the sine wave (0.0-1.0)
    ///   - sampleCount: The number of samples to generate
    ///   - sampleRate: The sample rate in Hz
    /// - Returns: Array of float samples representing the sine wave
    func generateSineWave(
        frequency: Double,
        amplitude: Float = 0.8,
        sampleCount: Int,
        sampleRate: Double
    ) -> [Float] {
        var samples = [Float](repeating: 0.0, count: sampleCount)

        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Double.pi * frequency * Double(sampleIndex) / sampleRate
            samples[sampleIndex] = amplitude * Float(sin(phase))
        }

        return samples
    }

    /// Generates white noise with the specified parameters
    /// - Parameters:
    ///   - amplitude: The amplitude of the noise (0.0-1.0)
    ///   - sampleCount: The number of samples to generate
    /// - Returns: Array of float samples containing random noise
    func generateWhiteNoise(amplitude: Float = 0.5, sampleCount: Int) -> [Float] {
        var samples = [Float](repeating: 0.0, count: sampleCount)

        for sampleIndex in 0..<sampleCount {
            samples[sampleIndex] = amplitude * (Float.random(in: -1.0...1.0))
        }

        return samples
    }

    /// Generates a test signal with multiple frequency components
    /// - Parameters:
    ///   - frequencies: Array of frequencies to include (in Hz)
    ///   - amplitudes: Array of amplitudes for each frequency (0.0-1.0)
    ///   - sampleCount: The number of samples to generate
    ///   - sampleRate: The sample rate in Hz
    /// - Returns: Array of float samples containing the composite signal
    func generateMultiFrequencySignal(
        frequencies: [Double],
        amplitudes: [Float],
        sampleCount: Int,
        sampleRate: Double
    ) -> [Float] {
        // Verify input arrays have the same length
        guard frequencies.count == amplitudes.count else {
            fatalError("Frequencies and amplitudes arrays must have the same count")
        }

        var samples = [Float](repeating: 0.0, count: sampleCount)

        // Generate each frequency component and add to the result
        for componentIndex in 0..<frequencies.count {
            let frequency = frequencies[componentIndex]
            let amplitude = amplitudes[componentIndex]

            for sampleIndex in 0..<sampleCount {
                let phase = 2.0 * Double.pi * frequency * Double(sampleIndex) / sampleRate
                samples[sampleIndex] += amplitude * Float(sin(phase))
            }
        }

        return samples
    }

    // MARK: - Helper Methods

    /// Updates the input buffer with test data and processes it through the FFT node
    /// - Parameters:
    ///   - testData: The audio data to process
    /// - Returns: The FFT magnitude spectrum
    func processTestData(_ testData: [Float]) async throws -> [Float] {
        // Update input buffer with test data
        try pipelineCore.updateBuffer(
            id: inputBufferId,
            data: testData,
            size: testData.count * MemoryLayout<Float>.stride
        )

        // Process the buffer through the FFT node
        try await pipelineCore.processBuffer(
            inputBufferId: inputBufferId,
            outputBufferId: outputBufferId,
            nodeId: fftNode.id,
            size: testData.count * MemoryLayout<Float>.stride
        )

        // Read the output buffer
        var outputData = [Float](repeating: 0.0, count: defaultFFTSize / 2)
        try pipelineCore.readBuffer(
            id: outputBufferId,
            data: &outputData,
            size: outputData.count * MemoryLayout<Float>.stride
        )

        return outputData
    }

    /// Finds the peak in an FFT spectrum
    /// - Parameter spectrum: The FFT magnitude spectrum
    /// - Returns: Tuple with (index, magnitude) of the peak
    func findPeakInSpectrum(_ spectrum: [Float]) -> (index: Int, magnitude: Float) {
        var peakMagnitude: Float = 0.0
        var peakIndex = 0

        for index in 0..<spectrum.count where spectrum[index] > peakMagnitude {
            peakMagnitude = spectrum[index]
            peakIndex = index
        }

        return (peakIndex, peakMagnitude)
    }

    /// Calculates the frequency corresponding to an FFT bin index
    /// - Parameters:
    ///   - binIndex: The FFT bin index
    ///   - fftSize: The size of the FFT
    ///   - sampleRate: The sample rate
    /// - Returns: The frequency in Hz
    func frequencyForBinIndex(_ binIndex: Int, fftSize: Int, sampleRate: Double) -> Double {
        return Double(binIndex) * (sampleRate / Double(fftSize))
    }

    /// Finds the FFT bin index for a given frequency
    /// - Parameters:
    ///   - frequency: The frequency in Hz
    ///   - fftSize: The size of the FFT
    ///   - sampleRate: The sample rate
    /// - Returns: The bin index
    func binIndexForFrequency(_ frequency: Double, fftSize: Int, sampleRate: Double) -> Int {
        return Int(frequency / (sampleRate / Double(fftSize)))
    }

    /// Verifies that a frequency peak is detected within a margin of error
    /// - Parameters:
    ///   - spectrum: The FFT magnitude spectrum
    ///   - frequency: The expected frequency
    ///   - sampleRate: The sample rate
    ///   - fftSize: The FFT size
    ///   - errorMargin: The allowed error margin in bins
    /// - Returns: True if peak was detected within the expected range
    func verifyFrequencyPeak(
        in spectrum: [Float],
        frequency: Double,
        sampleRate: Double,
        fftSize: Int,
        errorMargin: Int = 2
    ) -> Bool {
        let expectedBin = binIndexForFrequency(frequency, fftSize: fftSize, sampleRate: sampleRate)
        let (actualPeakBin, _) = findPeakInSpectrum(spectrum)

        return abs(actualPeakBin - expectedBin) <= errorMargin
    }
}
