import XCTest
import AVFoundation
import Accelerate
import Metal
@testable import AudioProcessor
@testable import AudioBloomCore

@available(macOS 15.0, *)
final class FFTProcessingNodeTests: XCTestCase {
    
    // MARK: - Test Variables
    
    /// Audio pipeline for context in tests
    private var pipelineCore: AudioPipelineCore!
    
    /// FFT processing node under test
    private var fftNode: FFTProcessingNode!
    
    /// Input buffer ID for testing
    private var inputBufferId: AudioBufferID!
    
    /// Output buffer ID for testing
    private var outputBufferId: AudioBufferID!
    
    /// Sample rate for test audio
    private let sampleRate: Double = 48000
    
    /// Default FFT size
    private let defaultFFTSize = 2048
    
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
            pipelineCore = try AudioPipelineCore(configuration: AudioPipelineConfiguration(enableMetalCompute: false))
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
        inputBufferId = try pipelineCore.allocateBuffer(size: defaultFFTSize * MemoryLayout<Float>.stride, type: .cpu)
        outputBufferId = try pipelineCore.allocateBuffer(size: defaultFFTSize / 2 * MemoryLayout<Float>.stride, type: .cpu)
        
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
    
    // MARK: - Helper Methods
    
    /// Generate sine wave test signal
    /// - Parameters:
    ///   - frequency: Frequency of the sine wave in Hz
    ///   - amplitude: Amplitude of the sine wave (0.0-1.0)
    ///   - sampleCount: Number of samples to generate
    ///   - sampleRate: Sample rate in Hz
    /// - Returns: Array of float samples
    private func generateSineWave(frequency: Double, amplitude: Float, sampleCount: Int, sampleRate: Double) -> [Float] {
        var samples = [Float](repeating: 0, count: sampleCount)
        
        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Double.pi * frequency * Double(sampleIndex) / sampleRate
            samples[sampleIndex] = amplitude * Float(sin(phase))
        }
        
        return samples
    }
    
    /// Fill the input buffer with test data
    /// - Parameter samples: Audio samples to use
    private func fillInputBuffer(samples: [Float]) throws {
        guard let inputBufferId = inputBufferId else {
            XCTFail("Input buffer not allocated")
            return
        }
        
        try samples.withUnsafeBytes { rawBufferPointer in
            try pipelineCore.updateBuffer(
                id: inputBufferId,
                data: rawBufferPointer.baseAddress!,
                size: samples.count * MemoryLayout<Float>.stride,
                options: [.waitForCompletion]
            )
        }
    }
    
    /// Get output buffer contents
    /// - Returns: Array of float values from the output buffer
    private func getOutputBufferContents() throws -> [Float] {
        guard let outputBufferId = outputBufferId else {
            XCTFail("Output buffer not allocated")
            return []
        }
        
        let buffer = try pipelineCore.getBuffer(id: outputBufferId)
        guard let cpuBuffer = buffer.cpuBuffer else {
            XCTFail("Output buffer has no CPU-accessible data")
            return []
        }
        
        let floatPtr = cpuBuffer.assumingMemoryBound(to: Float.self)
        let count = buffer.size / MemoryLayout<Float>.stride
        
        return Array(UnsafeBufferPointer(start: floatPtr, count: count))
    }
    
    /// Calculate the frequency corresponding to a specific FFT bin
    /// - Parameters:
    ///   - binIndex: The bin index (0 to fftSize/2-1)
    ///   - fftSize: Size of the FFT
    ///   - sampleRate: Sample rate in Hz
    /// - Returns: Frequency in Hz
    private func binToFrequency(binIndex: Int, fftSize: Int, sampleRate: Double) -> Double {
        return Double(binIndex) * sampleRate / Double(fftSize)
    }
    
    /// Find the bin index with the maximum magnitude
    /// - Parameter spectrum: FFT magnitude spectrum
    /// - Returns: Index of the bin with maximum magnitude
    private func findPeakBin(spectrum: [Float]) -> Int {
        guard !spectrum.isEmpty else { return 0 }
        
        var maxIndex = 0
        var maxValue: Float = -Float.greatestFiniteMagnitude
        
        for (index, value) in spectrum.enumerated() {
            if value > maxValue {
                maxValue = value
                maxIndex = index
            }
        }
        
        return maxIndex
    }
    
    // MARK: - Initialization Tests
    
    func testInitialization() {
        // Test basic initialization
        let fftNode = FFTProcessingNode()
        
        XCTAssertNotNil(fftNode)
        XCTAssertTrue(fftNode.isEnabled)
        XCTAssertEqual(fftNode.name, "FFT Processor")
        
        // Verify default values
        let levels = fftNode.getFrequencyBandLevels()
        XCTAssertEqual(levels.bass, 0.0)
        XCTAssertEqual(levels.mid, 0.0)
        XCTAssertEqual(levels.treble, 0.0)
    }
    
    func testInitializationWithCustomParameters() {
        // Test initialization with custom settings
        let customFFTSize = 4096
        let customNode = FFTProcessingNode(
            id: UUID(),
            name: "Custom FFT Node",
            fftSize: customFFTSize,
            windowFunction: .blackman
        )
        
        XCTAssertNotNil(customNode)
        XCTAssertEqual(customNode.name, "Custom FFT Node")
        
        // Verify input/output requirements reflect the custom FFT size
        XCTAssertEqual(customNode.outputCapabilities.bufferSize.min, customFFTSize / 2)
        XCTAssertEqual(customNode.outputCapabilities.bufferSize.max, customFFTSize / 2)
    }
    
    func testFFTSizeNormalization() {
        // Test that non-power-of-2 FFT sizes are normalized
        
        // Test rounding up
        let node1 = FFTProcessingNode(fftSize: 1000) // Should round to 1024
        XCTAssertEqual(node1.outputCapabilities.bufferSize.min, 512)
        
        // Test rounding down
        let node2 = FFTProcessingNode(fftSize: 3000) // Should round to 2048 or 4096
        let expectedSize = node2.outputCapabilities.bufferSize.min
        XCTAssertTrue(expectedSize == 1024 || expectedSize == 2048, "FFT size should be normalized to a power of 2")
        
        // Test minimum size
        let node3 = FFTProcessingNode(fftSize: 32) // Should use minimum size (64)
        XCTAssertGreaterThanOrEqual(node3.outputCapabilities.bufferSize.min, 32)
    }
    
    // MARK: - Configuration Tests
    
    func testConfigurationParameters() throws {
        // Test configuration parameter handling
        try fftNode.configure(parameters: [
            "fftSize": 1024,
            "windowFunction": "hamming",
            "bassMin": 30.0,
            "bassMax": 300.0,
            "midMin": 300.0,
            "midMax": 5000.0,
            "trebleMin": 5000.0,
            "trebleMax": 18000.0
        ])
        
        // Verify output size matches the configured FFT size
        XCTAssertEqual(fftNode.outputCapabilities.bufferSize.min, 512)
        XCTAssertEqual(fftNode.outputCapabilities.bufferSize.max, 512)
    }
    
    func testInvalidConfiguration() throws {
        // FFT node should handle invalid configuration parameters gracefully
        
        // Test with excessively large FFT size
        try fftNode.configure(parameters: ["fftSize": 1_000_000])
        
        // Even with invalid input, the node should configure to a valid size
        XCTAssertLessThanOrEqual(fftNode.outputCapabilities.bufferSize.min, 16384)
        
        // Test with invalid window function
        try fftNode.configure(parameters: ["windowFunction": "invalid_window"])
        
        // Should keep the previously valid window function
        XCTAssertNoThrow(try fftNode.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        ))
    }
    
    // MARK: - Window Function Tests
    
    func testWindowFunctions() async throws {
        // Create a simple sine wave for testing
        let sineWave = generateSineWave(
            frequency: 1000.0,
            amplitude: 0.5,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )
        
        // Test each window function
        for windowFunction in [FFTProcessingNode.WindowFunction.hann,
                              .hamming,
                              .blackman,
                              .none] {
            
            // Configure node to use this window
            try fftNode.configure(parameters: ["windowFunction": windowFunction.rawValue])
            
            // Fill input with sine wave
            try fillInputBuffer(samples: sineWave)
            
            // Process the data
            let success = try await fftNode.process(
                inputBuffers: [inputBufferId],
                outputBuffers: [outputBufferId],
                context: pipelineCore
            )
            
            XCTAssertTrue(success, "Processing with \(windowFunction) window should succeed")
            
            // Get output and verify we have data
            let outputData = try getOutputBufferContents()
            XCTAssertFalse(outputData.isEmpty, "Output data should not be empty")
            
            // Verify peak detection (should be around 1000 Hz)
            let peakBin = findPeakBin(spectrum: outputData)
            let peakFrequency = binToFrequency(binIndex: peakBin, fftSize: defaultFFTSize, sampleRate: sampleRate)
            
            // Allow some tolerance in frequency detection due to different window functions
            XCTAssertTrue(abs(peakFrequency - 1000.0) < 100.0,
                         "Peak frequency with \(windowFunction) window should be close to 1000 Hz, was \(peakFrequency) Hz")
            
            // Verify the window function spectral characteristics
            // (This is more qualitative, but we can check for expected patterns)
            switch windowFunction {
            case .none:
                // Rectangular window should have narrower main lobe but higher side lobes
                let sideLobeRatio = calculateSideLobeRatio(spectrum: outputData, mainPeakBin: peakBin)
                XCTAssertGreaterThan(sideLobeRatio, 0.05, "Rectangular window should have significant side lobes")
                
            case .hann, .hamming, .blackman:
                // These windows should have reduced side lobes
                let sideLobeRatio = calculateSideLobeRatio(spectrum: outputData, mainPeakBin: peakBin)
                XCTAssertLessThan(sideLobeRatio, 0.2, "\(windowFunction) window should have reduced side lobes")
            }
        }
    }
    
    /// Calculate the ratio of side lobe energy to main lobe energy
    /// - Parameters:
    ///   - spectrum: FFT magnitude spectrum
    ///   - mainPeakBin: Bin index of the main peak
    /// - Returns: Ratio of side lobe energy to main lobe energy
    private func calculateSideLobeRatio(spectrum: [Float], mainPeakBin: Int) -> Float {
        guard !spectrum.isEmpty && mainPeakBin < spectrum.count else { return 0 }
        
        let mainPeakValue = spectrum[mainPeakBin]
        
        // Define main lobe width (adjust as needed based on window type)
        let mainLobeWidth = 5
        
        // Calculate bounds of main lobe
        let lowerBound = max(0, mainPeakBin - mainLobeWidth)
        let upperBound = min(spectrum.count - 1, mainPeakBin + mainLobeWidth)
        
        // Calculate main lobe energy
        var mainLobeEnergy: Float = 0
        for binIndex in lowerBound...upperBound {
            mainLobeEnergy += spectrum[binIndex] * spectrum[binIndex]
        }
        
        // Calculate side lobe energy (all bins outside the main lobe)
        var sideLobeEnergy: Float = 0
        for binIndex in 0..<spectrum.count {
            if binIndex < lowerBound || binIndex > upperBound {
                sideLobeEnergy += spectrum[binIndex] * spectrum[binIndex]
            }
        }
        
        // Calculate ratio (avoid division by zero)
        return mainLobeEnergy > 0 ? sideLobeEnergy / mainLobeEnergy : 0
    }
    
    // MARK: - Audio Processing Tests
    
    func testProcessingSingleFrequencies() async throws {
        // Test FFT accuracy with various single-frequency sine waves
        // Frequencies to test
        let testFrequencies = [100.0, 440.0, 1000.0, 4000.0, 10000.0]
        
        for frequency in testFrequencies {
            // Generate sine wave
            let sineWave = generateSineWave(
                frequency: frequency,
                amplitude: 0.5,
                sampleCount: defaultFFTSize,
                sampleRate: sampleRate
            )
            
            // Fill input with sine wave
            try fillInputBuffer(samples: sineWave)
            
            // Process the data
            let success = try await fftNode.process(
                inputBuffers: [inputBufferId],
                outputBuffers: [outputBufferId],
                context: pipelineCore
            )
            
            XCTAssertTrue(success, "Processing \(frequency) Hz sine wave should succeed")
            
            // Get output and verify frequency detection
            let outputData = try getOutputBufferContents()
            XCTAssertFalse(outputData.isEmpty, "Output data should not be empty")
            
            // Find peak and calculate its frequency
            let peakBin = findPeakBin(spectrum: outputData)
            let peakFrequency = binToFrequency(binIndex: peakBin, fftSize: defaultFFTSize, sampleRate: sampleRate)
            
            // Allow for some frequency resolution error based on FFT size
            let frequencyResolution = sampleRate / Double(defaultFFTSize)
            let allowedError = frequencyResolution * 1.5 // 1.5x bin width for margin
            
            XCTAssertTrue(abs(peakFrequency - frequency) < allowedError,
                         "Peak frequency should be \(frequency) Hz within \(allowedError) Hz error margin, but was \(peakFrequency) Hz")
        }
    }
    
    func testMagnitudeAccuracy() async throws {
        // Test that magnitudes in the spectrum are proportional to input amplitude
        
        // Generate test signals with different amplitudes
        let testFrequency = 1000.0 // Hz
        let testAmplitudes: [Float] = [0.1, 0.25, 0.5, 0.75, 1.0]
        
        var peakMagnitudes: [Float] = []
        
        for amplitude in testAmplitudes {
            // Generate sine wave
            let sineWave = generateSineWave(
                frequency: testFrequency,
                amplitude: amplitude,
                sampleCount: defaultFFTSize,
                sampleRate: sampleRate
            )
            
            // Fill input with sine wave
            try fillInputBuffer(samples: sineWave)
            
            // Process the data
            _ = try await fftNode.process(
                inputBuffers: [inputBufferId],
                outputBuffers: [outputBufferId],
                context: pipelineCore
            )
            
            // Get output spectrum
            let outputData = try getOutputBufferContents()
            
            // Find peak and record its magnitude
            let peakBin = findPeakBin(spectrum: outputData)
            peakMagnitudes.append(outputData[peakBin])
        }
        
        // Verify that magnitudes increase with amplitude
        for i in 1..<peakMagnitudes.count {
            XCTAssertGreaterThan(peakMagnitudes[i], peakMagnitudes[i-1],
                              "Magnitude should increase with amplitude")
        }
        
        // Verify approximate linear relationship (compensating for logarithmic scaling in FFT processing)
        // For example, check that doubling the amplitude increases the output significantly
        let ratio1 = peakMagnitudes[2] / peakMagnitudes[0] // 0.5 amplitude vs 0.1 amplitude
        XCTAssertGreaterThan(ratio1, 1.2, "Increasing amplitude from 0.1 to 0.5 should significantly increase output magnitude")
    }
    
    func testProcessingSilenceAndNoise() async throws {
        // Test handling of silence
        let silence = [Float](repeating: 0.0, count: defaultFFTSize)
        
        // Fill input with silence
        try fillInputBuffer(samples: silence)
        
        // Process the data
        let successSilence = try await fftNode.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )
        
        XCTAssertTrue(successSilence, "Processing silence should succeed")
        
        // Get output and verify low magnitudes across the spectrum
        let silenceOutput = try getOutputBufferContents()
        let silenceMaxValue = silenceOutput.max() ?? 0.0
        XCTAssertLessThan(silenceMaxValue, 0.1, "Silence should produce very low FFT magnitudes")
        
        // Test handling of white noise
        var noise = [Float](repeating: 0.0, count: defaultFFTSize)
        for i in 0..<defaultFFTSize {
            noise[i] = Float.random(in: -0.5...0.5)
        }
        
        // Fill input with noise
        try fillInputBuffer(samples: noise)
        
        // Process the data
        let successNoise = try await fftNode.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )
        
        XCTAssertTrue(successNoise, "Processing noise should succeed")
        
        // Get output and verify distribution across the spectrum
        let noiseOutput = try getOutputBufferContents()
        
        // Calculate standard deviation of noise spectrum (should be relatively consistent)
        let noiseMean = noiseOutput.reduce(0, +) / Float(noiseOutput.count)
        let noiseVariance = noiseOutput.reduce(0) { $0 + pow($1 - noiseMean, 2) } / Float(noiseOutput.count)
        let noiseStdDev = sqrt(noiseVariance)
        
        // White noise should have energy spread across the spectrum
        XCTAssertGreaterThan(noiseStdDev, 0.01, "Noise should have variation across the spectrum")
    }
    
    func testMultiFrequencySignal() async throws {
        // Test FFT response to a complex signal with multiple frequency components
        
        // Generate a complex test signal with multiple sine waves
        var complexSignal = [Float](repeating: 0.0, count: defaultFFTSize)
        
        // Add several frequency components
        let components: [(frequency: Double, amplitude: Float)] = [
            (100.0, 0.3),   // Low frequency (bass)
            (500.0, 0.5),   // Mid frequency
            (5000.0, 0.4)   // High frequency (treble)
        ]
        
        for sampleIndex in 0..<defaultFFTSize {
            for component in components {
                let phase = 2.0 * Double.pi * component.frequency * Double(sampleIndex) / sampleRate
                complexSignal[sampleIndex] += component.amplitude * sin(Float(phase))
            }
            // Normalize to prevent clipping
            complexSignal[sampleIndex] = min(max(complexSignal[sampleIndex], -1.0), 1.0)
        }
        
        // Fill input with complex signal
        try fillInputBuffer(samples: complexSignal)
        
        // Process the data
        let success = try await fftNode.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )
        
        XCTAssertTrue(success, "Processing multi-frequency signal should succeed")
        
        // Get output and verify peaks at expected frequencies
        let outputData = try getOutputBufferContents()
        
        // Find approximate bin indices for each frequency component
        for component in components {
            let freqBin = Int(component.frequency * Double(defaultFFTSize) / sampleRate)
            let binRangeStart = max(0, freqBin - 2)
            let binRangeEnd = min(outputData.count - 1, freqBin + 2)
            
            // Get the maximum value in the range around the expected bin
            let rangeMax = outputData[binRangeStart...binRangeEnd].max() ?? 0.0
            
            // Verify the component is detected (has significant magnitude)
            XCTAssertGreaterThan(rangeMax, 0.1, 
                              "Component at \(component.frequency) Hz should be detected in the spectrum")
        }
    }
    
    // MARK: - Frequency Band Analysis Tests
    
    func testFrequencyBandDetection() async throws {
        // Test detection of bass, mid, and treble frequency bands
        
        // Configure frequency bands for testing
        try fftNode.configure(parameters: [
            "bassMin": 20.0,
            "bassMax": 250.0,
            "midMin": 250.0,
            "midMax": 4000.0,
            "trebleMin": 4000.0,
            "trebleMax": 20000.0
        ])
        
        // Test frequencies in each band
        let bassFreq = 100.0    // Bass frequency
        let midFreq = 1000.0    // Mid frequency
        let trebleFreq = 8000.0 // Treble frequency
        
        // Test bass frequency
        try processSingleFrequencyAndCheckBands(
            frequency: bassFreq,
            expectedDominantBand: "bass"
        )
        
        // Test mid frequency
        try await processSingleFrequencyAndCheckBands(
            frequency: midFreq,
            expectedDominantBand: "mid"
        )
        
        // Test treble frequency
        try await processSingleFrequencyAndCheckBands(
            frequency: trebleFreq,
            expectedDominantBand: "treble"
        )
    }
    
    /// Process a single frequency and check that the correct frequency band has the highest level
    /// - Parameters:
    ///   - frequency: Frequency to process
    ///   - expectedDominantBand: Expected dominant frequency band
    private func processSingleFrequencyAndCheckBands(
        frequency: Double,
        expectedDominantBand: String
    ) async throws {
        // Generate sine wave
        let sineWave = generateSineWave(
            frequency: frequency,
            amplitude: 0.5,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )
        
        // Fill input with sine wave
        try fillInputBuffer(samples: sineWave)
        
        // Process the data
        _ = try await fftNode.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )
        
        // Get frequency band levels
        let levels = fftNode.getFrequencyBandLevels()
        
        // Determine which band has the highest level
        let bassLevel = levels.bass
        let midLevel = levels.mid
        let trebleLevel = levels.treble
        
        // Verify the expected band has the highest level
        switch expectedDominantBand {
        case "bass":
            XCTAssertGreaterThan(bassLevel, midLevel, "Bass level should be greater than mid level for \(frequency) Hz")
            XCTAssertGreaterThan(bassLevel, trebleLevel, "Bass level should be greater than treble level for \(frequency) Hz")
        case "mid":
            XCTAssertGreaterThan(midLevel, bassLevel, "Mid level should be greater than bass level for \(frequency) Hz")
            XCTAssertGreaterThan(midLevel, trebleLevel, "Mid level should be greater than treble level for \(frequency) Hz")
        case "treble":
            XCTAssertGreaterThan(trebleLevel, bassLevel, "Treble level should be greater than bass level for \(frequency) Hz")
            XCTAssertGreaterThan(trebleLevel, midLevel, "Treble level should be greater than mid level for \(frequency) Hz")
        default:
            XCTFail("Unknown expected band: \(expectedDominantBand)")
        }
    }
    
    func testBandSmoothingBehavior() async throws {
        // Test that frequency band levels have appropriate smoothing behavior
        
        // Configure node
        try fftNode.configure(parameters: [:])
        
        // First process silence to reset band levels
        let silence = [Float](repeating: 0.0, count: defaultFFTSize)
        try fillInputBuffer(samples: silence)
        _ = try await fftNode.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        // Verify initial band levels
        var initialLevels = fftNode.getFrequencyBandLevels()
        XCTAssertEqual(initialLevels.bass, 0.0, "Initial bass level should be zero after silence")
        XCTAssertEqual(initialLevels.mid, 0.0, "Initial mid level should be zero after silence")
        XCTAssertEqual(initialLevels.treble, 0.0, "Initial treble level should be zero after silence")
        
        // Now generate a strong bass signal
        let bassSignal = generateSineWave(
            frequency: 100.0,
            amplitude: 0.9,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )
        
        // Process bass signal
        try fillInputBuffer(samples: bassSignal)
        _ = try await fftNode.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )
        
        // Get band levels after first bass processing
        let firstBassLevels = fftNode.getFrequencyBandLevels()
        XCTAssertGreaterThan(firstBassLevels.bass, 0.5, "Bass level should be significant after bass signal")
        
        // Now immediately switch to silence
        try fillInputBuffer(samples: silence)
        _ = try await fftNode.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )
        
        // Check levels after switching to silence - should still have some bass due to smoothing
        let afterSilenceLevels = fftNode.getFrequencyBandLevels()
        XCTAssertGreaterThan(afterSilenceLevels.bass, 0.0, "Bass level should not immediately drop to zero due to smoothing")
        XCTAssertLessThan(afterSilenceLevels.bass, firstBassLevels.bass, "Bass level should decrease after switching to silence")
        
        // Process silence a few more times to verify levels eventually approach zero
        for _ in 0..<5 {
            _ = try await fftNode.process(
                inputBuffers: [inputBufferId],
                outputBuffers: [outputBufferId],
                context: pipelineCore
            )
        }
        
        // Verify levels after multiple silence frames
        let finalLevels = fftNode.getFrequencyBandLevels()
        XCTAssertLessThan(finalLevels.bass, 0.1, "Bass level should approach zero after multiple silence frames")
    }
    
    // MARK: - Performance Tests
    
    func testProcessingPerformance() async throws {
        // Test FFT processing performance
        
        // Create a complex test signal
        var complexSignal = [Float](repeating: 0.0, count: defaultFFTSize)
        for sampleIndex in 0..<defaultFFTSize {
            // Mix several frequencies
            let phase1 = 2.0 * Double.pi * 100.0 * Double(sampleIndex) / sampleRate
            let phase2 = 2.0 * Double.pi * 1000.0 * Double(sampleIndex) / sampleRate
            let phase3 = 2.0 * Double.pi * 5000.0 * Double(sampleIndex) / sampleRate
            
            complexSignal[sampleIndex] = 0.3 * sin(Float(phase1)) + 
                                        0.3 * sin(Float(phase2)) + 
                                        0.3 * sin(Float(phase3))
        }
        
        // Fill input buffer
        try fillInputBuffer(samples: complexSignal)
        
        // Measure processing time for 100 consecutive FFTs
        let iterations = 100
        let startTime = CACurrentMediaTime()
        
        for _ in 0..<iterations {
            _ = try await fftNode.process(
                inputBuffers: [inputBufferId],
                outputBuffers: [outputBufferId],
                context: pipelineCore
            )
        }
        
        let endTime = CACurrentMediaTime()
        let totalTime = endTime - startTime
        let averageTime = totalTime / Double(iterations)
        
        // Print performance metrics
        print("FFT Processing Performance:")
        print("  FFT Size: \(defaultFFTSize)")
        print("  Total time for \(iterations) iterations: \(totalTime) seconds")
        print("  Average time per FFT: \(averageTime * 1000) ms")
        
        // Verify performance is suitable for real-time processing
        // Target: process FFT faster than ~20ms for real-time use (assuming 50 fps visualization)
        XCTAssertLessThan(averageTime, 0.020, "FFT processing should be fast enough for real-time use")
    }
    
    func testRealTimeCapability() async throws {
        // Test the node's capability to handle real-time processing rates
        
        // Configure a more demanding FFT size
        try fftNode.configure(parameters: ["fftSize": 4096])
        
        // Reallocate input/output buffers for larger FFT size
        if let inputBufferId = inputBufferId {
            pipelineCore.releaseBuffer(id: inputBufferId)
        }
        if let outputBufferId = outputBufferId {
            pipelineCore.releaseBuffer(id: outputBufferId)
        }
        
        inputBufferId = try pipelineCore.allocateBuffer(size: 4096 * MemoryLayout<Float>.stride, type: .cpu)
        outputBufferId = try pipelineCore.allocateBuffer(size: 2048 * MemoryLayout<Float>.stride, type: .cpu)
        
        // Create a complex test signal
        var complexSignal = [Float](repeating: 0.0, count: 4096)
        for sampleIndex in 0..<4096 {
            // Mix several frequencies with some noise
            let phase1 = 2.0 * Double.pi * 100.0 * Double(sampleIndex) / sampleRate
            let phase2 = 2.0 * Double.pi * 1000.0 * Double(sampleIndex) / sampleRate
            
            complexSignal[sampleIndex] = 0.4 * sin(Float(phase1)) + 
                                        0.4 * sin(Float(phase2)) + 
                                        0.2 * Float.random(in: -1.0...1.0) // Add some noise
        }
        
        // Fill input buffer
        try fillInputBuffer(samples: complexSignal)
        
        // Target processing rate: 60 fps = ~16.7ms per frame
        let targetFrameTime = 1.0 / 60.0
        var successCount = 0
        let testIterations = 30
        
        // Run multiple frames and count how many would meet real-time requirements
        for _ in 0..<testIterations {
            let frameStartTime = CACurrentMediaTime()
            
            // Process the data
            _ = try await fftNode.process(
                inputBuffers: [inputBufferId],
                outputBuffers: [outputBufferId],
                context: pipelineCore
            )
            
            let frameEndTime = CACurrentMediaTime()
            let frameTime = frameEndTime - frameStartTime
            
            if frameTime < targetFrameTime {
                successCount += 1
            }
        }
        
        // Calculate success rate (percentage of frames that would meet real-time requirements)
        let successRate = Double(successCount) / Double(testIterations)
        print("Real-time processing success rate: \(successRate * 100)% at 60 fps target")
        
        // Assert that at least 90% of frames meet real-time requirements
        XCTAssertGreaterThanOrEqual(successRate, 0.9, "FFT node should meet real-time requirements for at least 90% of frames")
    }
    
    func testMemoryUsage() throws {
        // Test memory usage patterns during FFT processing
        
        // Capture initial memory baseline
        let initialMemory = reportMemoryUsage()
        print("Initial memory baseline: \(initialMemory) MB")
        
        // Run a sequence of FFT operations with different configurations
        autoreleasepool {
            // Test multiple FFT sizes
            let fftSizes = [512, 1024, 2048, 4096, 8192]
            
            for fftSize in fftSizes {
                autoreleasepool {
                    do {
                        // Configure node with this FFT size
                        try fftNode.configure(parameters: ["fftSize": fftSize])
                        
                        // Reallocate input/output buffers
                        if let inputBufferId = inputBufferId {
                            pipelineCore.releaseBuffer(id: inputBufferId)
                        }
                        if let outputBufferId = outputBufferId {
                            pipelineCore.releaseBuffer(id: outputBufferId)
                        }
                        
                        inputBufferId = try pipelineCore.allocateBuffer(
                            size: fftSize * MemoryLayout<Float>.stride,
                            type: .cpu
                        )
                        outputBufferId = try pipelineCore.allocateBuffer(
                            size: fftSize / 2 * MemoryLayout<Float>.stride,
                            type: .cpu
                        )
                        
                        // Generate test signal
                        let testSignal = generateSineWave(
                            frequency: 1000.0,
                            amplitude: 0.5,
                            sampleCount: fftSize,
                            sampleRate: sampleRate
                        )
                        
                        // Fill input buffer
                        try fillInputBuffer(samples: testSignal)
                        
                        // Process multiple times
                        for _ in 0..<10 {
                            Task {
                                try await fftNode.process(
                                    inputBuffers: [inputBufferId],
                                    outputBuffers: [outputBufferId],
                                    context: pipelineCore
                                )
                            }
                        }
                    } catch {
                        XCTFail("Memory usage test failed: \(error)")
                    }
                }
            }
        }
        
        // Force memory cleanup
        inputBufferId = nil
        outputBufferId = nil
        
        // Capture final memory usage
        let finalMemory = reportMemoryUsage()
        print("Final memory usage: \(finalMemory) MB")
        
        // Verify memory usage is stable (no significant leaks)
        let memoryDifference = finalMemory - initialMemory
        print("Memory difference: \(memoryDifference) MB")
        
        // Tolerate a small amount of memory growth
        XCTAssertLessThan(memoryDifference, 10.0, "Memory usage should be stable, with no significant leaks")
    }
    
    /// Reports the current memory usage of the process in megabytes
    private func reportMemoryUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size) / (1024 * 1024)
        } else {
            return 0
        }
    }
}
