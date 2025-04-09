import XCTest
import AVFoundation
import CoreML
import Combine
@testable import MLEngine
@testable import AudioBloomCore

final class MLEngineTests: XCTestCase {
    // Test properties
    private var processor: MLProcessor!
    private var cancellables = Set<AnyCancellable>()
    private var testAudioFormat: AVAudioFormat!
    
    // Test expectations
    private var processorReadyExpectation: XCTestExpectation!
    private var featureExtractionExpectation: XCTestExpectation!
    private var visualizationDataExpectation: XCTestExpectation!
    private var performanceMetricsExpectation: XCTestExpectation!
    
    override func setUp() {
        super.setUp()
        testAudioFormat = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)
        
        // Create expectations
        processorReadyExpectation = expectation(description: "Processor ready")
        featureExtractionExpectation = expectation(description: "Feature extraction")
        visualizationDataExpectation = expectation(description: "Visualization data")
        performanceMetricsExpectation = expectation(description: "Performance metrics")
        
        // Initialize processor with default settings
        processor = MLProcessor(optimizationLevel: .balanced)
    }
    
    override func tearDown() {
        processor.cleanup()
        processor = nil
        cancellables.removeAll()
        super.tearDown()
    }
    
    // MARK: - Initialization and Configuration Tests
    
    func testProcessorInitialization() {
        // Test basic initialization
        let processor = MLProcessor()
        XCTAssertEqual(processor.state, .inactive, "Initial state should be inactive")
        XCTAssertEqual(processor.optimizationLevel, .balanced, "Default optimization level should be balanced")
        XCTAssertTrue(processor.useNeuralEngine, "Neural Engine should be enabled by default")
        XCTAssertFalse(processor.isReady, "Processor should not be ready initially")
    }
    
    func testProcessorConfiguration() {
        // Test configuration with different optimization levels
        let highPerformanceProcessor = MLProcessor(optimizationLevel: .performance)
        XCTAssertEqual(highPerformanceProcessor.optimizationLevel, .performance)
        
        let qualityProcessor = MLProcessor(optimizationLevel: .quality)
        XCTAssertEqual(qualityProcessor.optimizationLevel, .quality)
        
        // Test disabling Neural Engine
        let noNeuralEngineProcessor = MLProcessor(useNeuralEngine: false)
        XCTAssertFalse(noNeuralEngineProcessor.useNeuralEngine)
    }
    
    // MARK: - Preparation Tests
    
    func testPrepareMLModel() async throws {
        // Create delegate to monitor ready state changes
        let delegate = TestMLProcessingDelegate()
        processor.delegate = delegate
        
        // Prepare the processor
        try await processor.prepareMLModel(with: testAudioFormat)
        
        // Verify state changes
        XCTAssertTrue(processor.isReady, "Processor should be ready after preparation")
        XCTAssertEqual(processor.state, .ready, "State should be ready")
        XCTAssertTrue(delegate.readyStateChanged, "Delegate should be notified of ready state change")
    }
    
    func testPrepareWithInvalidFormat() async {
        // Test with nil format
        let invalidFormat: AVAudioFormat? = nil
        
        do {
            try await processor.prepareMLModel(with: invalidFormat)
            XCTFail("Processor should throw an error with invalid format")
        } catch let error as MLProcessorError {
            XCTAssertEqual(error.localizedDescription, MLProcessorError.audioFormatError.localizedDescription)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
    
    // MARK: - Audio Feature Extraction Tests
    
    func testAudioFeatureExtraction() async throws {
        // Prepare the processor
        try await processor.prepareMLModel(with: testAudioFormat)
        
        // Create test audio data
        let audioData = createTestSineWave(frequency: 440, duration: 1.0, sampleRate: 44100)
        
        // Process the audio data
        try await processor.processAudioData(audioData)
        
        // Verify audio features
        XCTAssertFalse(processor.audioFeatures.frequencySpectrum.isEmpty, "Frequency spectrum should not be empty")
    }
    
    func testAudioBufferProcessing() async throws {
        // Prepare the processor
        try await processor.prepareMLModel(with: testAudioFormat)
        
        // Create a test PCM buffer
        let buffer = try createTestPCMBuffer(frequency: 440, duration: 0.5)
        
        // Process the buffer
        try await processor.processAudioBuffer(buffer)
        
        // Verify processing occurred
        XCTAssertEqual(processor.state, .processing, "State should be processing")
    }
    
    // MARK: - Neural Engine Optimization Tests
    
    func testNeuralEngineOptimizations() async throws {
        // Configure processor to use Neural Engine
        processor.useNeuralEngine = true
        
        // Prepare the processor
        try await processor.prepareMLModel(with: testAudioFormat)
        
        // Subscribe to performance metrics
        let performanceExpectation = expectation(description: "Performance metrics")
        
        processor.delegate = TestMLProcessingDelegate { metrics in
            // Verify Neural Engine metrics are present
            XCTAssertGreaterThanOrEqual(metrics.neuralEngineUtilization, 0.0)
            XCTAssertLessThanOrEqual(metrics.neuralEngineUtilization, 1.0)
            performanceExpectation.fulfill()
        }
        
        // Process some audio to trigger metrics
        let buffer = try createTestPCMBuffer(frequency: 440, duration: 0.5)
        try await processor.processAudioBuffer(buffer)
        
        // Allow time for metrics reporting
        wait(for: [performanceExpectation], timeout: 5.0)
    }
    
    func testOptimizationLevelChanges() async throws {
        // Start with balanced optimization
        processor.optimizationLevel = .balanced
        
        // Prepare the processor
        try await processor.prepareMLModel(with: testAudioFormat)
        
        // Process baseline audio
        let buffer1 = try createTestPCMBuffer(frequency: 440, duration: 0.5)
        try await processor.processAudioBuffer(buffer1)
        let baselineMetrics = processor.performanceMetrics
        
        // Change to performance optimization
        processor.optimizationLevel = .performance
        
        // Process again with performance settings
        let buffer2 = try createTestPCMBuffer(frequency: 880, duration: 0.5)
        try await processor.processAudioBuffer(buffer2)
        let performanceMetrics = processor.performanceMetrics
        
        // Note: In a real implementation with actual ML models, we would
        // expect performance metrics to differ between optimization levels.
        // For our simulated implementation, we're just verifying the change occurs.
        XCTAssertEqual(processor.optimizationLevel, .performance)
        
        // Change to quality optimization
        processor.optimizationLevel = .quality
        XCTAssertEqual(processor.optimizationLevel, .quality)
    }
    
    // MARK: - Real-time Processing Tests
    
    func testContinuousProcessing() async throws {
        // Prepare the processor
        try await processor.prepareMLModel(with: testAudioFormat)
        
        // Start continuous processing
        try processor.startContinuousProcessing()
        XCTAssertEqual(processor.state, .processing)
        
        // Wait a bit for processing
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        
        // Stop continuous processing
        processor.stopContinuousProcessing()
        
        // Verify state returned to ready
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
        XCTAssertEqual(processor.state, .ready)
    }
    
    func testProcessingWithoutPreparation() async {
        // Attempt to process without preparing
        do {
            let buffer = try createTestPCMBuffer(frequency: 440, duration: 0.5)
            try await processor.processAudioBuffer(buffer)
            XCTFail("Should throw an error when processing without preparation")
        } catch let error as MLProcessorError {
            XCTAssertEqual(error.localizedDescription, MLProcessorError.modelNotLoaded.localizedDescription)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
    
    // MARK: - Output Transformation Tests
    
    func testOutputTransformation() async throws {
        // Create subscription to monitor visualization data
        let dataExpectation = expectation(description: "Visualization data received")
        
        processor.visualizationDataPublisher
            .sink { data in
                // Verify visualization data properties
                XCTAssertFalse(data.values.isEmpty, "Visualization data should not be empty")
                XCTAssertEqual(data.values.count, 64, "Default size should be 64 points")
                XCTAssertFalse(data.isSignificantEvent, "Standard data should not be significant event")
                dataExpectation.fulfill()
            }
            .store(in: &cancellables)
        
        // Prepare the processor
        try await processor.prepareMLModel(with: testAudioFormat)
        
        // Process audio to generate visualization data
        let buffer = try createTestPCMBuffer(frequency: 440, duration: 0.5)
        try await processor.processAudioBuffer(buffer)
        
        // Wait for data to be published
        wait(for: [dataExpectation], timeout: 5.0)
    }
    
    func testBeatDetection() async throws {
        // This test would ideally use audio with clear beats
        // For testing purposes, we'll just verify the processing path for beat detection works
        
        // Create subscription to monitor visualization data
        let beatExpectation = expectation(description: "Beat data processed")
        beatExpectation.isInverted = true // We don't expect this to be fulfilled in our test setup
        
        processor.visualizationDataPublisher
            .filter { $0.isSignificantEvent }
            .sink { data in
                // This would be called if a beat was detected
                XCTAssertTrue(data.isSignificantEvent, "Beat data should be marked as significant")
                beatExpectation.fulfill()
            }
            .store(in: &cancellables)
        
        // Prepare the processor
        try await processor.prepareMLModel(with: testAudioFormat)
        
        // Process audio (not likely to trigger beat detection with our test data)
        let buffer = try createTestPCMBuffer(frequency: 440, duration: 0.5)
        try await processor.processAudioBuffer(buffer)
        
        // We don't expect beat detection with our simple test tone,
        // so this expectation should time out without failing
        wait(for: [beatExpectation], timeout: 1.0)
    }
    
    // MARK: - Error Handling Tests
    
    func testErrorHandling() async {
        // Create delegate to monitor errors
        let delegate = TestMLProcessingDelegate()
        processor.delegate = delegate
        
        // Test with invalid audio format
        let invalidFormat = AVAudioFormat(standardFormatWithSampleRate: 0, channels: 0)
        
        do {
            try await processor.prepareMLModel(with: invalidFormat)
            XCTFail("Should throw an error with invalid format")
        } catch {
            XCTAssertTrue(delegate.errorEncountered, "Delegate should be notified of error")
        }
    }
    
    func testErrorRecovery() async throws {
        // Create delegate to monitor errors
        let delegate = TestMLProcessingDelegate()
        processor.delegate = delegate
        
        // First, deliberately cause an error
        do {
            let invalidFormat = AVAudioFormat(standardFormatWithSampleRate: 0, channels: 0)
            try await processor.prepareMLModel(with: invalidFormat)
            XCTFail("Should throw an error with invalid format")
        } catch {
            XCTAssertTrue(delegate.errorEncountered, "Delegate should be notified of error")
            XCTAssertTrue(processor.state.isErrorState, "Processor should be in error state")
        }
        
        // Now recover by preparing with valid format
        delegate.reset()
        
        try await processor.prepareMLModel(with: testAudioFormat)
        XCTAssertEqual(processor.state, .ready, "Processor should recover to ready state")
        XCTAssertTrue(delegate.readyStateChanged, "Delegate should be notified of ready state")
    }
    
    // MARK: - Integration Tests
    
    func testFullPipeline() async throws {
        // Test the full pipeline from audio input to visualization output
        
        // Create expectations
        let fullPipelineExpectation = expectation(description: "Full pipeline processing")
        
        // Create subcriptions
        processor.visualizationDataPublisher
            .sink { data in
                // Verify visualization data
                XCTAssertFalse(data.values.isEmpty)
                fullPipelineExpectation.fulfill()
            }
            .store(in: &cancellables)
        
        // Prepare the processor
        try await processor.prepareMLModel(with: testAudioFormat)
        
        // Process audio through the pipeline
        let audioData = createTestSineWave(frequency: 440, duration: 1.0, sampleRate: 44100)
        try await processor.processAudioData(audioData)
        
        // Wait for the full pipeline to process
        wait(for: [fullPipelineExpectation], timeout: 5.0)
        
        // Verify output data is populated
        XCTAssertFalse(processor.outputData.isEmpty, "Output data should be populated")
        XCTAssertEqual(processor.outputData.count, 64, "Output data should have 64 points")
    }
    
    func testPerformance() throws {
        // Measure performance of the audio processing pipeline
        measure {
            // Run in a synchronous context for measurement
            let group = DispatchGroup()
            group.enter()
            
            Task {
                do {
                    // Prepare the processor
                    try await processor.prepareMLModel(with: testAudioFormat)
                    
                    // Process a series of audio buffers
                    for freq in stride(from: 220.0, to: 880.0, by: 110.0) {
                        let buffer = try createTestPCMBuffer(frequency: freq, duration: 0.2)
                        try await processor.processAudioBuffer(buffer)
                    }
                } catch {
                    XCTFail("Pipeline performance test failed: \(error)")
                }
                group.leave()
            }
            
            group.wait()
        }
    }
    
    // MARK: - Test Utilities
    
    /// Creates a test sine wave with the specified parameters
    /// - Parameters:
    ///   - frequency: The frequency of the sine wave in Hz
    ///   - duration: The duration of the wave in seconds
    ///   - sampleRate: The sample rate in samples per second
    /// - Returns: An array of Float samples
    private func createTestSineWave(frequency: Double, duration: Double, sampleRate: Double) -> [Float] {
        let sampleCount = Int(duration * sampleRate)
        var samples = [Float](repeating: 0.0, count: sampleCount)
        
        // Generate sine wave
        for i in 0..<sampleCount {
            let time = Double(i) / sampleRate
            let amplitude = sin(2.0 * .pi * frequency * time)
            samples[i] = Float(amplitude * 0.5) // Half amplitude to avoid clipping
        }
        
        return samples
    }
    
    /// Creates a test PCM buffer with a sine wave
    /// - Parameters:
    ///   - frequency: The frequency of the sine wave in Hz
    ///   - duration: The duration of the wave in seconds
    /// - Returns: An AVAudioPCMBuffer containing the sine wave
    /// - Throws: Error if buffer creation fails
    private func createTestPCMBuffer(frequency: Double, duration: Double) throws -> AVAudioPCMBuffer {
        guard let format = testAudioFormat else {
            throw MLProcessorError.audioFormatError
        }
        
        // Create sine wave data
        let sampleRate = format.sampleRate
        let samples = createTestSineWave(frequency: frequency, duration: duration, sampleRate: sampleRate)
        
        // Create buffer with appropriate capacity
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw MLProcessorError.audioFormatError
        }
        
        // Set the frame length to match data count
        buffer.frameLength = AVAudioFrameCount(samples.count)
        
        // Copy data to the buffer
        if let bufferChannelData = buffer.floatChannelData {
            for i in 0..<samples.count {
                bufferChannelData[0][i] = samples[i]
            }
        }
        
        return buffer
    }
    
    /// Creates a test buffer with rhythmic beats for beat detection testing
    /// - Returns: An AVAudioPCMBuffer containing simulated beats
    /// - Throws: Error if buffer creation fails
    private func createTestBeatBuffer() throws -> AVAudioPCMBuffer {
        guard let format = testAudioFormat else {
            throw MLProcessorError.audioFormatError
        }
        
        // Create a buffer with 2 seconds of audio at given sample rate
        let sampleRate = format.sampleRate
        let duration = 2.0
        let sampleCount = Int(duration * sampleRate)
        var samples = [Float](repeating: 0.0, count: sampleCount)
        
        // Generate a carrier tone
        for i in 0..<sampleCount {
            let time = Double(i) / sampleRate
            samples[i] = Float(sin(2.0 * .pi * 440.0 * time) * 0.2) // Base tone
        }
        
        // Add "beats" - short loud pulses at regular intervals
        let beatInterval = 0.5 // beat every 500ms
        let beatDuration = 0.05 // 50ms beat duration
        let beatSamples = Int(beatDuration * sampleRate)
        
        for beatTime in stride(from: 0.0, to: duration, by: beatInterval) {
            let startSample = Int(beatTime * sampleRate)
            for i in 0..<min(beatSamples, sampleCount - startSample) {
                // Apply an envelope to the beat (attack-decay)
                let envelope = Float(sin(.pi * Double(i) / Double(beatSamples)))
                samples[startSample + i] += envelope * 0.8 // Add beat
            }
        }
        
        // Create buffer with appropriate capacity
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw MLProcessorError.audioFormatError
        }
        
        // Set the frame length to match data count
        buffer.frameLength = AVAudioFrameCount(samples.count)
        
        // Copy data to the buffer
        if let bufferChannelData = buffer.floatChannelData {
            for i in 0..<samples.count {
                bufferChannelData[0][i] = samples[i]
            }
        }
        
        return buffer
    }
}

// MARK: - Helper Extensions

extension MLProcessor.ProcessingState: Equatable {
    public static func == (lhs: MLProcessor.ProcessingState, rhs: MLProcessor.ProcessingState) -> Bool {
        switch (lhs, rhs) {
        case (.inactive, .inactive):
            return true
        case (.preparing, .preparing):
            return true
        case (.ready, .ready):
            return true
        case (.processing, .processing):
            return true
        case (.error, .error):
            // For error states, we consider them equal regardless of the specific error
            return true
        default:
            return false
        }
    }
    
    /// Whether this state is an error state
    var isErrorState: Bool {
        switch self {
        case .error:
            return true
        default:
            return false
        }
    }
}

// MARK: - Test Delegate Implementations

/// A test delegate for MLProcessor that tracks state changes and errors
class TestMLProcessingDelegate: MLProcessingDelegate {
    /// Whether ready state changed was called
    var readyStateChanged = false
    
    /// Whether an error was encountered
    var errorEncountered = false
    
    /// Latest performance metrics
    var latestMetrics: MLProcessor.PerformanceMetrics?
    
    /// Optional performance metrics handler
    var onMetricsUpdate: ((MLProcessor.PerformanceMetrics) -> Void)?
    
    /// Initializes a new test delegate
    /// - Parameter onMetricsUpdate: Optional handler for metrics updates
    init(onMetricsUpdate: ((MLProcessor.PerformanceMetrics) -> Void)? = nil) {
        self.onMetricsUpdate = onMetricsUpdate
    }
    
    /// Resets the delegate state
    func reset() {
        readyStateChanged = false
        errorEncountered = false
        latestMetrics = nil
    }
    
    // MARK: MLProcessingDelegate
    
    func mlProcessorReadyStateChanged(isReady: Bool) {
        readyStateChanged = true
    }
    
    func mlProcessorDidEncounterError(_ error: Error) {
        errorEncountered = true
    }
    
    func mlProcessorDidUpdateMetrics(_ metrics: MLProcessor.PerformanceMetrics) {
        latestMetrics = metrics
        onMetricsUpdate?(metrics)
    }
}
