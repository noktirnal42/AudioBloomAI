import XCTest
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for basic audio pipeline functionality
final class AudioPipelineBasicTests: XCTestCase {
    // MARK: - Test Properties
    
    /// Audio pipeline under test
    private var audioPipeline: AudioPipelineCore!
    
    // MARK: - Test Lifecycle
    
    override func setUp() async throws {
        try await super.setUp()
        audioPipeline = try AudioPipelineCore(
            configuration: AudioPipelineTestHelpers.createDefaultConfig()
        )
    }
    
    override func tearDown() async throws {
        audioPipeline = nil
        try await super.tearDown()
    }
    
    // MARK: - Basic Processing Tests
    
    /// Tests basic signal flow through the pipeline
    /// - Verifies that:
    ///   - Pipeline processes audio without corruption
    ///   - Output buffer size matches input buffer size
    ///   - Signal maintains integrity through processing
    func testSignalFlow() throws {
        // Create test buffers
        let inputBuffer = try audioPipeline.allocateBuffer(
            size: 1024 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        let outputBuffer = try audioPipeline.allocateBuffer(
            size: 1024 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        
        defer {
            audioPipeline.releaseBuffer(id: inputBuffer)
            audioPipeline.releaseBuffer(id: outputBuffer)
        }
        
        // Create test signal
        let testData = AudioPipelineTestHelpers.generateSineWave(
            frequency: 440.0,
            duration: 0.1,
            amplitude: 0.5
        )
        
        // Fill input buffer
        try testData.withUnsafeBytes { bytes in
            try audioPipeline.updateBuffer(
                id: inputBuffer,
                data: bytes.baseAddress!,
                size: testData.count * MemoryLayout<Float>.stride,
                options: [.waitForCompletion]
            )
        }
        
        // Configure pipeline
        try audioPipeline.reset()
        try audioPipeline.addNode(PassthroughNode(), connections: [])
        try audioPipeline.startStream()
        
        // Process buffer
        let success = try audioPipeline.process(
            inputBuffers: [inputBuffer],
            outputBuffers: [outputBuffer],
            context: audioPipeline
        )
        
        XCTAssertTrue(success, "Processing should succeed")
        
        // Verify output
        let processedData = try AudioPipelineTestHelpers.getBufferContents(
            outputBuffer,
            pipeline: audioPipeline
        )
        
        XCTAssertEqual(
            processedData.count,
            testData.count,
            "Processed buffer should have same length as input"
        )
        
        for (index, sample) in testData.enumerated() {
            XCTAssertEqual(
                processedData[index],
                sample,
                accuracy: 0.001,
                "Sample at index \(index) should match input"
            )
        }
    }
    
    /// Tests gain adjustment processing
    /// - Verifies that:
    ///   - GainNode multiplies all samples by the gain factor
    ///   - Both RMS level and individual samples show correct amplification
    ///   - Input signal is preserved in shape but changed in amplitude
    func testGainAdjustment() throws {
        // Create test buffers
        let inputBuffer = try audioPipeline.allocateBuffer(
            size: 1024 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        let outputBuffer = try audioPipeline.allocateBuffer(
            size: 1024 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        
        defer {
            audioPipeline.releaseBuffer(id: inputBuffer)
            audioPipeline.releaseBuffer(id: outputBuffer)
        }
        
        // Create test signal
        let testData = AudioPipelineTestHelpers.generateSineWave(
            frequency: 440.0,
            duration: 0.1,
            amplitude: 0.5
        )
        
        // Fill input buffer
        try testData.withUnsafeBytes { bytes in
            try audioPipeline.updateBuffer(
                id: inputBuffer,
                data: bytes.baseAddress!,
                size: testData.count * MemoryLayout<Float>.stride,
                options: [.waitForCompletion]
            )
        }
        
        // Configure pipeline
        try audioPipeline.reset()
        let gainNode = GainNode(name: "Gain Node", gain: 2.0)
        try audioPipeline.addNode(gainNode, connections: [])
        try audioPipeline.startStream()
        
        // Process buffer
        let success = try audioPipeline.process(
            inputBuffers: [inputBuffer],
            outputBuffers: [outputBuffer],
            context: audioPipeline
        )
        
        XCTAssertTrue(success, "Processing should succeed")
        
        // Verify output
        let processedData = try AudioPipelineTestHelpers.getBufferContents(
            outputBuffer,
            pipeline: audioPipeline
        )
        
        // Check RMS level
        let inputRMS = AudioPipelineTestHelpers.calculateRMSLevel(testData)
        let outputRMS = AudioPipelineTestHelpers.calculateRMSLevel(processedData)
        
        XCTAssertEqual(
            outputRMS,
            inputRMS * 2.0,
            accuracy: 0.01,
            "Output RMS level should be doubled"
        )
        
        // Check individual samples
        for (index, sample) in testData.enumerated() {
            let expectedSample = sample * 2.0
            XCTAssertEqual(
                processedData[index],
                expectedSample,
                accuracy: 0.001,
                "Sample at index \(index) should be multiplied by gain factor"
            )
        }
    }
}

