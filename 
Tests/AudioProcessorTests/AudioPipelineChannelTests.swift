import XCTest
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for audio pipeline channel processing
final class AudioPipelineChannelTests: XCTestCase {
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
    
    // MARK: - Channel Processing Tests
    
    /// Tests stereo channel handling capabilities
    /// - Verifies that:
    ///   - Pipeline processes stereo signals correctly
    ///   - Channel-aware processing maintains stereo separation
    ///   - Gain adjustments are applied per channel
    func testChannelHandling() throws {
        // Create stereo test buffers
        let inputBuffer = try audioPipeline.allocateBuffer(
            size: 2048 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        let outputBuffer = try audioPipeline.allocateBuffer(
            size: 2048 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        
        defer {
            audioPipeline.releaseBuffer(id: inputBuffer)
            audioPipeline.releaseBuffer(id: outputBuffer)
        }
        
        // Create stereo test data
        let monoData = AudioPipelineTestHelpers.generateSineWave(
            frequency: 440.0,
            duration: 0.1,
            amplitude: 0.5
        )
        
        let stereoData = AudioPipelineTestHelpers.generateTestBuffer(
            data: monoData,
            channelCount: 2
        )
        
        // Fill input buffer
        try stereoData.withUnsafeBytes { bytes in
            try audioPipeline.updateBuffer(
                id: inputBuffer,
                data: bytes.baseAddress!,
                size: stereoData.count * MemoryLayout<Float>.stride,
                options: [.waitForCompletion]
            )
        }
        
        // Configure pipeline
        try audioPipeline.reset()
        try audioPipeline.addNode(
            ChannelAwareNode(name: "Channel Test Node"),
            connections: []
        )
        try audioPipeline.startStream()
        
        // Process buffer
        let success = try audioPipeline.process(
            inputBuffers: [inputBuffer],
            outputBuffers: [outputBuffer],
            context: audioPipeline
        )
        
        XCTAssertTrue(success, "Processing should succeed")
        
        // Verify channel processing
        let processedData = try AudioPipelineTestHelpers.getBufferContents(
            outputBuffer,
            pipeline: audioPipeline
        )
        
        // Test a subset of frames to keep the verification manageable
        let framesToTest = min(20, processedData.count / 2)
        let framesToSkip = (processedData.count / 2) / framesToTest
        
        for frameIndex in stride(from: 0, to: processedData.count / 2, by: framesToSkip) {
            let leftChannelIndex = frameIndex * 2
            let rightChannelIndex = leftChannelIndex + 1
            
            let expectedLeftSample = stereoData[leftChannelIndex] * 2.0
            let expectedRightSample = stereoData[rightChannelIndex] * 0.5
            
            XCTAssertEqual(
                processedData[leftChannelIndex],
                expectedLeftSample,
                accuracy: 0.001,
                "Left channel should be amplified by 2.0"
            )
            
            XCTAssertEqual(
                processedData[rightChannelIndex],
                expectedRightSample,
                accuracy: 0.001,
                "Right channel should be attenuated by 0.5"
            )
        }
    }
    
    /// Tests channel mapping operations
    /// - Verifies that:
    ///   - ChannelMapperNode correctly swaps left and right channels
    ///   - Different frequencies in each channel are correctly swapped
    ///   - Channel mapping preserves signal quality
    func testChannelMapping() throws {
        // Create stereo test buffers
        let inputBuffer = try audioPipeline.allocateBuffer(
            size: 2048 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        let outputBuffer = try audioPipeline.allocateBuffer(
            size: 2048 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        
        defer {
            audioPipeline.releaseBuffer(id: inputBuffer)
            audioPipeline.releaseBuffer(id: outputBuffer)
        }
        
        // Create stereo test buffer with different content in each channel
        

