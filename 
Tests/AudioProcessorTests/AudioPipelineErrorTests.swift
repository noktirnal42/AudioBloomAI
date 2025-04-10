import XCTest
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for audio pipeline error handling
final class AudioPipelineErrorTests: XCTestCase {
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
    
    // MARK: - Error Handling Tests
    
    /// Tests pipeline error handling during processing
    /// - Verifies that:
    ///   - Pipeline properly handles node processing errors
    ///   - Error propagation maintains pipeline state
    ///   - Cleanup occurs after error conditions
    func testErrorHandling() throws {
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
        
        // Configure pipeline with node that throws errors
        try audioPipeline.reset()
        try audioPipeline.addNode(
            ErrorThrowingNode(name: "Error Node"),
            connections: []
        )
        try audioPipeline.startStream()
        
        // Process should throw an error
        XCTAssertThrowsError(
            try audioPipeline.process(
                inputBuffers: [inputBuffer],
                outputBuffers: [outputBuffer],
                context: audioPipeline
            ),
            "Pipeline should propagate node processing errors"
        ) { error in
            XCTAssertTrue(
                error is AudioProcessingError,
                "Error should be of type AudioProcessingError"
            )
        }
        
        // Verify pipeline state after error
        XCTAssertTrue(
            audioPipeline.streamStatus.isActive,
            "Pipeline should remain active after handling error"
        )
    }
    
    /// Tests pipeline shutdown during processing
    /// - Verifies that:
    ///   - Pipeline stops gracefully when shutdown during processing
    ///   - All nodes receive proper cleanup calls
    ///   - No resources are leaked during shutdown
    func testPipelineShutdown() throws {
        // Create test node that tracks operations
        let counterNode = OperationCounterNode(name: "Counter Node")
        
        // Configure pipeline
        try audioPipeline.reset()
        try audioPipeline.addNode(counterNode, connections: [])
        try audioPipeline.startStream()
        
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
        
        // Process some data
        let testData = AudioPipelineTestHelpers.generateSineWave(
            frequency: 440.0,
            duration: 0.1,
            amplitude: 0.5
        )
        
        try testData.withUnsafeBytes { bytes in
            try audioPipeline.updateBuffer(
                id: inputBuffer,
                data: bytes.baseAddress!,
                size: testData.count * MemoryLayout<Float>.stride,
                options: [.waitForCompletion]
            )
        }
        
        _ = try audioPipeline.process(
            inputBuffers: [inputBuffer],
            outputBuffers: [outputBuffer],
            context: audioPipeline
        )
        
        // Verify node received process call
        XCTAssertEqual(
            counterNode.processCount,
            1,
            "Node should receive exactly one process call"
        )
        
        // Shutdown pipeline
        audioPipeline.stopStream()
        
        // Verify cleanup
        XCTAssertEqual(
            counterNode.resetCallCount,
            1,
            "Node should receive reset call during shutdown"
        )
        XCTAssertFalse(
            audioPipeline.streamStatus.isActive,
            "Pipeline should be inactive after shutdown"
        )
        
        // Attempt to process after shutdown
        XCTAssertThrowsError(
            try audioPipeline.process(
                inputBuffers: [inputBuffer],
                outputBuffers: [outputBuffer],
                context: audioPipeline
            ),
            "Processing should fail after shutdown"
        ) { error in
            XCTAssertTrue(
                error is AudioPipelineError,
                "Error should be of type AudioPipelineError"
            )
        }
    }
}

