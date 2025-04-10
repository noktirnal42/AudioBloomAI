import XCTest
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for AudioPipelineCore initialization and configuration
final class AudioPipelineSetupTests: AudioPipelineBaseTests {

    // MARK: - Basic Initialization Tests

    /// Tests that the pipeline can be initialized with default configuration
    func testDefaultInitialization() throws {
        // Create a pipeline with default parameters
        let pipeline = AudioPipelineCore()

        // Verify default values
        XCTAssertEqual(pipeline.bufferSize, 1024, "Default buffer size should be 1024")
        XCTAssertEqual(pipeline.sampleRate, 44100.0, "Default sample rate should be 44100.0 Hz")
        XCTAssertEqual(pipeline.channelCount, 2, "Default channel count should be 2")
        XCTAssertFalse(pipeline.isRunning, "Pipeline should not be running initially")
        XCTAssertEqual(pipeline.nodeCount, 0, "Pipeline should have no nodes initially")
    }

    /// Tests that the pipeline can be initialized with custom configuration
    func testCustomInitialization() throws {
        // Create a pipeline with custom parameters
        let customBufferSize = 512
        let customSampleRate = 48000.0
        let customChannelCount = 1

        let pipeline = AudioPipelineCore(
            bufferSize: customBufferSize,
            sampleRate: customSampleRate,
            channelCount: customChannelCount
        )

        // Verify custom values
        XCTAssertEqual(pipeline.bufferSize, customBufferSize, "Buffer size doesn't match custom value")
        XCTAssertEqual(pipeline.sampleRate, customSampleRate, "Sample rate doesn't match custom value")
        XCTAssertEqual(pipeline.channelCount, customChannelCount, "Channel count doesn't match custom value")
    }

    /// Tests that the pipeline validates configuration values
    func testInvalidConfigurations() throws {
        // Test with invalid buffer size (not power of 2)
        XCTAssertThrowsError(try AudioPipelineCore(
            bufferSize: 1000,
            sampleRate: 44100.0,
            channelCount: 2
        ), "Should throw for non-power-of-2 buffer size")

        // Test with invalid sample rate (negative)
        XCTAssertThrowsError(try AudioPipelineCore(
            bufferSize: 1024,
            sampleRate: -44100.0,
            channelCount: 2
        ), "Should throw for negative sample rate")

        // Test with invalid channel count (negative)
        XCTAssertThrowsError(try AudioPipelineCore(
            bufferSize: 1024,
            sampleRate: 44100.0,
            channelCount: 0
        ), "Should throw for zero channel count")

        // Test with extreme buffer size (too large)
        XCTAssertThrowsError(try AudioPipelineCore(
            bufferSize: 1 << 20, // 1M samples
            sampleRate: 44100.0,
            channelCount: 2
        ), "Should throw for excessively large buffer size")
    }

    // MARK: - Configuration Tests

    /// Tests changing buffer size configuration
    func testBufferSizeConfiguration() throws {
        // Changing buffer size should update the pipeline configuration
        let newBufferSize = 2048

        try audioPipeline.updateConfiguration(
            bufferSize: newBufferSize,
            sampleRate: nil,
            channelCount: nil
        )

        XCTAssertEqual(audioPipeline.bufferSize, newBufferSize, "Buffer size should be updated")

        // Attempting to set invalid buffer size should throw
        XCTAssertThrowsError(try audioPipeline.updateConfiguration(
            bufferSize: 999, // Not power of 2
            sampleRate: nil,
            channelCount: nil
        ), "Should throw for non-power-of-2 buffer size")

        // Cannot change buffer size while pipeline is running
        try audioPipeline.start()
        XCTAssertThrowsError(try audioPipeline.updateConfiguration(
            bufferSize: 4096,
            sampleRate: nil,
            channelCount: nil
        ), "Should throw when changing buffer size while running")
        try audioPipeline.stop()
    }

    /// Tests changing sample rate configuration
    func testSampleRateConfiguration() throws {
        // Changing sample rate should update the pipeline configuration
        let newSampleRate = 96000.0

        try audioPipeline.updateConfiguration(
            bufferSize: nil,
            sampleRate: newSampleRate,
            channelCount: nil
        )

        XCTAssertEqual(audioPipeline.sampleRate, newSampleRate, "Sample rate should be updated")

        // Attempting to set invalid sample rate should throw
        XCTAssertThrowsError(try audioPipeline.updateConfiguration(
            bufferSize: nil,
            sampleRate: 0.0,
            channelCount: nil
        ), "Should throw for zero sample rate")

        // Cannot change sample rate while pipeline is running
        try audioPipeline.start()
        XCTAssertThrowsError(try audioPipeline.updateConfiguration(
            bufferSize: nil,
            sampleRate: 48000.0,
            channelCount: nil
        ), "Should throw when changing sample rate while running")
        try audioPipeline.stop()
    }

    /// Tests changing channel count configuration
    func testChannelCountConfiguration() throws {
        // Changing channel count should update the pipeline configuration
        let newChannelCount = 4

        try audioPipeline.updateConfiguration(
            bufferSize: nil,
            sampleRate: nil,
            channelCount: newChannelCount
        )

        XCTAssertEqual(audioPipeline.channelCount, newChannelCount, "Channel count should be updated")

        // Attempting to set invalid channel count should throw
        XCTAssertThrowsError(try audioPipeline.updateConfiguration(
            bufferSize: nil,
            sampleRate: nil,
            channelCount: -1
        ), "Should throw for negative channel count")

        // Cannot change channel count while pipeline is running
        try audioPipeline.start()
        XCTAssertThrowsError(try audioPipeline.updateConfiguration(
            bufferSize: nil,
            sampleRate: nil,
            channelCount: 2
        ), "Should throw when changing channel count while running")
        try audioPipeline.stop()
    }

    // MARK: - Pipeline Node Tests

    /// Tests registering processing nodes in the pipeline
    func testNodeRegistration() throws {
        // Create and register a test node
        let testNode = TestAudioProcessingNode(name: "Test Node")
        let nodeId = try audioPipeline.registerNode(testNode)

        // Verify the node was registered
        XCTAssertEqual(audioPipeline.nodeCount, 1, "Pipeline should have one node")
        XCTAssertTrue(audioPipeline.hasNode(id: nodeId), "Pipeline should contain the node")

        // Get the node from the pipeline and verify its identity
        let retrievedNode = try audioPipeline.getNode(id: nodeId)
        XCTAssertIdentical(
            retrievedNode as AnyObject,
            testNode as AnyObject,
            "Retrieved node should be the same object"
        )
        XCTAssertEqual(
            (retrievedNode as? TestAudioProcessingNode)?.name,
            "Test Node",
            "Node name should match"
        )
    }

    /// Tests removing processing nodes from the pipeline
    func testNodeRemoval() throws {
        // Create and register multiple test nodes
        let node1 = TestAudioProcessingNode(name: "Node 1")
        let node2 = TestAudioProcessingNode(name: "Node 2")

        let id1 = try audioPipeline.registerNode(node1)
        let id2 = try audioPipeline.registerNode(node2)

        XCTAssertEqual(audioPipeline.nodeCount, 2, "Pipeline should have two nodes")

        // Remove the first node
        try audioPipeline.removeNode(id: id1)

        // Verify node count and presence
        XCTAssertEqual(audioPipeline.nodeCount, 1, "Pipeline should have one node")
        XCTAssertFalse(audioPipeline.hasNode(id: id1), "Pipeline should not contain removed node")
        XCTAssertTrue(audioPipeline.hasNode(id: id2), "Pipeline should still contain other node")

        // Remove the second node
        try audioPipeline.removeNode(id: id2)

        // Verify node count
        XCTAssertEqual(audioPipeline.nodeCount, 0, "Pipeline should have no nodes")

        // Attempting to remove a non-existent node should throw
        XCTAssertThrowsError(try audioPipeline.removeNode(id: id1), "Should throw for non-existent node")
    }

    /// Tests node processing order in the pipeline
    func testNodeProcessingOrder() throws {
        // Create test nodes with order tracking
        let node1 = OrderTrackingNode(name: "Node 1")
        let node2 = OrderTrackingNode(name: "Node 2")
        let node3 = OrderTrackingNode(name: "Node 3")

        // Register nodes in a specific order
        let id1 = try audioPipeline.registerNode(node1)
        let id2 = try audioPipeline.registerNode(node2)
        let id3 = try audioPipeline.registerNode(node3)

        // Start the pipeline and process some audio
        try audioPipeline.start()

        // Generate test buffer and process it
        let testBuffer = generateSineWave(frequency: 440.0, duration: 0.1)
        let processedBuffer = try audioPipeline.processAudio(testBuffer)

        // Stop the pipeline
        try audioPipeline.stop()

        // Verify processing order
        XCTAssertEqual(OrderTrackingNode.processingOrder.count, 3, "All nodes should have processed")
        XCTAssertEqual(OrderTrackingNode.processingOrder[0], "Node 1", "Node 1 should be processed first")
        XCTAssertEqual(OrderTrackingNode.processingOrder[1], "Node 2", "Node 2 should be processed second")
        XCTAssertEqual(OrderTrackingNode.processingOrder[2], "Node 3", "Node 3 should be processed third")

        // Change node order
        try audioPipeline.moveNodeToIndex(id: id3, index: 0)

        // Reset processing order tracking
        OrderTrackingNode.processingOrder = []

        // Process audio again with the new order
        try audioPipeline.start()
        _ = try audioPipeline.processAudio(testBuffer)
        try audioPipeline.stop()

        // Verify the new processing order
        XCTAssertEqual(OrderTrackingNode.processingOrder[0], "Node 3", "Node 3 should be processed first")
        XCTAssertEqual(OrderTrackingNode.processingOrder[1], "Node 1", "Node 1 should be processed second")
        XCTAssertEqual(OrderTrackingNode.processingOrder[2], "Node 2", "Node 2 should be processed third")
    }

    // MARK: - Test Helpers

    /// Simple test processing node for registration and removal tests
    class TestAudioProcessingNode: AudioProcessingNode {
        let name: String

        init(name: String) {
            self.name = name
        }

        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            // Simple pass-through implementation
            return buffer
        }
    }

    /// Processing node that tracks the order in which nodes are processed
    class OrderTrackingNode: AudioProcessingNode {
        static var processingOrder: [String] = []

        let name: String

        init(name: String) {
            self.name = name
        }

        func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
            // Record the processing order
            OrderTrackingNode.processingOrder.append(name)
            return buffer
        }
    }
}
