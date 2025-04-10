// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import XCTest
@testable import AudioProcessor

/// Advanced test cases for the audio pipeline component.
/// Focuses on complex pipeline configurations and error handling.
@available(macOS 15.0, *)
final class AudioPipelineAdvancedTests: XCTestCase {
    // MARK: - Test Fixtures
    
    var pipeline: AudioPipeline!
    var mockSource: MockAudioSource!
    var mockSink: MockAudioSink!
    
    override func setUp() {
        super.setUp()
        mockSource = AudioProcessorTestUtils.createMockSource()
        mockSink = AudioProcessorTestUtils.createMockSink()
        pipeline = AudioPipeline()
    }
    
    override func tearDown() {
        pipeline = nil
        mockSource = nil
        mockSink = nil
        super.tearDown()
    }
    
    // MARK: - Complex Pipeline Tests
    
    /// Tests a complex pipeline with multiple parallel branches
    func testComplexPipelineWithParallelBranches() {
        // Set up a complex pipeline:
        // Source -> [Gain1, Gain2, Gain3] -> Mixer -> Sink
        let gain1 = GainNode(id: "gain1", gain: 0.5)
        let gain2 = GainNode(id: "gain2", gain: 1.0)
        let gain3 = GainNode(id: "gain3", gain: 2.0)
        let mixer = MixerNode(id: "mixer")
        
        // Add nodes
        pipeline.addNode(mockSource)
        pipeline.addNode(gain1)
        pipeline.addNode(gain2)
        pipeline.addNode(gain3)
        pipeline.addNode(mixer)
        pipeline.addNode(mockSink)
        
        // Connect nodes
        XCTAssertNoThrow(try pipeline.connect(from: mockSource, to: gain1))
        XCTAssertNoThrow(try pipeline.connect(from: mockSource, to: gain2))
        XCTAssertNoThrow(try pipeline.connect(from: mockSource, to: gain3))
        XCTAssertNoThrow(try pipeline.connect(from: gain1, to: mixer))
        XCTAssertNoThrow(try pipeline.connect(from: gain2, to: mixer))
        XCTAssertNoThrow(try pipeline.connect(from: gain3, to: mixer))
        XCTAssertNoThrow(try pipeline.connect(from: mixer, to: mockSink))
        
        // Configure test data
        let testBuffer = AudioProcessorTestUtils.createTestBuffer(samples: [1.0, 2.0, 3.0, 4.0])
        mockSource.nextBuffer = testBuffer
        
        // Process data
        pipeline.process()
        
        // Verify output
        XCTAssertEqual(mockSink.receivedBuffers.count, 1, "Sink should receive 1 buffer")
        
        if let outputBuffer = mockSink.receivedBuffers.first {
            XCTAssertEqual(outputBuffer.count, testBuffer.count, "Output buffer should have same count as input")
            
            // Verify all gains applied and mixed (0.5 + 1.0 + 2.0 = 3.5 multiplier)
            for bufferIndex in 0..<min(outputBuffer.count, testBuffer.count) {
                XCTAssertEqual(outputBuffer[bufferIndex], testBuffer[bufferIndex] * 3.5, accuracy: 0.001)
            }
        }
    }
    
    /// Tests a pipeline with a feedback loop that should be detected
    func testPipelineWithFeedbackLoop() {
        // Create nodes for a feedback loop
        let node1 = PassthroughNode(id: "node1")
        let node2 = PassthroughNode(id: "node2")
        let node3 = PassthroughNode(id: "node3")
        
        // Add nodes
        pipeline.addNode(node1)
        pipeline.addNode(node2)
        pipeline.addNode(node3)
        
        // Create a loop: node1 -> node2 -> node3 -> node1
        XCTAssertNoThrow(try pipeline.connect(from: node1, to: node2))
        XCTAssertNoThrow(try pipeline.connect(from: node2, to: node3))
        XCTAssertNoThrow(try pipeline.connect(from: node3, to: node1))
        
        // Validate should detect the cycle
        XCTAssertThrowsError(try pipeline.validate()) { error in
            XCTAssertEqual(error as? AudioPipelineError, .cyclicalConnection)
        }
    }
    
    // MARK: - Error Handling Tests
    
    /// Tests handling of a node that throws an error during processing
    func testErrorHandlingDuringProcessing() {
        // Create a node that will throw an error
        let errorNode = ThrowingNode(id: "error_node", throwsOn: .process)
        
        // Set up pipeline
        pipeline.addNode(mockSource)
        pipeline.addNode(errorNode)
        pipeline.addNode(mockSink)
        
        // Connect nodes
        XCTAssertNoThrow(try pipeline.connect(from: mockSource, to: errorNode))
        XCTAssertNoThrow(try pipeline.connect(from: errorNode, to: mockSink))
        
        // Configure test data
        mockSource.nextBuffer = [1.0, 2.0, 3.0]
        
        // Process should not crash but report error
        let result = pipeline.process()
        XCTAssertFalse(result, "Processing should fail due to node error")
        
        // The sink should not receive any data
        XCTAssertEqual(mockSink.receivedBuffers.count, 0)
    }
    
    /// Tests handling of a node that throws an error during configuration
    func testErrorHandlingDuringConfiguration() {
        // Create a node that will throw during configuration
        let errorNode = ThrowingNode(id: "error_node", throwsOn: .configure)
        
        // Set up pipeline
        pipeline.addNode(mockSource)
        pipeline.addNode(errorNode)
        
        // Configure the error node - should throw
        XCTAssertThrowsError(try pipeline.configureNode(errorNode.id, parameters: ["test": true])) { error in
            XCTAssertEqual(error as? AudioPipelineError, .nodeConfigurationFailed)
        }
    }
}

/// A node that throws errors on command for testing error handling
@available(macOS 15.0, *)
class ThrowingNode: AudioProcessingNode {
    enum ThrowingStage {
        case configure
        case process
        case never
    }
    
    let id: String
    var name: String
    var isEnabled: Bool = true
    var throwsOn: ThrowingStage
    
    init(id: String, throwsOn: ThrowingStage) {
        self.id = id
        self.name = "ThrowingNode"
        self.throwsOn = throwsOn
    }
    
    var inputRequirements: AudioNodeIORequir

