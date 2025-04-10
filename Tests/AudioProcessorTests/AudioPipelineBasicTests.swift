// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import XCTest
@testable import AudioProcessor

/// Tests for basic pipeline operations and data flow.
@available(macOS 15.0, *)
final class AudioPipelineBasicTests: XCTestCase {
    var pipeline: AudioPipeline!
    var mockSource: MockAudioSource!
    var mockSink: MockAudioSink!
    
    override func setUp() {
        super.setUp()
        mockSource = MockAudioSource()
        mockSink = MockAudioSink()
        pipeline = AudioPipeline()
    }
    
    override func tearDown() {
        pipeline = nil
        mockSource = nil
        mockSink = nil
        super.tearDown()
    }
    
    // MARK: - Data Flow Tests
    
    /// Tests basic data flow through the pipeline
    func testBasicDataFlow() {
        // Set up a simple pipeline: source -> gain -> sink
        let gainNode = GainNode(id: "gain", gain: 2.0)
        
        // Add nodes
        pipeline.addNode(mockSource)
        pipeline.addNode(gainNode)
        pipeline.addNode(mockSink)
        
        // Connect nodes
        XCTAssertNoThrow(try pipeline.connect(from: mockSource, to: gainNode))
        XCTAssertNoThrow(try pipeline.connect(from: gainNode, to: mockSink))
        
        // Configure test data
        let testBuffer = createTestBuffer(samples: [1.0, 2.0, 3.0, 4.0])
        mockSource.nextBuffer = testBuffer
        
        // Process data
        pipeline.process()
        
        // Verify output
        XCTAssertEqual(mockSink.receivedBuffers.count, 1, "Sink should receive 1 buffer")
        
        if let outputBuffer = mockSink.receivedBuffers.first {
            XCTAssertEqual(outputBuffer.count, testBuffer.count, "Output buffer should have same count as input")
            
            // Verify gain was applied (multiplication by 2)
            for index in 0..<min(outputBuffer.count, testBuffer.count) {
                XCTAssertEqual(outputBuffer[index], testBuffer[index] * 2.0, accuracy: 0.001)
            }
        }
    }
    
    /// Tests pipeline with multiple processing stages
    func testMultistageProcessing() {
        // Set up a pipeline: source -> gain1 -> gain2 -> sink
        let gain1 = GainNode(id: "gain1", gain: 2.0)
        let gain2 = GainNode(id: "gain2", gain: 1.5)
        
        // Add nodes
        pipeline.addNode(mockSource)
        pipeline.addNode(gain1)
        pipeline.addNode(gain2)
        pipeline.addNode(mockSink)
        
        // Connect nodes
        XCTAssertNoThrow(try pipeline.connect(from: mockSource, to: gain1))
        XCTAssertNoThrow(try pipeline.connect(from: gain1, to: gain2))
        XCTAssertNoThrow(try pipeline.connect(from: gain2, to: mockSink))
        
        // Configure test data
        let testBuffer = createTestBuffer(samples: [1.0, 2.0, 3.0, 4.0])
        mockSource.nextBuffer = testBuffer
        
        // Process data
        pipeline.process()
        
        // Verify output
        XCTAssertEqual(mockSink.receivedBuffers.count, 1, "Sink should receive 1 buffer")
        
        if let outputBuffer = mockSink.receivedBuffers.first {
            XCTAssertEqual(outputBuffer.count, testBuffer.count, "Output buffer should have same count as input")
            
            // Verify gain was applied (multiplication by 2 then by 1.5 = 3)
            for index in 0..<min(outputBuffer.count, testBuffer.count) {
                XCTAssertEqual(outputBuffer[index], testBuffer[index] * 3.0, accuracy: 0.001)
            }
        }
    }
    
    /// Tests pipeline with branching flow
    func testBranchingFlow() {
        // Set up a pipeline with branches: source -> [gain1, gain2] -> mixer -> sink
        let gain1 = GainNode(id: "gain1", gain: 2.0)
        let gain2 = GainNode(id: "gain2", gain: 0.5)
        let mixer = MixerNode(id: "mixer")
        
        // Add nodes
        pipeline.addNode(mockSource)
        pipeline.addNode(gain1)
        pipeline.addNode(gain2)
        pipeline.addNode(mixer)
        pipeline.addNode(mockSink)
        
        // Connect nodes
        XCTAssertNoThrow(try pipeline.connect(from: mockSource, to: gain1))
        XCTAssertNoThrow(try pipeline.connect(from: mockSource, to: gain2))
        XCTAssertNoThrow(try pipeline.connect(from: gain1, to: mixer))
        XCTAssertNoThrow(try pipeline.connect(from: gain2, to: mixer))
        XCTAssertNoThrow(try pipeline.connect(from: mixer, to: mockSink))
        
        // Configure test data
        let testBuffer = createTestBuffer(samples: [1.0, 2.0, 3.0, 4.0])
        mockSource.nextBuffer = testBuffer
        
        // Process data
        pipeline.process()
        
        // Verify output
        XCTAssertEqual(mockSink.receivedBuffers.count, 1, "Sink should receive 1 buffer")
        
        if let outputBuffer = mockSink.receivedBuffers.first {
            XCTAssertEqual(outputBuffer.count, testBuffer.count, "Output buffer should have same count as input")
            
            // Verify mixing (2.0*x + 0.5*x = 2.5*x)
            for index in 0..<min(outputBuffer.count, testBuffer.count) {
                XCTAssertEqual(outputBuffer[index], testBuffer[index] * 2.5, accuracy: 0.001)
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func createTestBuffer(samples: [Float]) -> [Float] {
        return samples
    }
}

