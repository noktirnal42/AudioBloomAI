// AudioPipelineValidationTests.swift
import XCTest
import ActorIsolation
import Combine
@testable import AudioProcessor

@available(macOS 15.0, *)
final class AudioPipelineValidationTests: XCTestCase {
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
    
    func testPipelineValidation() {
        let source = mockSource
        let sink = mockSink
        let processor = GainNode(id: "gain", gain: 2.0)
        
        pipeline.addNode(source)
        pipeline.addNode(processor)
        pipeline.addNode(sink)
        
        XCTAssertNoThrow(try pipeline.connect(from: source, to: processor))
        XCTAssertNoThrow(try pipeline.connect(from: processor, to: sink))
        
        XCTAssertNoThrow(try pipeline.validate())
    }
    
    func testPipelineDetectsCycles() {
        let node1 = PassthroughNode(id: "node1")
        let node2 = PassthroughNode(id: "node2")
        let node3 = PassthroughNode(id: "node3")
        
        pipeline.addNode(node1)
        pipeline.addNode(node2)
        pipeline.addNode(node3)
        
        XCTAssertNoThrow(try pipeline.connect(from: node1, to: node2))
        XCTAssertNoThrow(try pipeline.connect(from: node2, to: node3))
        XCTAssertNoThrow(try pipeline.connect(from: node3, to: node1))
        
        XCTAssertThrowsError(try pipeline.validate())
    }
}
