// AudioPipelineStateTests.swift
import XCTest
import ActorIsolation
import Combine
@testable import AudioProcessor

@available(macOS 15.0, *)
final class AudioPipelineStateTests: XCTestCase {
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
    
    func testPipelineInitialization() {
        XCTAssertNotNil(pipeline, "Pipeline should be created successfully")
        XCTAssertEqual(pipeline.nodes.count, 0, "Pipeline should start with no nodes")
        XCTAssertEqual(pipeline.connections.count, 0, "Pipeline should start with no connections")
    }
    
    func testNodeAdditionAndRetrieval() {
        let node1 = PassthroughNode(id: "node1")
        let node2 = PassthroughNode(id: "node2")
        
        pipeline.addNode(node1)
        pipeline.addNode(node2)
        
        XCTAssertEqual(pipeline.nodes.count, 2, "Pipeline should have 2 nodes")
        XCTAssertTrue(pipeline.nodes.contains { -zsh.id == "node1" }, "Pipeline should contain node1")
        XCTAssertTrue(pipeline.nodes.contains { -zsh.id == "node2" }, "Pipeline should contain node2")
    }
}
