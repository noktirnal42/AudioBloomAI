// AudioPipelineConnectionTests.swift
import XCTest
import ActorIsolation
import Combine
@testable import AudioProcessor

@available(macOS 15.0, *)
final class AudioPipelineConnectionTests: XCTestCase {
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
    
    func testConnectingNodes() {
        let node1 = PassthroughNode(id: "node1")
        let node2 = PassthroughNode(id: "node2")
        
        pipeline.addNode(node1)
        pipeline.addNode(node2)
        
        XCTAssertNoThrow(try pipeline.connect(from: node1, to: node2))
        XCTAssertEqual(pipeline.connections.count, 1)
    }
    
    func testDisconnectingNodes() {
        let node1 = PassthroughNode(id: "node1")
        let node2 = PassthroughNode(id: "node2")
        
        pipeline.addNode(node1)
        pipeline.addNode(node2)
        XCTAssertNoThrow(try pipeline.connect(from: node1, to: node2))
        
        pipeline.disconnect(node1, from: node2)
        XCTAssertEqual(pipeline.connections.count, 0)
    }
}
