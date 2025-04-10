// Split AudioPipelineCoreTests.swift into focused files
import XCTest
@testable import AudioProcessor

@available(macOS 15.0, *)
final class AudioPipelineBasicTests: XCTestCase {
    var pipeline: AudioPipeline!
    
    override func setUp() {
        super.setUp()
        pipeline = AudioPipeline()
    }
    
    override func tearDown() {
        pipeline = nil
        super.tearDown()
    }
    
    func testBasicPipelineOperations() {
        // Test basic operations
        XCTAssertNotNil(pipeline)
        XCTAssertEqual(pipeline.nodes.count, 0)
    }
}

@available(macOS 15.0, *)
final class AudioPipelineStateTests: XCTestCase {
    var pipeline: AudioPipeline!
    
    override func setUp() {
        super.setUp()
        pipeline = AudioPipeline()
    }
    
    override func tearDown() {
        pipeline = nil
        super.tearDown()
    }
    
    func testStateManagement() {
        // Test state changes
        XCTAssertFalse(pipeline.isActive)
        pipeline.activate()
        XCTAssertTrue(pipeline.isActive)
    }
}

@available(macOS 15.0, *)
final class AudioPipelineConnectionTests: XCTestCase {
    var pipeline: AudioPipeline!
    
    override func setUp() {
        super.setUp()
        pipeline = AudioPipeline()
    }
    
    override func tearDown() {
        pipeline = nil
        super.tearDown()
    }
    
    func testConnectionHandling() {
        let node1 = createTestNode()
        let node2 = createTestNode()
        
        pipeline.addNode(node1)
        pipeline.addNode(node2)
        
        XCTAssertNoThrow(try pipeline.connect(from: node1, to: node2))
    }
    
    private func createTestNode() -> AudioNode {
        return PassthroughNode(id: UUID().uuidString)
    }
}

@available(macOS 15.0, *)
final class AudioPipelineValidationTests: XCTestCase {
    var pipeline: AudioPipeline!
    
    override func setUp() {
        super.setUp()
        pipeline = AudioPipeline()
    }
    
    override func tearDown() {
        pipeline = nil
        super.tearDown()
    }
    
    func testValidation() {
        let node1 = createTestNode()
        let node2 = createTestNode()
        
        pipeline.addNode(node1)
        pipeline.addNode(node2)
        
        XCTAssertNoThrow(try pipeline.validate())
    }
    
    private func createTestNode() -> AudioNode {
        return PassthroughNode(id: UUID().uuidString)
    }
}

@available(macOS 15.0, *)
final class AudioPipelineErrorTests: XCTestCase {
    var pipeline: AudioPipeline!
    
    override func setUp() {
        super.setUp()
        pipeline = AudioPipeline()
    }
    
    override func tearDown() {
        pipeline = nil
        super.tearDown()
    }
    
    func testErrorHandling() {
        let invalidNode = PassthroughNode(id: "invalid")
        
        XCTAssertThrowsError(try pipeline.addNode(invalidNode))
    }
}
