// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import XCTest
import Combine
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for AudioPipeline connection management
@available(macOS 15.0, *)
final class AudioPipelineConnectionTests: XCTestCase {
    
    // MARK: - Test Components
    
    // Mock processing node for testing
    class MockAudioProcessingNode: AudioProcessingNode {
        let id: UUID = UUID()
        var name: String
        var isEnabled: Bool = true
        
        // Track calls for verification
        var configureCallCount = 0
        var processCallCount = 0
        var resetCallCount = 0
        
        // Track process parameters
        var lastInputBuffers: [AudioBufferID] = []
        var lastOutputBuffers: [AudioBufferID] = []
        
        // Control behavior
        var shouldSucceedProcessing: Bool = true
        var shouldThrowOnProcess: Bool = false
        var processingDelay: TimeInterval = 0
        
        let inputRequirements: AudioNodeIORequirements
        let outputCapabilities: AudioNodeIORequirements
        
        init(name: String) {
            self.name = name
            
            // Standard requirements for testing
            let standardFormat = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
            
            self.inputRequirements = AudioNodeIORequirements(
                supportedFormats: [standardFormat],
                channels: .oneOrMore,
                bufferSize: .oneOrMore,
                sampleRates: [48000]
            )
            
            self.outputCapabilities = AudioNodeIORequirements(
                supportedFormats: [standardFormat],
                channels: .oneOrMore,
                bufferSize: .oneOrMore,
                sampleRates: [48000]
            )
        }
        
        func configure(parameters: [String: Any]) throws {
            configureCallCount += 1
            if let shouldFail = parameters["shouldFail"] as? Bool, shouldFail {
                throw AudioPipelineError.invalidConfiguration
            }
        }
        
        func process(inputBuffers: [AudioBufferID], outputBuffers: [AudioBufferID], context: AudioProcessingContext) async throws -> Bool {
            processCallCount += 1
            lastInputBuffers = inputBuffers
            lastOutputBuffers = outputBuffers
            
            // Simulate processing delay if configured
            if processingDelay > 0 {
                try await Task.sleep(nanoseconds: UInt64(processingDelay * 1_000_000_000))
            }
            
            // Simulate error if configured
            if shouldThrowOnProcess {
                throw AudioPipelineError.processingNodeInitFailed
            }
            
            return shouldSucceedProcessing
        }
        
        func reset() {
            resetCallCount += 1
        }
    }
    
    // Test fixture variables
    var pipelineCore: AudioPipelineCore!
    var mockNodes: [MockAudioProcessingNode] = []
    var allocatedBuffers: [AudioBufferID] = []
    
    // MARK: - Test Setup and Teardown
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Create pipeline with default configuration
        let config = AudioPipelineConfiguration(
            enableMetalCompute: true,
            defaultFormat: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
            bufferSize: 1024,
            maxProcessingLoad: 0.8
        )
        
        do {
            pipelineCore = try AudioPipelineCore(configuration: config)
        } catch {
            // If Metal is not available, try again with Metal disabled
            pipelineCore = try AudioPipelineCore(configuration: AudioPipelineConfiguration(enableMetalCompute: false))
        }
        
        // Clean state for tests
        mockNodes = []
        allocatedBuffers = []
    }
    
    override func tearDown() async throws {
        // Clean up any allocated buffers
        for bufferId in allocatedBuffers {
            pipelineCore.releaseBuffer(id: bufferId)
        }
        allocatedBuffers = []
        
        // Clean up pipeline
        pipelineCore.stopStream()
        pipelineCore = nil
        
        try await super.tearDown()
    }
    
    // MARK: - Node Connection Tests
    
    func testAddNodeWithConnections() throws {
        // Create a source node
        let sourceNode = MockAudioProcessingNode(name: "Source Node")
        mockNodes.append(sourceNode)
        try pipelineCore.addNode(sourceNode, connections: [])
        
        // Create a destination node with connections
        let destNode = MockAudioProcessingNode(name: "Destination Node")
        mockNodes.append(destNode)
        
        // Create a connection from source to destination
        let connection = AudioNodeConnection(
            sourceNodeID: sourceNode.id,
            sourceOutputIndex: 0,
            destinationNodeID: destNode.id,
            destinationInputIndex: 0
        )
        
        try pipelineCore.addNode(destNode, connections: [connection])
        
        // Verify connections
        let nodeConnections = pipelineCore.getNodeConnections(nodeID: destNode.id)
        XCTAssertEqual(nodeConnections.count, 1)
        XCTAssertEqual(nodeConnections.first?.sourceNodeID, sourceNode.id)
        XCTAssertEqual(nodeConnections.first?.destinationNodeID, destNode.id)
    }
    
    func testRemoveNode() throws {
        // Add a node
        let mockNode = MockAudioProcessingNode(name: "Node to remove")
        mockNodes.append(mockNode)
        try pipelineCore.addNode(mockNode, connections: [])
        
        // Verify node was added
        XCTAssertNotNil(pipelineCore.getNode(nodeID: mockNode.id))
        
        // Remove the node
        try pipelineCore.removeNode(nodeID: mockNode.id)
        
        // Verify node was removed
        XCTAssertNil(pipelineCore.getNode(nodeID: mockNode.id))
        
        // Try to remove a non-existent node (should not throw)
        XCTAssertNoThrow(try pipelineCore.removeNode(nodeID: UUID()))
    }
    
    func testAddAndGetNode() throws {
        // Create a mock node
        let mockNode = MockAudioProcessingNode(name: "Test Node")
        mockNodes.append(mockNode)
        
        // Add the node to the pipeline
        try pipelineCore.addNode(mockNode, connections: [])
        
        // Verify the node was added
        let retrievedNode = pipelineCore.getNode(nodeID: mockNode.id)
        XCTAssertNotNil(retrievedNode)
        XCTAssertEqual(retrievedNode?.id, mockNode.id)
        XCTAssertEqual(retrievedNode?.name, "Test Node")
        
        // Test getAllNodes
        let allNodes = pipelineCore.getAllNodes()
        XCTAssertEqual(allNodes.count, 1)
        XCTAssertEqual(allNodes.first?.id, mockNode.id)
        
        // Test node connections
        let nodeConnections = pipelineCore.getNodeConnections(nodeID: mockNode.id)
        XCTAssertEqual(nodeConnections.count, 0)
    }
    
    func testNodeEnableDisable() throws {
        // Add a node
        let mockNode =

