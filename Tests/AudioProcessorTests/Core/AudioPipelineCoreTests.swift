import XCTest
import Combine
import AVFoundation
import Metal
import Logging
@testable import AudioProcessor
@testable import AudioBloomCore

@available(macOS 15.0, *)
final class AudioPipelineCoreTests: XCTestCase {

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

    // MARK: - Initialization and Configuration Tests

    func testInitialization() throws {
        // Test basic initialization succeeded
        XCTAssertNotNil(pipelineCore)

        // Check that the default format is set correctly
        XCTAssertEqual(pipelineCore.format.sampleRate, 48000)
        XCTAssertEqual(pipelineCore.format.channelCount, 2)

        // Verify stream is not active on initialization
        XCTAssertFalse(pipelineCore.streamStatus.isActive)

        // Test Metal integration availability (will vary by system)
        let hasMetalCompute = (pipelineCore.metalCommandQueue != nil)
        print("Test system Metal compute availability: \(hasMetalCompute)")
    }

    func testInitializationWithCustomConfiguration() throws {
        // Test initialization with custom configuration
        let customConfig = AudioPipelineConfiguration(
            enableMetalCompute: false,
            defaultFormat: AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!,
            bufferSize: 512,
            maxProcessingLoad: 0.5
        )

        let customPipeline = try AudioPipelineCore(configuration: customConfig)
        XCTAssertNotNil(customPipeline)

        // Verify custom configuration was applied
        XCTAssertEqual(customPipeline.format.sampleRate, 44100)
        XCTAssertEqual(customPipeline.format.channelCount, 1)
        XCTAssertNil(customPipeline.metalCommandQueue, "Metal should be disabled")
    }

    // MARK: - Stream Management Tests

    func testStreamConfiguration() async throws {
        // Configure stream with valid parameters
        let customFormat = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!

        let result = try await pipelineCore.configureStream(
            format: customFormat,
            bufferSize: 512,
            channels: 1
        )

        XCTAssertTrue(result, "Stream configuration should succeed")
        XCTAssertEqual(pipelineCore.format.sampleRate, 44100)
        XCTAssertEqual(pipelineCore.format.channelCount, 1)

        // Test stream status was updated
        XCTAssertFalse(pipelineCore.streamStatus.isActive)
        XCTAssertEqual(pipelineCore.streamStatus.droppedFrames, 0)
        XCTAssertGreaterThan(pipelineCore.streamStatus.latencyMs, 0)
    }

    func testStreamConfigurationWithInvalidParameters() async throws {
        // Test with invalid buffer size
        do {
            _ = try await pipelineCore.configureStream(
                format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
                bufferSize: 0,
                channels: 2
            )
            XCTFail("Should have thrown an error for invalid buffer size")
        } catch {
            XCTAssertTrue(error is AudioPipelineError)
        }

        // Test with invalid channel count
        do {
            _ = try await pipelineCore.configureStream(
                format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
                bufferSize: 1024,
                channels: 0
            )
            XCTFail("Should have thrown an error for invalid channel count")
        } catch {
            XCTAssertTrue(error is AudioPipelineError)
        }
    }

    func testStartStopStream() async throws {
        // Start the stream
        try await pipelineCore.startStream()
        XCTAssertTrue(pipelineCore.streamStatus.isActive)

        // Try starting again (should not throw)
        try await pipelineCore.startStream()
        XCTAssertTrue(pipelineCore.streamStatus.isActive)

        // Stop the stream
        pipelineCore.stopStream()
        XCTAssertFalse(pipelineCore.streamStatus.isActive)

        // Stop again (should not throw)
        pipelineCore.stopStream()
        XCTAssertFalse(pipelineCore.streamStatus.isActive)
    }

    // MARK: - Buffer Management Tests

    func testBufferAllocation() throws {
        // Test CPU buffer allocation
        let cpuBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        allocatedBuffers.append(cpuBufferId)
        XCTAssertNotNil(cpuBufferId)

        let cpuBuffer = try pipelineCore.getBuffer(id: cpuBufferId)
        XCTAssertNotNil(cpuBuffer.cpuBuffer)
        XCTAssertNil(cpuBuffer.metalBuffer)
        XCTAssertEqual(cpuBuffer.size, 1024)
        XCTAssertEqual(cpuBuffer.type, .cpu)

        // Test shared buffer allocation if Metal is available
        if pipelineCore.metalCommandQueue != nil {
            let sharedBufferId = try pipelineCore.allocateBuffer(size: 2048, type: .shared)
            allocatedBuffers.append(sharedBufferId)

            let sharedBuffer = try pipelineCore.getBuffer(id: sharedBufferId)
            XCTAssertNotNil(sharedBuffer.cpuBuffer)
            XCTAssertNotNil(sharedBuffer.metalBuffer)
            XCTAssertEqual(sharedBuffer.size, 2048)
            XCTAssertEqual(sharedBuffer.type, .shared)
        }
    }

    func testBufferUpdate() throws {
        // Allocate a test buffer
        let bufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        allocatedBuffers.append(bufferId)

        // Create test data
        var testData = [UInt8](repeating: 0, count: 1024)
        for index in 0..<1024 {
            testData[i] = UInt8(i % 256)
        }

        // Update the buffer
        try testData.withUnsafeBytes { rawBufferPointer in
            try pipelineCore.updateBuffer(
                id: bufferId,
                data: rawBufferPointer.baseAddress!,
                size: 1024,
                options: [.waitForCompletion]
            )
        }

        // Verify the update
        let buffer = try pipelineCore.getBuffer(id: bufferId)
        let bufferContents = UnsafeBufferPointer<UInt8>(
            start: buffer.cpuBuffer?.assumingMemoryBound(to: UInt8.self),
            count: 1024
        )

        // Check the first few bytes
        XCTAssertEqual(bufferContents[0], 0)
        XCTAssertEqual(bufferContents[1], 1)
        XCTAssertEqual(bufferContents[2], 2)
    }

    func testBufferRelease() throws {
        // Allocate and immediately release a buffer
        let bufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        pipelineCore.releaseBuffer(id: bufferId)

        // Try to get the released buffer (should throw)
        do {
            _ = try pipelineCore.getBuffer(id: bufferId)
            XCTFail("Should have thrown an error for accessing released buffer")
        } catch {
            XCTAssertTrue(error is AudioPipelineError)
        }

        // Test releasing a non-existent buffer (should not throw)
        pipelineCore.releaseBuffer(id: AudioBufferID(9999))
    }

    // MARK: - Node Management Tests

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
            destinationInputIndex:
        // Add a node
        let mockNode = MockAudioProcessingNode(name: "Toggleable Node")
        mockNodes.append(mockNode)
        try pipelineCore.addNode(mockNode, connections: [])

        // Initial state should be enabled
        XCTAssertTrue(mockNode.isEnabled)

        // Disable the node
        try pipelineCore.setNodeEnabled(nodeID: mockNode.id, enabled: false)

        // Verify node was disabled
        XCTAssertFalse(mockNode.isEnabled)

        // Enable the node
        try pipelineCore.setNodeEnabled(nodeID: mockNode.id, enabled: true)

        // Verify node was enabled
        XCTAssertTrue(mockNode.isEnabled)

        // Test with non-existent node (should throw)
        do {
            try pipelineCore.setNodeEnabled(nodeID: UUID(), enabled: true)
            XCTFail("Should have thrown an error for non-existent node")
        } catch {
            XCTAssertTrue(error is AudioPipelineError)
        }
    }

    // MARK: - Processing Chain Operation Tests

    func testBasicProcessing() async throws {
        // Configure and start stream
        try await pipelineCore.configureStream(
            format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
            bufferSize: 1024,
            channels: 2
        )
        try await pipelineCore.startStream()

        // Create a simple linear chain with three nodes
        let inputNode = MockAudioProcessingNode(name: "Input Node")
        let processorNode = MockAudioProcessingNode(name: "Processor Node")
        let outputNode = MockAudioProcessingNode(name: "Output Node")

        // Add them to our tracking
        mockNodes.append(inputNode)
        mockNodes.append(processorNode)
        mockNodes.append(outputNode)

        // Add nodes to the pipeline
        try pipelineCore.addNode(inputNode, connections: [])

        // Connect input to processor
        let inputToProcessorConnection = AudioNodeConnection(
            sourceNodeID: inputNode.id,
            sourceOutputIndex: 0,
            destinationNodeID: processorNode.id,
            destinationInputIndex: 0
        )
        try pipelineCore.addNode(processorNode, connections: [inputToProcessorConnection])

        // Connect processor to output
        let processorToOutputConnection = AudioNodeConnection(
            sourceNodeID: processorNode.id,
            sourceOutputIndex: 0,
            destinationNodeID: outputNode.id,
            destinationInputIndex: 0
        )
        try pipelineCore.addNode(outputNode, connections: [processorToOutputConnection])

        // Create input and output buffers
        let inputBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        let outputBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        allocatedBuffers.append(inputBufferId)
        allocatedBuffers.append(outputBufferId)

        // Fill input buffer with test data
        var testData = [Float](repeating: 0.5, count: 256)
        try testData.withUnsafeBytes { rawBufferPointer in
            try pipelineCore.updateBuffer(
                id: inputBufferId,
                data: rawBufferPointer.baseAddress!,
                size: 1024,
                options: [.waitForCompletion]
            )
        }

        // Process the chain
        let success = try await pipelineCore.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )

        // Verify processing succeeded
        XCTAssertTrue(success)

        // Verify each node was called
        XCTAssertEqual(inputNode.processCallCount, 1)
        XCTAssertEqual(processorNode.processCallCount, 1)
        XCTAssertEqual(outputNode.processCallCount, 1)
    }

    func testProcessingWithDisabledNode() async throws {
        // Configure and start stream
        try await pipelineCore.configureStream(
            format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
            bufferSize: 1024,
            channels: 2
        )
        try await pipelineCore.startStream()

        // Create two nodes
        let sourceNode = MockAudioProcessingNode(name: "Source Node")
        let disabledNode = MockAudioProcessingNode(name: "Disabled Node")
        mockNodes.append(sourceNode)
        mockNodes.append(disabledNode)

        // Add nodes to pipeline
        try pipelineCore.addNode(sourceNode, connections: [])
        try pipelineCore.addNode(disabledNode, connections: [])

        // Disable the second node
        try pipelineCore.setNodeEnabled(nodeID: disabledNode.id, enabled: false)

        // Create input and output buffers
        let inputBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        let outputBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        allocatedBuffers.append(inputBufferId)
        allocatedBuffers.append(outputBufferId)

        // Process the chain
        let success = try await pipelineCore.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )

        // Verify processing succeeded
        XCTAssertTrue(success)

        // Verify only the enabled node was processed
        XCTAssertEqual(sourceNode.processCallCount, 1)
        XCTAssertEqual(disabledNode.processCallCount, 0)
    }

    func testFailedProcessing() async throws {
        // Configure and start stream
        try await pipelineCore.configureStream(
            format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
            bufferSize: 1024,
            channels: 2
        )
        try await pipelineCore.startStream()

        // Create a node configured to fail
        let failingNode = MockAudioProcessingNode(name: "Failing Node")
        failingNode.shouldSucceedProcessing = false
        mockNodes.append(failingNode)

        // Add node to pipeline
        try pipelineCore.addNode(failingNode, connections: [])

        // Create input and output buffers
        let inputBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        let outputBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        allocatedBuffers.append(inputBufferId)
        allocatedBuffers.append(outputBufferId)

        // Process the chain
        let success = try await pipelineCore.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )

        // Verify processing failed
        XCTAssertFalse(success)

        // Verify the node was called
        XCTAssertEqual(failingNode.processCallCount, 1)
    }

    func testErrorThrownDuringProcessing() async throws {
        // Configure and start stream
        try await pipelineCore.configureStream(
            format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
            bufferSize: 1024,
            channels: 2
        )
        try await pipelineCore.startStream()

        // Create a node configured to throw
        let throwingNode = MockAudioProcessingNode(name: "Throwing Node")
        throwingNode.shouldThrowOnProcess = true
        mockNodes.append(throwingNode)

        // Add node to pipeline
        try pipelineCore.addNode(throwingNode, connections: [])

        // Create input and output buffers
        let inputBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        let outputBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        allocatedBuffers.append(inputBufferId)
        allocatedBuffers.append(outputBufferId)

        // Process the chain - should still complete but report failure
        let success = try await pipelineCore.process(
            inputBuffers: [inputBufferId],
            outputBuffers: [outputBufferId],
            context: pipelineCore
        )

        // Verify processing failed
        XCTAssertFalse(success)

        // Check stream status - should show dropped frames
        XCTAssertGreaterThan(pipelineCore.streamStatus.droppedFrames, 0)
    }

    // MARK: - Thread Safety Tests

    func testConcurrentBufferOperations() async throws {
        // Test concurrent buffer allocation and release
        let concurrentTasks = 100
        let taskGroup = DispatchGroup()
        let queue = DispatchQueue(label: "com.audiobloom.test.concurrent", attributes: .concurrent)

        var createdBuffers: [AudioBufferID] = []
        let bufferAccessLock = NSLock()

        // Launch concurrent buffer operations
        for _ in 0..<concurrentTasks {
            taskGroup.enter()
            queue.async {
                do {
                    // Allocate a buffer
                    let bufferId = try self.pipelineCore.allocateBuffer(size: 1024, type: .cpu)

                    // Update the buffer
                    var testData = [UInt8](repeating: 0, count: 1024)
                    try testData.withUnsafeBytes { rawBufferPointer in
                        try self.pipelineCore.updateBuffer(
                            id: bufferId,
                            data: rawBufferPointer.baseAddress!,
                            size: 1024,
                            options: [.waitForCompletion]
                        )
                    }

                    // Track the buffer for later cleanup
                    bufferAccessLock.lock()
                    createdBuffers.append(bufferId)
                    bufferAccessLock.unlock()
                } catch {
                    XCTFail("Concurrent buffer operation failed: \(error)")
                }
                taskGroup.leave()
            }
        }

        // Wait for all tasks to complete
        taskGroup.wait()

        // Check that all buffers were created successfully
        XCTAssertEqual(createdBuffers.count, concurrentTasks)

        // Clean up
        for bufferId in createdBuffers {
            pipelineCore.releaseBuffer(id: bufferId)
        }
    }

    func testConcurrentNodeOperations() async throws {
        // Configure stream
        try await pipelineCore.configureStream(
            format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
            bufferSize: 1024,
            channels: 2
        )

        // Create nodes for testing
        let nodesToCreate = 50
        var createdNodes: [UUID] = []
        let nodeAccessLock = NSLock()

        // Use task group for concurrent operations
        await withTaskGroup(of: Void.self) { group in
            for index in 0..<nodesToCreate {
                group.addTask {
                    do {
                        // Create and add a node
                        let node = MockAudioProcessingNode(name: "Concurrent Node \(i)")
                        try await Task.yield() // Encourage task interleaving
                        try self.pipelineCore.addNode(node, connections: [])

                        // Track node for verification
                        nodeAccessLock.lock()
                        createdNodes.append(node.id)
                        nodeAccessLock.unlock()

                        // Simulate varying workloads
                        if i % 5 == 0 {
                            try await Task.sleep(nanoseconds: 1_000_000) // 1ms
                        }
                    } catch {
                        XCTFail("Concurrent node operation failed: \(error)")
                    }
                }
            }
        }

        // Verify all nodes were added
        XCTAssertEqual(createdNodes.count, nodesToCreate)

        // Verify we can get all nodes
        let allNodes = pipelineCore.getAllNodes()
        XCTAssertEqual(allNodes.count, nodesToCreate)
    }

    func testConcurrentStreamAndBufferOperations() async throws {
        // Start with stream configuration
        try await pipelineCore.configureStream(
            format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
            bufferSize: 1024,
            channels: 2
        )

        // Create tasks to start and stop the stream while allocating buffers
        await withTaskGroup(of: Void.self) { group in
            // Task 1: Start and stop stream repeatedly
            group.addTask {
                for _ in 0..<10 {
                    do {
                        try await self.pipelineCore.startStream()
                        try await Task.sleep(nanoseconds: 10_000_000) // 10ms
                        self.pipelineCore.stopStream()
                        try await Task.sleep(nanoseconds: 5_000_000) // 5ms
                    } catch {
                        XCTFail("Stream operation failed: \(error)")
                    }
                }
            }

            // Task 2: Allocate and release buffers
            group.addTask {
                for _ in 0..<50 {
                    do {
                        let bufferId = try self.pipelineCore.allocateBuffer(size: 512, type: .cpu)
                        try await Task.sleep(nanoseconds: 2_000_000) // 2ms
                        self.pipelineCore.releaseBuffer(id: bufferId)
                    } catch {
                        XCTFail("Buffer operation failed: \(error)")
                    }


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

    // MARK: - Initialization and Configuration Tests

    func testInitialization() throws {
        // Test basic initialization succeeded
        XCTAssertNotNil(pipelineCore)

        // Check that the default format is set correctly
        XCTAssertEqual(pipelineCore.format.sampleRate, 48000)
        XCTAssertEqual(pipelineCore.format.channelCount, 2)

        // Verify stream is not active on initialization
        XCTAssertFalse(pipelineCore.streamStatus.isActive)

        // Test Metal integration availability (will vary by system)
        let hasMetalCompute = (pipelineCore.metalCommandQueue != nil)
        print("Test system Metal compute availability: \(hasMetalCompute)")
    }

    func testInitializationWithCustomConfiguration() throws {
        // Test initialization with custom configuration
        let customConfig = AudioPipelineConfiguration(
            enableMetalCompute: false,
            defaultFormat: AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!,
            bufferSize: 512,
            maxProcessingLoad: 0.5
        )

        let customPipeline = try AudioPipelineCore(configuration: customConfig)
        XCTAssertNotNil(customPipeline)

        // Verify custom configuration was applied
        XCTAssertEqual(customPipeline.format.sampleRate, 44100)
        XCTAssertEqual(customPipeline.format.channelCount, 1)
        XCTAssertNil(customPipeline.metalCommandQueue, "Metal should be disabled")
    }

    // MARK: - Stream Management Tests

    func testStreamConfiguration() async throws {
        // Configure stream with valid parameters
        let customFormat = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!

        let result = try await pipelineCore.configureStream(
            format: customFormat,
            bufferSize: 512,
            channels: 1
        )

        XCTAssertTrue(result, "Stream configuration should succeed")
        XCTAssertEqual(pipelineCore.format.sampleRate, 44100)
        XCTAssertEqual(pipelineCore.format.channelCount, 1)

        // Test stream status was updated
        XCTAssertFalse(pipelineCore.streamStatus.isActive)
        XCTAssertEqual(pipelineCore.streamStatus.droppedFrames, 0)
        XCTAssertGreaterThan(pipelineCore.streamStatus.latencyMs, 0)
    }

    func testStreamConfigurationWithInvalidParameters() async throws {
        // Test with invalid buffer size
        do {
            _ = try await pipelineCore.configureStream(
                format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
                bufferSize: 0,
                channels: 2
            )
            XCTFail("Should have thrown an error for invalid buffer size")
        } catch {
            XCTAssertTrue(error is AudioPipelineError)
        }

        // Test with invalid channel count
        do {
            _ = try await pipelineCore.configureStream(
                format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!,
                bufferSize: 1024,
                channels: 0
            )
            XCTFail("Should have thrown an error for invalid channel count")
        } catch {
            XCTAssertTrue(error is AudioPipelineError)
        }
    }

    func testStartStopStream() async throws {
        // Start the stream
        try await pipelineCore.startStream()
        XCTAssertTrue(pipelineCore.streamStatus.isActive)

        // Try starting again (should not throw)
        try await pipelineCore.startStream()
        XCTAssertTrue(pipelineCore.streamStatus.isActive)

        // Stop the stream
        pipelineCore.stopStream()
        XCTAssertFalse(pipelineCore.streamStatus.isActive)

        // Stop again (should not throw)
        pipelineCore.stopStream()
        XCTAssertFalse(pipelineCore.streamStatus.isActive)
    }

    // MARK: - Buffer Management Tests

    func testBufferAllocation() throws {
        // Test CPU buffer allocation
        let cpuBufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        allocatedBuffers.append(cpuBufferId)
        XCTAssertNotNil(cpuBufferId)

        let cpuBuffer = try pipelineCore.getBuffer(id: cpuBufferId)
        XCTAssertNotNil(cpuBuffer.cpuBuffer)
        XCTAssertNil(cpuBuffer.metalBuffer)
        XCTAssertEqual(cpuBuffer.size, 1024)
        XCTAssertEqual(cpuBuffer.type, .cpu)

        // Test shared buffer allocation if Metal is available
        if pipelineCore.metalCommandQueue != nil {
            let sharedBufferId = try pipelineCore.allocateBuffer(size: 2048, type: .shared)
            allocatedBuffers.append(sharedBufferId)

            let sharedBuffer = try pipelineCore.getBuffer(id: sharedBufferId)
            XCTAssertNotNil(sharedBuffer.cpuBuffer)
            XCTAssertNotNil(sharedBuffer.metalBuffer)
            XCTAssertEqual(sharedBuffer.size, 2048)
            XCTAssertEqual(sharedBuffer.type, .shared)
        }
    }

    func testBufferUpdate() throws {
        // Allocate a test buffer
        let bufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        allocatedBuffers.append(bufferId)

        // Create test data
        var testData = [UInt8](repeating: 0, count: 1024)
        for index in 0..<1024 {
            testData[i] = UInt8(i % 256)
        }

        // Update the buffer
        try testData.withUnsafeBytes { rawBufferPointer in
            try pipelineCore.updateBuffer(
                id: bufferId,
                data: rawBufferPointer.baseAddress!,
                size: 1024,
                options: [.waitForCompletion]
            )
        }

        // Verify the update
        let buffer = try pipelineCore.getBuffer(id: bufferId)
        let bufferContents = UnsafeBufferPointer<UInt8>(
            start: buffer.cpuBuffer?.assumingMemoryBound(to: UInt8.self),
            count: 1024
        )

        // Check the first few bytes
        XCTAssertEqual(bufferContents[0], 0)
        XCTAssertEqual(bufferContents[1], 1)
        XCTAssertEqual(bufferContents[2], 2)
    }

    func testBufferRelease() throws {
        // Allocate and immediately release a buffer
        let bufferId = try pipelineCore.allocateBuffer(size: 1024, type: .cpu)
        pipelineCore.releaseBuffer(id: bufferId)

        // Try to get the released buffer (should throw)
        do {
            _ = try pipelineCore.getBuffer(id: bufferId)
            XCTFail("Should have thrown an error for accessing released buffer")
        } catch {
            XCTAssertTrue(error is AudioPipelineError)
        }

        // Test releasing a non-existent buffer (should not throw)
        pipelineCore.releaseBuffer(id: AudioBufferID(9999))
    }

    // MARK: - Node Management Tests

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

        // Try to remove a non-existent node (shoul



