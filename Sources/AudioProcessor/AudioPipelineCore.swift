import Foundation
import AVFoundation
import Metal
import Accelerate
import Combine
import Logging
import MetalKit
import AudioBloomCore

/// The type of audio data processing operation
@available(macOS 15.0, *)
public enum AudioProcessingType: String, Codable, CaseIterable, Identifiable {
    /// Raw audio data processing
    case raw = "Raw"
    /// FFT-based frequency analysis
    case fft = "FFT"
    /// Metal compute-based processing
    case metalCompute = "Metal Compute"
    /// Custom processing pipeline
    case custom = "Custom"
    
    public var id: String { self.rawValue }
}

/// Error types that can occur in the audio pipeline
@available(macOS 15.0, *)
public enum AudioPipelineError: Error {
    /// Metal device could not be initialized
    case metalDeviceInitFailed
    /// Failed to create Metal compute pipeline
    case metalComputePipelineFailed
    /// Buffer allocation failed
    case bufferAllocationFailed
    /// Audio stream setup failed
    case audioStreamSetupFailed
    /// Processing node initialization failed
    case processingNodeInitFailed
    /// Invalid configuration parameters
    case invalidConfiguration
    /// Concurrent access error
    case concurrentAccessError
    /// Real-time processing requirements not met
    case realTimeConstraintViolation
}

/// Protocol for audio stream handling
@available(macOS 15.0, *)
public protocol AudioStreamHandler: AnyObject {
    /// Configure the audio stream with specified parameters
    /// - Parameters:
    ///   - format: The audio format to use
    ///   - bufferSize: The size of audio buffers in frames
    ///   - channels: The number of audio channels
    /// - Returns: Success or failure
    func configureStream(format: AVAudioFormat, bufferSize: Int, channels: Int) async throws -> Bool
    
    /// Start the audio stream processing
    func startStream() async throws
    
    /// Stop the audio stream processing
    func stopStream()
    
    /// Get the current stream status
    var streamStatus: AudioStreamStatus { get }
}

/// Audio stream status information
@available(macOS 15.0, *)
public struct AudioStreamStatus {
    /// Whether the stream is active
    public let isActive: Bool
    /// Current buffer utilization (0.0 - 1.0)
    public let bufferUtilization: Double
    /// Current processing load (0.0 - 1.0)
    public let processingLoad: Double
    /// Dropped frames (if any)
    public let droppedFrames: Int
    /// Stream latency in milliseconds
    public let latencyMs: Double
    
    /// Default initializer
    public init(
        isActive: Bool = false,
        bufferUtilization: Double = 0.0,
        processingLoad: Double = 0.0,
        droppedFrames: Int = 0,
        latencyMs: Double = 0.0
    ) {
        self.isActive = isActive
        self.bufferUtilization = bufferUtilization
        self.processingLoad = processingLoad
        self.droppedFrames = droppedFrames
        self.latencyMs = latencyMs
    }
}

/// Protocol for audio buffer management
@available(macOS 15.0, *)
public protocol AudioBufferManagement: AnyObject {
    /// Allocate a new buffer of specified size
    /// - Parameters:
    ///   - size: The size of the buffer in bytes
    ///   - type: The type of buffer needed (CPU, GPU, or shared)
    /// - Returns: An identifier for the allocated buffer
    func allocateBuffer(size: Int, type: AudioBufferType) throws -> AudioBufferID
    
    /// Release a previously allocated buffer
    /// - Parameter id: The identifier of the buffer to release
    func releaseBuffer(id: AudioBufferID)
    
    /// Get a buffer for reading
    /// - Parameter id: The buffer identifier
    /// - Returns: The requested buffer
    func getBuffer(id: AudioBufferID) throws -> AudioBuffer
    
    /// Update buffer data
    /// - Parameters:
    ///   - id: The buffer identifier
    ///   - data: The new data
    ///   - options: Options for the update operation
    func updateBuffer(id: AudioBufferID, data: UnsafeRawPointer, size: Int, options: AudioBufferUpdateOptions) throws
    
    /// Synchronize buffer between CPU and GPU when needed
    /// - Parameter id: The buffer identifier
    func synchronizeBuffer(id: AudioBufferID) throws
}

/// Types of audio buffers
@available(macOS 15.0, *)
public enum AudioBufferType {
    /// CPU-accessible buffer
    case cpu
    /// GPU-accessible buffer
    case gpu
    /// Shared buffer accessible by both CPU and GPU
    case shared
}

/// Options for updating audio buffers
@available(macOS 15.0, *)
public struct AudioBufferUpdateOptions: OptionSet {
    public let rawValue: Int
    
    public init(rawValue: Int) {
        self.rawValue = rawValue
    }
    
    /// Asynchronous update without blocking
    public static let async = AudioBufferUpdateOptions(rawValue: 1 << 0)
    /// Wait for the update to complete
    public static let waitForCompletion = AudioBufferUpdateOptions(rawValue: 1 << 1)
    /// Perform any necessary synchronization
    public static let synchronize = AudioBufferUpdateOptions(rawValue: 1 << 2)
}

/// Type representing a buffer identifier
@available(macOS 15.0, *)
public struct AudioBufferID: Hashable, Equatable {
    public let id: UInt64
    
    public init(_ id: UInt64) {
        self.id = id
    }
}

/// Protocol for a unit in the audio processing chain
@available(macOS 15.0, *)
public protocol AudioProcessingNode: AnyObject {
    /// The unique identifier for this node
    var id: UUID { get }
    
    /// The display name of this node
    var name: String { get set }
    
    /// Whether this node is enabled in the processing chain
    var isEnabled: Bool { get set }
    
    /// Configure this node with parameters
    /// - Parameter parameters: Configuration parameters for this node
    func configure(parameters: [String: Any]) throws
    
    /// Process incoming audio data
    /// - Parameters:
    ///   - inputBuffers: The input audio buffer identifiers
    ///   - outputBuffers: The output audio buffer identifiers
    ///   - context: The processing context
    /// - Returns: Whether processing was successful
    func process(inputBuffers: [AudioBufferID], outputBuffers: [AudioBufferID], context: AudioProcessingContext) async throws -> Bool
    
    /// Reset the state of this node
    func reset()
    
    /// Input requirements for this node
    var inputRequirements: AudioNodeIORequirements { get }
    
    /// Output capabilities of this node
    var outputCapabilities: AudioNodeIORequirements { get }
}

/// IO requirements for an audio processing node
@available(macOS 15.0, *)
public struct AudioNodeIORequirements {
    /// Supported audio formats
    public let supportedFormats: [AVAudioFormat]
    /// Number of channels required/provided
    public let channels: CountRange
    /// Buffer size requirements
    public let bufferSize: CountRange
    /// Sample rate requirements
    public let sampleRates: [Double]
    
    /// Value range for count-based requirements
    public struct CountRange {
        // Check if any other nodes depend on this one
        for (otherNodeID, nodeConnections) in connections {
            for connection in nodeConnections {
                if connection.destinationNodeID == nodeID || connection.sourceNodeID == nodeID {
                    logger.error("Cannot remove node \(nodeID) because it is referenced by node \(otherNodeID)")
                    throw AudioPipelineError.invalidConfiguration
                }
            }
        }
        
        // Remove the node and its connections
        nodes.removeValue(forKey: nodeID)
        connections.removeValue(forKey: nodeID)
        
        logger.info("Node removed successfully: id=\(nodeID)")
    }
    
    /// Enable or disable a node
    /// - Parameters:
    ///   - nodeID: The ID of the node to modify
    ///   - enabled: Whether the node should be enabled
    public func setNodeEnabled(nodeID: UUID, enabled: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        
        logger.debug("Setting node enabled state: id=\(nodeID), enabled=\(enabled)")
        
        guard let node = nodes[nodeID] else {
            logger.error("Attempted to modify non-existent node: \(nodeID)")
            throw AudioPipelineError.invalidConfiguration
        }
        
        // Update the node's enabled state
        node.isEnabled = enabled
        
        logger.debug("Node enabled state updated: id=\(nodeID), enabled=\(enabled)")
    }
    
    /// Get a node by its ID
    /// - Parameter nodeID: The ID of the node to retrieve
    /// - Returns: The requested node, if it exists
    public func getNode(nodeID: UUID) -> AudioProcessingNode? {
        lock.lock()
        defer { lock.unlock() }
        
        return nodes[nodeID]
    }
    
    /// Get all nodes in the processing chain
    /// - Returns: An array of all nodes
    public func getAllNodes() -> [AudioProcessingNode] {
        lock.lock()
        defer { lock.unlock() }
        
        return Array(nodes.values)
    }
    
    /// Get the connections for a specific node
    /// - Parameter nodeID: The ID of the node
    /// - Returns: The node's connections
    public func getNodeConnections(nodeID: UUID) -> [AudioNodeConnection] {
        lock.lock()
        defer { lock.unlock() }
        
        return connections[nodeID] ?? []
    }
    
    /// Process audio through the entire chain
    /// - Parameters:
    ///   - inputBuffers: The input audio buffer identifiers
    ///   - outputBuffers: The output audio buffer identifiers
    ///   - context: The processing context
    /// - Returns: Whether processing was successful
    public func process(inputBuffers: [AudioBufferID], outputBuffers: [AudioBufferID], context: AudioProcessingContext) async throws -> Bool {
        lock.lock()
        
        // Check if the stream is active
        guard _isStreamActive else {
            lock.unlock()
            logger.warning("Attempted to process audio with inactive stream")
            return false
        }
        
        // Start tracking processing time for performance monitoring
        loadTracker.startProcessingCycle()
        
        // Create a mapping of intermediate buffers between nodes
        var intermediateBuffers: [UUID: [AudioBufferID]] = [:]
        var processingOrder: [UUID] = []
        
        // Determine processing order using topological sort
        do {
            processingOrder = try getTopologicalNodeOrder()
        } catch {
            lock.unlock()
            logger.error("Failed to determine processing order: \(error)")
            throw AudioPipelineError.invalidConfiguration
        }
        
        // Release the lock during actual processing to allow parallel operations
        lock.unlock()
        
        var success = true
        
        do {
            // Process each node in order
            for nodeID in processingOrder {
                guard let node = getNode(nodeID) else {
                    logger.error("Node disappeared during processing: \(nodeID)")
                    continue
                }
                
                // Skip disabled nodes
                if !node.isEnabled {
                    continue
                }
                
                // Determine input buffers for this node
                let nodeInputs: [AudioBufferID]
                if processingOrder.first == nodeID {
                    // First node gets the original input buffers
                    nodeInputs = inputBuffers
                } else {
                    // Other nodes get intermediate buffers from their connected inputs
                    var inputs: [AudioBufferID] = []
                    for connection in getNodeConnections(nodeID: nodeID) {
                        if let sourceBuffers = intermediateBuffers[connection.sourceNodeID],
                           connection.sourceOutputIndex < sourceBuffers.count {
                            inputs.append(sourceBuffers[connection.sourceOutputIndex])
                        }
                    }
                    nodeInputs = inputs
                }
                
                // Determine output buffers for this node
                let nodeOutputs: [AudioBufferID]
                if processingOrder.last == nodeID {
                    // Last node outputs to the final output buffers
                    nodeOutputs = outputBuffers
                } else {
                    // Create intermediate buffers for this node's outputs
                    var outputs: [AudioBufferID] = []
                    
                    // Allocate buffers based on the node's output capabilities
                    let outputCount = max(1, node.outputCapabilities.channels.min)
                    let sampleCount = Int(format.sampleRate / 100) // 10ms buffer
                    let bytesPerFrame = format.streamDescription.pointee.mBytesPerFrame
                    let bufferSize = sampleCount * Int(bytesPerFrame)
                    
                    for _ in 0..<outputCount {
                        do {
                            let buffer = try allocateBuffer(size: bufferSize, type: .shared)
                            outputs.append(buffer)
                        } catch {
                            logger.error("Failed to allocate intermediate buffer: \(error)")
                            throw AudioPipelineError.bufferAllocationFailed
                        }
                    }
                    
                    intermediateBuffers[nodeID] = outputs
                    nodeOutputs = outputs
                }
                
                // Process this node
                do {
                    let nodeSuccess = try await node.process(
                        inputBuffers: nodeInputs,
                        outputBuffers: nodeOutputs,
                        context: self
                    )
                    
                    if !nodeSuccess {
                        logger.warning("Node processing returned failure: id=\(nodeID), name=\(node.name)")
                        success = false
                    }
                } catch {
                    logger.error("Error processing node: id=\(nodeID), error=\(error)")
                    success = false
                }
            }
        } catch {
            logger.error("Processing chain failed: \(error)")
            success = false
        }
        
        // Clean up intermediate buffers
        lock.lock()
        defer { 
            lock.unlock()
            // Finish tracking processing time
            loadTracker.endProcessingCycle()
            
            // Update processing load in stream status
            _streamStatus = AudioStreamStatus(
                isActive: _streamStatus.isActive,
                bufferUtilization: _streamStatus.bufferUtilization,
                processingLoad: loadTracker.averageLoad,
                droppedFrames: _streamStatus.droppedFrames + (success ? 0 : 1),
                latencyMs: _streamStatus.latencyMs
            )
            
            // Update sample time
            _sampleTime += Double(format.streamDescription.pointee.mFramesPerPacket) / format.sampleRate
        }
        
        // Release intermediate buffers
        for (_, buffers) in intermediateBuffers {
            for buffer in buffers {
                releaseBuffer(id: buffer)
            }
        }
        
        return success
    }
    
    /// Determines the order in which nodes should be processed using topological sort
    /// - Returns: An ordered array of node IDs
    private func getTopologicalNodeOrder() throws -> [UUID] {
        var visited: Set<UUID> = []
        var order: [UUID] = []
        
        // Create adjacency list
        var adjacencyList: [UUID: [UUID]] = [:]
        for (nodeID, nodeConnections) in connections {
            adjacencyList[nodeID] = []
            for connection in nodeConnections {
                if connection.destinationNodeID != UUID.init(uuidString: "00000000-0000-0000-0000-000000000001")! {
                    adjacencyList[nodeID]?.append(connection.destinationNodeID)
                }
            }
        }
        
        // Visit each node
        for nodeID in nodes.keys {
            if !visited.contains(nodeID) {
                try visitNode(nodeID, adjacencyList: adjacencyList, visited: &visited, order: &order)
            }
        }
        
        return order.reversed()
    }
    
    /// Helper function for topological sort
    private func visitNode(_ nodeID: UUID, adjacencyList: [UUID: [UUID]], visited: inout Set<UUID>, order: inout [UUID]) throws {
        // Mark as visiting
        visited.insert(nodeID)
        
        // Visit all neighbors
        if let neighbors = adjacencyList[nodeID] {
            for neighborID in neighbors {
                if !visited.contains(neighborID) {
                    try visitNode(neighborID, adjacencyList: adjacencyList, visited: &visited, order: &order)
                }
            }
        }
        
        // Add to result
        order.append(nodeID)
    }
    
    /// Clean up resources when the object is deallocated
    deinit {
        // Stop the stream if it's running
        if _isStreamActive {
            stopStream()
        }
        
        // Release all buffers
        for bufferID in buffers.keys {
            releaseBuffer(id: bufferID)
        }
        
        logger.info("AudioPipelineCore deinitialized")
    }
}

/// Configuration options for the audio pipeline
@available(macOS 15.0, *)
public struct AudioPipelineConfiguration {
    /// Whether to enable Metal compute capabilities
    public let enableMetalCompute: Bool
    
    /// Default audio format
    public let defaultFormat: AVAudioFormat
    
    /// Buffer size in samples
    public let bufferSize: Int
    
    /// Maximum processing load percentage (0.0-1.0)
    public let maxProcessingLoad: Double
    
    /// Default initializer
    public init(
        enableMetalCompute: Bool = true,
        defaultFormat: AVAudioFormat = AVAudioFormat(
            standardFormatWithSampleRate: 48000,
            channels: 2
        )!,
        bufferSize: Int = 1024,
        maxProcessingLoad: Double = 0.8
    ) {
        self.enableMetalCompute = enableMetalCompute
        self.defaultFormat = defaultFormat
        self.bufferSize = bufferSize
        self.maxProcessingLoad = maxProcessingLoad
    }
}

/// Tracks real-time processing load
@available(macOS 15.0, *)
private class RealTimeLoadTracker {
    /// Timestamp of processing start
    private var processingStartTime: CFAbsoluteTime = 0
    
    /// Cumulative processing time
    private var cumulativeProcessingTime: CFAbsoluteTime = 0
    
    /// Start of measurement window
    private var measurementWindowStart: CFAbsoluteTime = 0
    
    /// Number of processing cycles in the current window
    private var cyclesInWindow: Int = 0
    
    /// Size of the measurement window in seconds
    private let windowSize: CFAbsoluteTime = 1.0
    
    /// Average processing load (0.0-1.0)
    private(set) var averageLoad: Double = 0.0
    
    /// Maximum observed processing load
    private(set) var peakLoad: Double = 0.0
    
    /// Initialize a new load tracker
    init() {
        reset()
    }
    
    /// Reset all measurements
    func reset() {
        processingStartTime = 0
        cumulativeProcessingTime = 0
        measurementWindowStart = CFAbsoluteTimeGetCurrent()
        cyclesInWindow = 0
        averageLoad = 0.0
        peakLoad = 0.0
    }
    
    /// Start tracking a processing cycle
    func startProcessingCycle() {
        processingStartTime = CFAbsoluteTimeGetCurrent()
    }
    
    /// End tracking a processing cycle
    func endProcessingCycle() {
        guard processingStartTime > 0 else { return }
        
        let now = CFAbsoluteTimeGetCurrent()
        let cycleTime = now - processingStartTime
        cumulativeProcessingTime += cycleTime
        cyclesInWindow += 1
        
        // Check if the measurement window has elapsed
        if now - measurementWindowStart >= windowSize {
            // Calculate average load
            let totalTime = now - measurementWindowStart
            averageLoad = cumulativeProcessingTime / totalTime
            
            // Update peak load
            if averageLoad > peakLoad {
                peakLoad = averageLoad
            }
            
            // Reset for next window
            measurementWindowStart = now
            cumulativeProcessingTime = 0
            cyclesInWindow = 0
        }
        
        processingStartTime = 0
    }
}
        public let max: Int
        
        public init(min: Int, max: Int) {
            self.min = min
            self.max = max
        }
        
        /// Exactly one
        public static let one = CountRange(min: 1, max: 1)
        /// Zero or one
        public static let zeroOrOne = CountRange(min: 0, max: 1)
        /// One or more
        public static let oneOrMore = CountRange(min: 1, max: Int.max)
        /// Any number including zero
        public static let any = CountRange(min: 0, max: Int.max)
    }
    
    public init(
        supportedFormats: [AVAudioFormat],
        channels: CountRange,
        bufferSize: CountRange,
        sampleRates: [Double]
    ) {
        self.supportedFormats = supportedFormats
        self.channels = channels
        self.bufferSize = bufferSize
        self.sampleRates = sampleRates
    }
}

/// Protocol representing an audio processing context
@available(macOS 15.0, *)
public protocol AudioProcessingContext {
    /// The current sample time
    var sampleTime: Double { get }
    /// The current audio format
    var format: AVAudioFormat { get }
    /// The current processing chain
    var processingChain: AudioProcessingChain { get }
    /// The buffer manager
    var bufferManager: AudioBufferManagement { get }
    /// The Metal compute command queue (if available)
    var metalCommandQueue: MTLCommandQueue? { get }
    /// The logger for this context
    var logger: Logger { get }
}

/// Protocol for managing the audio processing chain
@available(macOS 15.0, *)
public protocol AudioProcessingChain: AnyObject {
    /// Add a node to the processing chain
    /// - Parameters:
    ///   - node: The node to add
    ///   - connections: The connections for this node
    func addNode(_ node: AudioProcessingNode, connections: [AudioNodeConnection]) throws
    
    /// Remove a node from the processing chain
    /// - Parameter nodeID: The ID of the node to remove
    func removeNode(nodeID: UUID) throws
    
    /// Enable or disable a node
    /// - Parameters:
    ///   - nodeID: The ID of the node to modify
    ///   - enabled: Whether the node should be enabled
    func setNodeEnabled(nodeID: UUID, enabled: Bool) throws
    
    /// Get a node by its ID
    /// - Parameter nodeID: The ID of the node to retrieve
    /// - Returns: The requested node, if it exists
    func getNode(nodeID: UUID) -> AudioProcessingNode?
    
    /// Get all nodes in the processing chain
    /// - Returns: An array of all nodes
    func getAllNodes() -> [AudioProcessingNode]
    
    /// Get the connections for a specific node
    /// - Parameter nodeID: The ID of the node
    /// - Returns: The node's connections
    func getNodeConnections(nodeID: UUID) -> [AudioNodeConnection]
    
    /// Process audio through the entire chain
    /// - Parameters:
    ///   - inputBuffers: The input audio buffer identifiers
    ///   - outputBuffers: The output audio buffer identifiers
    ///   - context: The processing context
    /// - Returns: Whether processing was successful
    func process(inputBuffers: [AudioBufferID], outputBuffers: [AudioBufferID], context: AudioProcessingContext) async throws -> Bool
}

/// Connection between audio processing nodes
@available(macOS 15.0, *)
public struct AudioNodeConnection: Hashable {
    /// Source node ID
    public let sourceNodeID: UUID
    /// Source output index
    public let sourceOutputIndex: Int
    /// Destination node ID
    public let destinationNodeID: UUID
    /// Destination input index
    public let destinationInputIndex: Int
    
    public init(
        sourceNodeID: UUID,
        sourceOutputIndex: Int,
        destinationNodeID: UUID,
        destinationInputIndex: Int
    ) {
        self.sourceNodeID = sourceNodeID
        self.sourceOutputIndex = sourceOutputIndex
        self.destinationNodeID = destinationNodeID
        self.destinationInputIndex = destinationInputIndex
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(sourceNodeID)
        hasher.combine(sourceOutputIndex)
        hasher.combine(destinationNodeID)
        hasher.combine(destinationInputIndex)
    }
    
    public static func == (lhs: AudioNodeConnection, rhs: AudioNodeConnection) -> Bool {
        return lhs.sourceNodeID == rhs.sourceNodeID &&
               lhs.sourceOutputIndex == rhs.sourceOutputIndex &&
               lhs.destinationNodeID == rhs.destinationNodeID &&
               lhs.destinationInputIndex == rhs.destinationInputIndex
    }
}

/// Audio buffer representation
@available(macOS 15.0, *)
public struct AudioBuffer {
    /// The CPU-accessible pointer (if available)
    public let cpuBuffer: UnsafeMutableRawPointer?
    /// The Metal buffer (if available)
    public let metalBuffer: MTLBuffer?
    /// The buffer size in bytes
    public let size: Int
    /// The type of this buffer
    public let type: AudioBufferType
    
    public init(
        cpuBuffer: UnsafeMutableRawPointer?,
        metalBuffer: MTLBuffer?,
        size: Int,
        type: AudioBufferType
    ) {
        self.cpuBuffer = cpuBuffer
        self.metalBuffer = metalBuffer
        self.size = size
        self.type = type
    }
}

/// Core audio pipeline implementation
@available(macOS 15.0, *)
public class AudioPipelineCore: AudioStreamHandler, AudioBufferManagement, AudioProcessingChain, AudioProcessingContext {
    /// Logger for the audio pipeline
    public let logger = Logger(label: "com.audiobloom.audio-pipeline")
    
    /// Metal device for compute operations
    private let metalDevice: MTLDevice?
    
    /// Metal command queue for submitting compute commands
    private let metalCommandQueue: MTLCommandQueue?
    
    /// The current audio format
    private var _format: AVAudioFormat
    
    /// Read-write lock for thread safety
    private let lock = NSRecursiveLock()
    
    /// Current sample time
    private var _sampleTime: Double = 0.0
    
    /// Processing nodes in the chain
    private var nodes: [UUID: AudioProcessingNode] = [:]
    
    /// Node connections in the processing chain
    private var connections: [UUID: [AudioNodeConnection]] = [:]
    
    /// Allocated buffers
    private var buffers: [AudioBufferID: AudioBuffer] = [:]
    
    /// Next buffer ID
    private var nextBufferID: UInt64 = 1
    
    /// Computational load tracker
    private var loadTracker = RealTimeLoadTracker()
    
    /// Stream status information
    private var _streamStatus = AudioStreamStatus()
    
    /// Whether the stream is currently active
    private var _isStreamActive = false
    
    /// Audio processing queue
    private let processingQueue = DispatchQueue(
        label: "com.audiobloom.audio-pipeline.processing",
        qos: .userInteractive
    )
    
    /// Initializes the audio pipeline with default configuration
    /// - Parameter configuration: Pipeline configuration options
    public init(configuration: AudioPipelineConfiguration = AudioPipelineConfiguration()) {
        // Set up Metal for GPU-accelerated processing
        if configuration.enableMetalCompute {
            self.metalDevice = MTLCreateSystemDefaultDevice()
            self.metalCommandQueue = metalDevice?.makeCommandQueue()
            
            if self.metalDevice == nil || self.metalCommandQueue == nil {
                logger.warning("Metal device initialization failed, falling back to CPU processing")
                throw AudioPipelineError.metalDeviceInitFailed
            } else {
                logger.info("Metal compute enabled with device: \(metalDevice!.name)")
            }
        } else {
            self.metalDevice = nil
            self.metalCommandQueue = nil
            logger.info("Metal compute disabled in configuration")
        }
        
        // Initialize with default audio format
        self._format = configuration.defaultFormat
        
        logger.info("Audio Pipeline Core initialized with configuration: \(configuration)")
    }
    
    // MARK: - AudioProcessingContext Protocol Implementation
    
    /// The current sample time
    public var sampleTime: Double {
        lock.lock()
        defer { lock.unlock() }
        return _sampleTime
    }
    
    /// The current audio format
    public var format: AVAudioFormat {
        lock.lock()
        defer { lock.unlock() }
        return _format
    }
    
    /// The current processing chain - this is self since AudioPipelineCore implements AudioProcessingChain
    public var processingChain: AudioProcessingChain { self }
    
    /// The buffer manager - this is self since AudioPipelineCore implements AudioBufferManagement
    public var bufferManager: AudioBufferManagement { self }
    
    // MARK: - AudioStreamHandler Protocol Implementation
    
    /// Configure the audio stream with specified parameters
    /// - Parameters:
    ///   - format: The audio format to use
    ///   - bufferSize: The size of audio buffers in frames
    ///   - channels: The number of audio channels
    /// - Returns: Success or failure
    public func configureStream(format: AVAudioFormat, bufferSize: Int, channels: Int) async throws -> Bool {
        lock.lock()
        defer { lock.unlock() }
        
        // Validate parameters
        guard bufferSize > 0 && channels > 0 else {
            logger.error("Invalid stream configuration: bufferSize=\(bufferSize), channels=\(channels)")
            throw AudioPipelineError.invalidConfiguration
        }
        
        logger.info("Configuring audio stream: format=\(format), bufferSize=\(bufferSize), channels=\(channels)")
        
        do {
            // Update the format
            _format = format
            
            // Reset any existing stream state
            for (_, node) in nodes {
                node.reset()
            }
            
            // Update stream status
            _streamStatus = AudioStreamStatus(
                isActive: false,
                bufferUtilization: 0.0,
                processingLoad: 0.0,
                droppedFrames: 0,
                latencyMs: Double(bufferSize) / format.sampleRate * 1000.0
            )
            
            // Reset load tracker
            loadTracker.reset()
            
            return true
        } catch {
            logger.error("Failed to configure audio stream: \(error)")
            throw AudioPipelineError.audioStreamSetupFailed
        }
    }
    
    /// Start the audio stream processing
    public func startStream() async throws {
        lock.lock()
        defer { lock.unlock() }
        
        guard !_isStreamActive else {
            logger.warning("Attempted to start already active stream")
            return
        }
        
        logger.info("Starting audio stream")
        
        // Verify that we have a valid processing chain
        if nodes.isEmpty {
            logger.warning("Starting stream with empty processing chain")
        }
        
        // Start the stream
        _isStreamActive = true
        
        // Update stream status
        _streamStatus = AudioStreamStatus(
            isActive: true,
            bufferUtilization: 0.0,
            processingLoad: 0.0,
            droppedFrames: 0,
            latencyMs: _streamStatus.latencyMs
        )
        
        // Reset load tracker
        loadTracker.reset()
        
        logger.info("Audio stream started successfully")
    }
    
    /// Stop the audio stream processing
    public func stopStream() {
        lock.lock()
        defer { lock.unlock() }
        
        guard _isStreamActive else {
            logger.warning("Attempted to stop inactive stream")
            return
        }
        
        logger.info("Stopping audio stream")
        
        // Stop the stream
        _isStreamActive = false
        
        // Update stream status
        _streamStatus = AudioStreamStatus(
            isActive: false,
            bufferUtilization: 0.0,
            processingLoad: 0.0,
            droppedFrames: 0,
            latencyMs: _streamStatus.latencyMs
        )
        
        logger.info("Audio stream stopped successfully")
    }
    
    /// Get the current stream status
    public var streamStatus: AudioStreamStatus {
        lock.lock()
        defer { lock.unlock() }
        return _streamStatus
    }
    
    // MARK: - AudioBufferManagement Protocol Implementation
    
    /// Allocate a new buffer of specified size
    /// - Parameters:
    ///   - size: The size of the buffer in bytes
    ///   - type: The type of buffer needed (CPU, GPU, or shared)
    /// - Returns: An identifier for the allocated buffer
    public func allocateBuffer(size: Int, type: AudioBufferType) throws -> AudioBufferID {
        lock.lock()
        defer { lock.unlock() }
        
        guard size > 0 else {
            logger.error("Cannot allocate buffer with size \(size)")
            throw AudioPipelineError.invalidConfiguration
        }
        
        logger.debug("Allocating buffer: size=\(size), type=\(type)")
        
        // Generate a new buffer ID
        let bufferID = AudioBufferID(nextBufferID)
        nextBufferID += 1
        
        var cpuBuffer: UnsafeMutableRawPointer? = nil
        var metalBuffer: MTLBuffer? = nil
        
        switch type {
        case .cpu:
            // Allocate CPU-accessible memory
            cpuBuffer = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 16)
            // Zero out the memory
            cpuBuffer?.initializeMemory(as: UInt8.self, repeating: 0, count: size)
            
        case .gpu:
            // Ensure we have a valid Metal device
            guard let device = metalDevice else {
                logger.error("Cannot allocate GPU buffer: no Metal device available")
                throw AudioPipelineError.metalDeviceInitFailed
            }
            
            // Create a Metal buffer for GPU access
            if let buffer = device.makeBuffer(length: size, options: .storageModePrivate) {
                metalBuffer = buffer
            } else {
                logger.error("Failed to allocate Metal buffer of size \(size)")
                throw AudioPipelineError.bufferAllocationFailed
            }
            
        case .shared:
            // Ensure we have a valid Metal device
            guard let device = metalDevice else {
                logger.error("Cannot allocate shared buffer: no Metal device available")
                throw AudioPipelineError.metalDeviceInitFailed
            }
            
            // Create a shared buffer accessible by both CPU and GPU
            if let buffer = device.makeBuffer(length: size, options: .storageModeShared) {
                metalBuffer = buffer
                cpuBuffer = buffer.contents()
            } else {
                logger.error("Failed to allocate shared Metal buffer of size \(size)")
                throw AudioPipelineError.bufferAllocationFailed
            }
        }
        
        // Create the buffer wrapper
        let buffer = AudioBuffer(
            cpuBuffer: cpuBuffer,
            metalBuffer: metalBuffer,
            size: size,
            type: type
        )
        
        // Store the buffer
        buffers[bufferID] = buffer
        
        logger.debug("Buffer allocated: id=\(bufferID.id)")
        return bufferID
    }
    
    /// Release a previously allocated buffer
    /// - Parameter id: The identifier of the buffer to release
    public func releaseBuffer(id: AudioBufferID) {
        lock.lock()
        defer { lock.unlock() }
        
        guard let buffer = buffers[id] else {
            logger.warning("Attempted to release non-existent buffer: \(id.id)")
            return
        }
        
        logger.debug("Releasing buffer: id=\(id.id)")
        
        // Free CPU memory if this was a CPU or shared buffer
        if let cpuBuffer = buffer.cpuBuffer, buffer.type != .shared {
            cpuBuffer.deallocate()
        }
        
        // Remove the buffer from our map
        buffers.removeValue(forKey: id)
    }
    
    /// Get a buffer for reading
    /// - Parameter id: The buffer identifier
    /// - Returns: The requested buffer
    public func getBuffer(id: AudioBufferID) throws -> AudioBuffer {
        lock.lock()
        defer { lock.unlock() }
        
        guard let buffer = buffers[id] else {
            logger.error("Attempted to access non-existent buffer: \(id.id)")
            throw AudioPipelineError.invalidConfiguration
        }
        
        return buffer
    }
    
    /// Update buffer data
    /// - Parameters:
    ///   - id: The buffer identifier
    ///   - data: The new data
    ///   - options: Options for the update operation
    public func updateBuffer(id: AudioBufferID, data: UnsafeRawPointer, size: Int, options: AudioBufferUpdateOptions) throws {
        lock.lock()
        defer { lock.unlock() }
        
        guard let buffer = buffers[id] else {
            logger.error("Attempted to update non-existent buffer: \(id.id)")
            throw AudioPipelineError.invalidConfiguration
        }
        
        guard size <= buffer.size else {
            logger.error("Buffer update size (\(size)) exceeds buffer capacity (\(buffer.size))")
            throw AudioPipelineError.invalidConfiguration
        }
        
        logger.debug("Updating buffer: id=\(id.id), size=\(size), options=\(options)")
        
        // Handle CPU buffer update
        if let cpuBuffer = buffer.cpuBuffer {
            // Copy the data to the CPU buffer
            cpuBuffer.copyMemory(from: data, byteCount: size)
        }
        
        // Handle GPU buffer update
        if let metalBuffer = buffer.metalBuffer, buffer.type == .gpu {
            guard let commandQueue = metalCommandQueue else {
                logger.error("Cannot update GPU buffer: no Metal command queue available")
                throw AudioPipelineError.metalDeviceInitFailed
            }
            
            // Create a shared buffer to transfer data to GPU
            guard let device = metalDevice,
                  let tempBuffer = device.makeBuffer(bytes: data, length: size, options: .storageModeShared) else {
                logger.error("Failed to create temporary buffer for GPU update")
                throw AudioPipelineError.bufferAllocationFailed
            }
            
            // Create a command buffer for the copy operation
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                logger.error("Failed to create Metal command buffer or encoder")
                throw AudioPipelineError.metalComputePipelineFailed
            }
            
            // Copy from the temp buffer to the destination buffer
            blitEncoder.copy(from: tempBuffer, sourceOffset: 0, to: metalBuffer, destinationOffset: 0, size: size)
            blitEncoder.endEncoding()
            
            // Execute the command buffer
            if options.contains(.waitForCompletion) {
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
            } else {
                commandBuffer.commit()
            }
        }
        
        // Synchronize if requested
        if options.contains(.synchronize) {
            try synchronizeBuffer(id: id)
        }
    }
    
    /// Synchronize buffer between CPU and GPU when needed
    /// - Parameter id: The buffer identifier
    public func synchronizeBuffer(id: AudioBufferID) throws {
        lock.lock()
        defer { lock.unlock() }
        
        guard let buffer = buffers[id] else {
            logger.error("Attempted to synchronize non-existent buffer: \(id.id)")
            throw AudioPipelineError.invalidConfiguration
        }
        
        // Synchronization is only needed for shared buffers
        guard buffer.type == .shared, let metalBuffer = buffer.metalBuffer else {
            return
        }
        
        logger.debug("Synchronizing buffer: id=\(id.id)")
        
        if metalBuffer.storageMode == .shared {
            // For shared storage mode, we can synchronize with a didModifyRange call
            let range = 0..<buffer.size
            metalBuffer.didModifyRange(range)
        }
    }
    
    // MARK: - AudioProcessingChain Protocol Implementation
    
    /// Add a node to the processing chain
    /// - Parameters:
    ///   - node: The node to add
    ///   - connections: The connections for this node
    public func addNode(_ node: AudioProcessingNode, connections: [AudioNodeConnection]) throws {
        lock.lock()
        defer { lock.unlock() }
        
        logger.info("Adding node to processing chain: id=\(node.id), name=\(node.name)")
        
        // Check for duplicate node IDs
        if nodes[node.id] != nil {
            logger.error("Node with ID \(node.id) already exists in the processing chain")
            throw AudioPipelineError.invalidConfiguration
        }
        
        // Validate connections
        for connection in connections {
            // Ensure the source node exists (except for input connections where the source node is external)
            if connection.sourceNodeID != UUID.init(uuidString: "00000000-0000-0000-0000-000000000000")! && nodes[connection.sourceNodeID] == nil {
                logger.error("Connection references non-existent source node: \(connection.sourceNodeID)")
                throw AudioPipelineError.invalidConfiguration
            }
            
            // Ensure the destination node exists (except for output connections where the destination is external)
            if connection.destinationNodeID != UUID.init(uuidString: "00000000-0000-0000-0000-000000000001")! && connection.destinationNodeID != node.id {
                logger.error("Connection to an unregistered destination node: \(connection.destinationNodeID)")
                throw AudioPipelineError.invalidConfiguration
            }
        }
        
        // Add the node to our map
        nodes[node.id] = node
        
        // Store the connections
        connections[node.id] = connections
        
        logger.info("Node added successfully: id=\(node.id)")
    }
    
    /// Remove a node from the processing chain
    /// - Parameter nodeID: The ID of the node to remove
    public func removeNode(nodeID: UUID) throws {
        lock.lock()
        defer { lock.unlock() }
        
        logger.info("Removing node from processing chain: id=\(nodeID)")
        
        // Ensure the node exists
        guard nodes[nodeID] != nil else {
            logger.error("Attempted to remove non-existent node: \(nodeID)")
            throw AudioPipelineError.invalidConfiguration
        }
        
        // Check if any other nodes depend on this one
        for (_, nodeConnections) in connections {
            for connection in node
