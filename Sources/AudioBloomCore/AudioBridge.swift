import Foundation
import AVFoundation
import Accelerate
import Combine
import CoreAudio
import CoreML
import QuartzCore
import MetalKit
import Logging

/// Bridge connecting audio data providers to ML processors
public final class AudioBridge: @unchecked Sendable {
    // MARK: - Types
    
    /// Connection state of the bridge
    public enum ConnectionState: String {
        case disconnected
        case connecting
        case connected
        case active
        case inactive
        case error
    }
    
    /// Errors that can occur in the audio bridge
    public enum AudioBridgeError: Error, CustomStringConvertible {
        case dataConversionFailed
        case connectionFailed(String)
        case streamingFailed(String)
        case processingFailed(String)
        
        public var description: String {
            switch self {
            case .dataConversionFailed: 
                return "Failed to convert audio data"
            case .connectionFailed(let details):
                return "Connection failed: \(details)"
            case .streamingFailed(let details):
                return "Streaming failed: \(details)"
            case .processingFailed(let details):
                return "Processing failed: \(details)"
            }
        }
    }
    
    /// Performance metrics for the audio bridge
    public struct PerformanceMetrics: Sendable {
        public var framesPerSecond: Double = 0
        public var eventsPerMinute: Double = 0
        public var errorRate: Double = 0
        public var averageProcessingTime: Double = 0
        public var conversionEfficiency: Double = 0
        
        public init() {}
    }
    
    // MARK: - Properties
    
    /// Logger instance
    private let logger = Logger(label: "com.audiobloom.bridge")
    /// The audio data provider
    private weak var audioProvider: AudioDataProvider?
    
    /// Subscription to audio data
    private var audioDataSubscription: AnyCancellable?
    
    /// Subscription to ML visualization data
    private var mlVisualizationSubscription: AnyCancellable?
    
    /// Current connection state
    private var connectionState: ConnectionState = .disconnected
    
    /// Audio processing queue
    private let processingQueue = DispatchQueue(label: "com.audiobloom.bridge", qos: .userInteractive)
    
    /// Format converter for audio processing
    private let formatConverter: FormatConverter
    
    /// Audio ML processor for AI analysis
    private let mlProcessor: MLProcessorProtocol
    
    /// Publisher for visualization data
    private let visualizationSubject = PassthroughSubject<VisualizationData, Never>()
    
    /// Thread safety lock
    private let lock = NSLock()
    
    // MARK: - FFT Properties
    
    /// FFT setup for frequency analysis
    private var fftSetup: OpaquePointer?
    
    /// Window buffer for FFT processing
    private var windowBuffer = [Float]()
    
    /// FFT size for processing (power of 2)
    private let fftSize = 1024
    
    // MARK: - Performance Tracking Properties
    
    /// Number of frames processed
    private var frameCount: Int = 0
    
    /// Total processing time (seconds)
    private var totalProcessingTime: Double = 0
    
    /// Recent processing times (seconds)
    private var recentProcessingTimes: [Double] = []
    
    /// Maximum number of recent times to track
    private let maxRecentTimes = 30
    
    /// Significant events detected
    private var significantEventCount: Int = 0
    
    /// Errors encountered
    private var errorCount: Int = 0
    
    /// Tracking start time
    private var trackingStartTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
    
    // MARK: - Public Properties
    
    /// Publisher for visualization data
    public var visualizationPublisher: AnyPublisher<VisualizationData, Never> {
        return visualizatio    
    // MARK: - Initialization
    
    /// Initializes the bridge with an ML processor
    /// - Parameter mlProcessor: The ML processor to use for analysis
    public init(mlProcessor: MLProcessorProtocol) {
        self.mlProcessor = mlProcessor
        self.formatConverter = FormatConverter()
        self.initializeFFT()
        self.setupMLSubscription()
    }
    
    /// Initialize FFT components
    private func initializeFFT() {
        // Create FFT setup
        fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.FORWARD
        )
        
        // Create window buffer
        windowBuffer = [Float](repeating: 0, count: fftSize)
        var window = windowBuffer
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        windowBuffer = window
    }
    
    deinit {
        // Clean up subscriptions
        audioDataSubscription?.cancel()
        mlVisualizationSubscription?.cancel()
        
        // Clean up FFT resources
        if let fftSetup = fftSetup {
            vDSP_DFT_DestroySetup(fftSetup)
        }
        
        // Ensure disconnection
        self.disconnect()
    }
    
    // MARK: - Private Methods
    
    /// Updates the connection state and notifies listeners
    private func updateConnectionState(_ newState: ConnectionState) {
        lock.lock()
        connectionState = newState
        lock.unlock()
                    // Update metrics
                    updatePerformanceMetrics()
                } catch {
                    handleProcessingError(error)
                }
            }
        } catch {
            handleProcessingError(error)
        }
    }
    
    /// Records processing time for performance tracking
    private func recordProcessingTime(_ time: Double) {
        // Add to running totals
        frameCount += 1
        totalProcessingTime += time
        
        // Add to recent times circular buffer
        if recentProcessingTimes.count >= maxRecentTimes {
            recentProcessingTimes.removeFirst()
        }
        recentProcessingTimes.append(time)
    }
    
    /// Handles errors that occur during processing
    private func handleProcessingError(_ error: Error) {
        // Update performance tracking
        performanceTracker.recordError()
        errorCount += 1
        
        // Log detailed error if it's a bridge error
        if let bridgeError = error as? AudioBridgeError {
            logger.error("Bridge error: \(bridgeError.description)")
        } else {
            logger.error("Processing error: \(error.localizedDescription)")
        }
        
        // Notify of error
        NotificationCenter.default.post(
            name: .audioBridgeError,
            object: self,
            userInfo: ["error": error]
        )
    }
    
    /// Updates performance metrics based on recent processing
    private func updatePerformanceMetrics() {
        // Get current metrics
        let metrics = getCurrentMetrics()
        
        // Notify observers
        NotificationCenter.default.post(
            name: .audioBridgePerformanceUpdate,
            object: self,
            userInfo: ["metrics": metrics]
        )
    }
    
    /// Gets the current performance metrics
    private func getCurrentMetrics() -> PerformanceMetrics {
        let currentTime = CFAbsoluteTimeGetCurrent()
        let elapsedMinutes = (currentTime - trackingStartTime) / 60.0
        
        var metrics = PerformanceMetrics()
        
        // Calculate frames per second
        if elapsedMinutes > 0 {
            metrics.framesPerSecond = Double(frameCount) / (elapsedMinutes * 60.0)
            metrics.eventsPerMinute = Double(significantEventCount) / elapsedMinutes
            metrics.errorRate = Double(errorCount) / elapsedMinutes
        }
        
        // Calculate average processing time
        if !recentProcessingTimes.isEmpty {
            let recentTotal = recentProcessingTimes.reduce(0, +)
            metrics.averageProcessingTime = (recentTotal / Double(recentProcessingTimes.count)) * 1000 // Convert to ms
        } else if frameCount > 0 {
            metrics.averageProcessingTime = (totalProcessingTime / Double(frameCount)) * 1000 // Convert to ms
        }
        
        // Calculate conversion efficiency
        metrics.conversionEfficiency = max(0, min(1, 1.0 - (metrics.averageProcessingTime / 16.0))) // Target is sub-16ms
        
        return metrics
    }
// MARK: - Public Methods

extension AudioBridge {
    /// Connects to an audio data provider
    /// - Parameter provider: The audio data provider
    public func connect(to provider: AudioDataProvider) {
        // Don't reconnect if already connected to this provider
        if audioProvider === provider && connectionState != .disconnected {
            return
        }
        
        // Disconnect from any current provider
        disconnect()
        
        logger.info("Connecting to audio provider")
        
        // Update state
        updateConnectionState(.connecting)
        
        // Store provider reference
        audioProvider = provider
        
        // Subscribe to audio data
        audioDataSubscription = provider.audioDataPublisher
            .receive(on: processingQueue)
            .sink { [weak self] audioData in
                self?.processAudioData(audioData)
            }
        
        // Update state
        updateConnectionState(.connected)
        
        logger.info("Connected to audio provider")
    }
    
    /// Disconnects from the current audio provider
    public func disconnect() {
        guard connectionState != .disconnected else { return }
        
        logger.info("Disconnecting from audio provider")
        
        // Cancel subscriptions
        audioDataSubscription?.cancel()
        audioDataSubscription = nil
        
        // Clear provider reference
        audioProvider = nil
        
        // Update state
        updateConnectionState(.disconnected)
    }
    
    /// Activates the bridge to start processing
    public func activate() {
        guard connectionState == .connected || connectionState == .inactive else {
            logger.warning("Cannot activate: bridge is not in a connected or inactive state")
            return
        }
        
        logger.info("Activating audio bridge")
        
        // Update state
        updateConnectionState(.active)
        
        // Reset performance tracker
        performanceTracker.reset()
        
        // Notify of activation
        NotificationCenter.default.post(
            name: .audioBridgeStateChanged,
            object: self,
            userInfo: ["state": ConnectionState.active]
        )
    }
    
    /// Deactivates the bridge to pause processing
    public func deactivate() {
        guard connectionState == .active else {
            logger.warning("Cannot deactivate: bridge is not active")
            return
        }
        
        logger.info("Deactivating audio bridge")
        
        // Update state
        updateConnectionState(.inactive)
        
        // Notify of deactivation
        NotificationCenter.default.post(
            name: .audioBridgeStateChanged,
            object: self,
            userInfo: ["state": ConnectionState.inactive]
        )
    }
    
    /// Performs FFT on raw audio samples to get frequency data
    /// - Parameter samples: Raw audio samples
    /// - Returns: Frequency spectrum data
    /// - Throws: Error if FFT fails
    public func performFFT(samples: [Float]) throws -> [Float] {
        guard let fftSetup = fftSetup,
              let window = windowBuffer,
              samples.count > 0 else {
            throw AudioBridgeError.dataConversionFailed
        }
        
        let count = min(samples.count, fftSize)
        
        // Split complex setup
        var realInput = [Float](repeating: 0, count: fftSize)
        var imagInput = [Float](repeating: 0, count: fftSize)
        var realOutput = [Float](repeating: 0, count: fftSize)
        var imagOutput = [Float](repeating: 0, count: fftSize)
        
        // Apply window to input data
        for i in 0..<count {
            realInput[i] = samples[i] * window[i]
        }
        
        // Perform FFT
        vDSP_DFT_Execute(
            fftSetup,
            realInput, imagInput,
            &realOutput, &imagOutput
        )
        
        // Calculate magnitude
        var magnitude = [Float](repeating: 0, count: fftSize/2)
        for i in 0..<fftSize/2 {
            let real = realOutput[i]
            let imag = imagOutput[i]
            magnitude[i] = sqrt(real * real + imag * imag)
        }
        
        // Scale the magnitudes
        var scale = Float(1.0 / Float(fftSize))
        vDSP_vsmul(magnitude, 1, &scale, &magnitude, 1, vDSP_Length(fftSize/2))
        
        return magnitude
    }
}

// MARK: - Supporting Types

extension AudioBridge {
    /// Connection state of the bridge
    public enum ConnectionState: String {
        /// Bridge is disconnected from audio provider
        case disconnected
        
        /// Bridge is in the process of connecting
        case connecting
        
        /// Bridge is connected but inactive (not processing)
        case connected
        
        /// Bridge is inactive (connected but paused)
        case inactive
        
        /// Bridge is active and processing audio
        case active
    }
    
    /// Error types that can occur in the bridge
    public enum AudioBridgeError: Error, CustomStringConvertible {
        /// Failed to connect to audio provider
        case connectionFailed
        
        /// Failed to convert audio data
        case dataConversionFailed
        
        /// Failed to process audio data
        case processingFailed
        
        /// Description of the error
        public var description: String {
            switch self {
            case .connectionFailed:
                return "Failed to connect to audio provider"
            case .dataConversionFailed:
                return "Failed to convert audio data"
            case .processingFailed:
                return "Failed to process audio data"
            }
        }
    }
    
    /// Performance metrics for monitoring
    public struct PerformanceMetrics {
        /// Frames processed per second
        public var framesPerSecond: Double = 0
        
        /// Average processing time per frame (ms)
        public var averageProcessingTime: Double = 0
        
        /// Events detected per minute
        public var eventsPerMinute: Double = 0
        
        /// Errors per minute
        public var errorRate: Double = 0
        
        /// Efficiency of audio conversion (0-1)
        /// Efficiency of audio conversion (0-1)
        public var conversionEfficiency: Double = 1.0
    }
}

/// Performance tracking for audio processing
class PerformanceTracker {
    /// Number of frames processed
    private var frameCount: Int = 0
    
    /// Total processing time (seconds)
    private var totalProcessingTime: Double = 0
    
    /// Recent processing times (seconds)
    private var recentProcessingTimes: [Double] = []
    
    /// Maximum number of recent times to track
    private let maxRecentTimes = 30
    
    /// Significant events detected
    private var significantEventCount: Int = 0
    
    /// Errors encountered
    private var errorCount: Int = 0
    
    /// Tracking start time
    private var trackingStartTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
    
    /// Indicates the start of a processing operation
    func beginProcessing() {
        // Implementation in real code would track start time
    }
    
    /// Indicates the end of a processing operation
    func endProcessing() {
        // Implementation in real code would calculate elapsed time
    }
    
    /// Resets all tracking counters
    func reset() {
        frameCount = 0
        totalProcessingTime = 0
        recentProcessingTimes = []
        significantEventCount = 0
        errorCount = 0
        trackingStartTime = CFAbsoluteTimeGetCurrent()
    }
    
    /// Records a significant event
    func recordSignificantEvent() {
        significantEventCount += 1
    }
    
    /// Records an error
    func recordError() {
        errorCount += 1
    }
    
    /// Records processing time
    /// - Parameter time: Processing time in seconds
    func recordProcessingTime(_ time: Double) {
        // Add to running totals
        frameCount += 1
        totalProcessingTime += time
        
        // Add to recent times circular buffer
        if recentProcessingTimes.count >= maxRecentTimes {
            recentProcessingTimes.removeFirst()
        }
        recentProcessingTimes.append(time)
    }
    
    /// Gets the current performance metrics
    /// - Returns: Performance metrics
    func getCurrentMetrics() -> AudioBridge.PerformanceMetrics {
        let currentTime = CFAbsoluteTimeGetCurrent()
        let elapsedMinutes = (currentTime - trackingStartTime) / 60.0
        
        var metrics = AudioBridge.PerformanceMetrics()
        
        // Calculate frames per second
        if elapsedMinutes > 0 {
            metrics.framesPerSecond = Double(frameCount) / (elapsedMinutes * 60.0)
            metrics.eventsPerMinute = Double(significantEventCount) / elapsedMinutes
            metrics.errorRate = Double(errorCount) / elapsedMinutes
        }
        
        // Calculate average processing time
        if !recentProcessingTimes.isEmpty {
            let recentTotal = recentProcessingTimes.reduce(0, +)
            metrics.averageProcessingTime = (recentTotal / Double(recentProcessingTimes.count)) * 1000 // Convert to ms
        } else if frameCount > 0 {
            metrics.averageProcessingTime = (totalProcessingTime / Double(frameCount)) * 1000 // Convert to ms
        }
        
        // Calculate conversion efficiency
        metrics.conversionEfficiency = max(0, min(1, 1.0 - (metrics.averageProcessingTime / 16.0))) // Target is sub-16ms
        
        return metrics
    }
}

/// Format converter for audio processing
private class FormatConverter {
    private let standardFrequencySize = 512
    
    /// Converts audio data to ML-processable format
    /// - Parameter audioData: The audio data to convert
    /// - Returns: ML-processable audio data
    /// - Throws: Error if conversion fails
    func convertForProcessing(_ audioData: AudioData) throws -> [Float] {
        // Handle empty data case
        guard !audioData.frequencyData.isEmpty else {
            return createEmptyProcessableData()
        }
        
        // Normalize frequency data to standard size
        var processableData = normalizeFrequencyData(audioData.frequencyData)
        
        // Add amplitude information at the end
        // This helps the ML model correlate frequency patterns with overall loudness
        processableData.append(audioData.levels.left)
        processableData.append(audioData.levels.right)
        
        return processableData
    }
    
    /// Creates an empty data array for when no audio is available
    /// - Returns: Empty processable data
    private func createEmptyProcessableData() -> [Float] {
        var result = [Float](repeating: 0, count: standardFrequencySize)
        // Add zero amplitude
        result.append(0)
        result.append(0)
        return result
    }
    
    /// Normalizes frequency data to standard size using efficient algorithms
    /// - Parameter frequencyData: Original frequency data
    /// - Returns: Normalized data
    private func normalizeFrequencyData(_ frequencyData: [Float]) -> [Float] {
        let originalCount = frequencyData.count
        
        // If already the right size, return a copy
        if originalCount == standardFrequencySize {
            return frequencyData
        }
        
        // Create output buffer of standard size
        var result = [Float](repeating: 0, count: standardFrequencySize)
        
        // Use Accelerate framework for efficient resampling
        if originalCount < standardFrequencySize {
            // Upsample (interpolate)
            vDSP_vgenp(
                frequencyData,
                1,
                &result,
                1,
                vDSP_Length(standardFrequencySize),
                vDSP_Length(originalCount)
            )
        } else {
            // Downsample
            let stride = Double(originalCount) / Double(standardFrequencySize)
            
            for i in 0..<standardFrequencySize {
                let originalIndex = Int(Double(i) * stride)
                if originalIndex < originalCount {
                    result[i] = frequencyData[originalIndex]
                }
            }
            
            // Apply smoothing to avoid aliasing
            smoothData(&result)
        }
        
        // Normalize values to [0.0, 1.0] range for ML processing
        normalizeAmplitude(&result)
        
        return result
    }
    
    /// Smooths data using a simple moving average filter
    /// - Parameter data: Data to smooth (modified in place)
    private func smoothData(_ data: inout [Float]) {
        let windowSize = 3
        let halfWindow = windowSize / 2
        let count = data.count
        
        // Create a temporary buffer
        var smoothed = data
        
        // Apply moving average
        for i in halfWindow..<(count - halfWindow) {
            var sum: Float = 0
            for j in -halfWindow...halfWindow {
                sum += data[i + j]
            }
            smoothed[i] = sum / Float(windowSize)
        }
        
        // Copy smoothed data back
        data = smoothed
    }
    
    /// Normalizes amplitude values to [0.0, 1.0] range
    /// - Parameter data: Data to normalize (modified in place)
    private func normalizeAmplitude(_ data: inout [Float]) {
        // Find the min and max values
        var min: Float = 0
        var max: Float = 0
        
        vDSP_minv(data, 1, &min, vDSP_Length(data.count))
        vDSP_maxv(data, 1, &max, vDSP_Length(data.count))
        
        // Skip normalization if the range is too small
        let range = max - min
        if range < 1e-6 {
            // Just center around 0.5 if all values are the same
            var value: Float = 0.5
            vDSP_vfill(&value, &data, 1, vDSP_Length(data.count))
            return
        }
        
        // Apply normalization
        var a = Float(1.0 / range)
        var b = Float(-min * a)
        
        vDSP_vsmsa(data, 1, &a, &b, &data, 1, vDSP_Length(data.count))
    }
    
    /// Converts PCM buffer to frequency data using FFT
    /// - Parameter buffer: The PCM buffer to analyze
    /// - Returns: Frequency spectrum data
    /// - Throws: Error if conversion fails
    func convertBufferToFrequencyData(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        guard let channelData = buffer.floatChannelData,
              buffer.frameLength > 0 else {
            throw AudioBridgeError.dataConversionFailed
        }
        
        // Extract sample data from buffer
        let count = Int(buffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: count))
        
        // Prepare FFT
        let fftSize = 1024 // Use power of 2 for efficiency
        let fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.FORWARD
        )
        
        // Prepare input data (apply windowing)
        var windowedInput = [Float](repeating: 0, count: fftSize)
        var window = [Float](repeating: 0, count: fftSize)
        
        // Create Hann window
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        
        // Copy input data (zero-pad if needed)
        let sampleCount = min(count, fftSize)
        for i in 0..<sampleCount {
            windowedInput[i] = samples[i] * window[i]
        }
        
        // Prepare for FFT
        var realInput = windowedInput
        var imagInput = [Float](repeating: 0, count: fftSize)
        var realOutput = [Float](repeating: 0, count: fftSize)
        var imagOutput = [Float](repeating: 0, count: fftSize)
        
        // Perform FFT
        vDSP_DFT_Execute(
            fftSetup!,
            realInput, imagInput,
            &realOutput, &imagOutput
        )
        
        // Calculate magnitude
        var magnitude = [Float](repeating: 0, count: fftSize/2)
        for i in 0..<fftSize/2 {
            let real = realOutput[i]
            let imag = imagOutput[i]
            magnitude[i] = sqrt(real * real + imag * imag)
        }
        
        // Scale the magnitudes
        var scale = Float(1.0 / Float(fftSize))
        vDSP_vsmul(magnitude, 1, &scale, &magnitude, 1, vDSP_Length(fftSize/2))
        
        // Apply logarithmic scaling for better visualization
        var scaledMagnitude = [Float](repeating: 0, count: fftSize/2)
        for i in 0..<fftSize/2 {
            // Convert to dB with some scaling and clamping
            let logValue = 10.0 * log10f(magnitude[i] + 1e-9)
            // Normalize to 0.0-1.0 range
            let normalizedValue = (logValue + 90.0) / 90.0
            scaledMagnitude[i] = min(max(normalizedValue, 0.0), 1.0)
        }
        
        // Destroy FFT setup
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
        
        return scaledMagnitude
    }
    
    /// Creates a PCM buffer from processable data
    /// - Parameters:
    ///   - data: The processable data
    ///   - format: The audio format
    /// - Returns: A PCM buffer
    /// - Throws: Error if conversion fails
    func createBufferFromProcessableData(_ data: [Float], format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        // Extract audio data (excluding level information at the end)
        let audioSampleCount = data.count - 2
        
        guard audioSampleCount > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audioSampleCount)) else {
            throw AudioBridgeError.dataConversionFailed
        }
        
        // Set frame length
        buffer.frameLength = AVAudioFrameCount(audioSampleCount)
        
        // Copy data to buffer
        if let channelData = buffer.floatChannelData {
            for i in 0..<audioSampleCount {
                channelData[0][i] = data[i]
            }
        }
        
        return buffer
    }
}

// MARK: - Notification Extensions

extension Notification.Name {
    /// Notification posted when audio bridge state changes
    public static let audioBridgeStateChanged = Notification.Name("audioBridgeStateChanged")
    
    /// Notification posted when bridge encounters an error
    public static let audioBridgeError = Notification.Name("audioBridgeError")
    
    /// Notification posted when performance metrics are updated
    public static let audioBridgePerformanceUpdate = Notification.Name("audioBridgePerformanceUpdate")
}

// MARK: - Protocol Extensions

/// Protocol for objects that provide audio data
public protocol AudioDataProvider: AnyObject {
    /// Publisher for audio data updates
    var audioDataPublisher: AnyPublisher<AudioData, Never> { get }
}

/// Protocol for objects that can process audio data
public protocol MLProcessorProtocol: AnyObject {
    /// Whether the processor is ready for processing
    var isReady: Bool { get }
    
    /// Publisher for visualization data
    var visualizationDataPublisher: AnyPublisher<VisualizationData, Never> { get }
    
    /// Processes audio data
    /// - Parameter audioData: The audio data to process
    /// - Throws: Error if processing fails
    func processAudioData(_ audioData: [Float]) async throws
}

/// Visualization data structure
public struct VisualizationData {
    /// Values to visualize (typically frequency spectrum)
    public let values: [Float]
    
    /// Whether this data represents a significant event
    public let isSignificantEvent: Bool
    
    /// Timestamp of the data
    public let timestamp: Date
    
    /// Initializes a new visualization data instance
    /// - Parameters:
    ///   - values: Values to visualize
    ///   - isSignificantEvent: Whether this is a significant event
    ///   - timestamp: Timestamp of the data
    public init(values: [Float], isSignificantEvent: Bool = false, timestamp: Date = Date()) {
        self.values = values
        self.isSignificantEvent = isSignificantEvent
        self.timestamp = timestamp
    }
}

/// Audio data structure
public struct AudioData {
    /// Frequency spectrum data
    public let frequencyData: [Float]
    
    /// Audio level data (left and right channels)
    public let levels: (left: Float, right: Float)
    
    /// Timestamp of the data
    public let timestamp: Date
    
    /// Initializes a new audio data instance
    /// - Parameters:
    ///   - frequencyData: Frequency spectrum data
    ///   - levels: Audio level data
    ///   - timestamp: Timestamp of the data
    public init(frequencyData: [Float], levels: (left: Float, right: Float), timestamp: Date = Date()) {
        self.frequencyData = frequencyData
        self.levels = levels
        self.timestamp = timestamp
    }
}

/// Protocol for audio data publishers
public protocol AudioDataPublisher {
    /// Publisher for audio data
    var publisher: AnyPublisher<AudioData, Never> { get }
    
    /// Publishes new audio data
    /// - Parameters:
    ///   - frequencyData: The frequency spectrum data
    ///   - levels: The audio level data
    func publish(frequencyData: [Float], levels: (Float, Float))
}

// MARK: - AudioBridge Settings

/// Settings for audio bridge configuration
public struct AudioBridgeSettings {
    /// Whether to apply Neural Engine optimizations
    public var useNeuralEngine: Bool = true
    
    /// Optimization level for processing
    public var optimizationLevel: OptimizationLevel = .balanced
    
    /// Buffer size for audio processing
    public var bufferSize: Int = 1024
    
    /// Optimization level for audio processing
    public enum OptimizationLevel {
        /// Prioritize efficiency (lower power usage)
        case efficiency
        
        /// Balance efficiency and quality
        case balanced
        
        /// Prioritize quality (higher power usage)
        case quality
    }
    
    /// Initializes with default settings
    public init() {}
    
    /// Initializes with custom settings
    /// - Parameters:
    ///   - useNeuralEngine: Whether to use Neural Engine
    ///   - optimizationLevel: Processing optimization level
    ///   - bufferSize: Audio buffer size
    public init(useNeuralEngine: Bool, optimizationLevel: OptimizationLevel, bufferSize: Int) {
        self.useNeuralEngine = useNeuralEngine
        self.optimizationLevel = optimizationLevel
        self.bufferSize = bufferSize
    }
}

