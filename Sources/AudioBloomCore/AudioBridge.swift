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
    // MARK: - Initialization
    
    /// Initializes the bridge with an ML processor
    /// - Parameter mlProcessor: The ML processor to use for analysis
    public init(mlProcessor: MLProcessorProtocol) {
        self.mlProcessor = mlProcessor
        self.formatConverter = FormatConverter()
        self.initializeFFT()
        self.setupMLSubscription()
        self.performanceTracker.reset()
    }
    
    /// Initialize FFT components
    private func initializeFFT() {
        // Create FFT setup
            // Add to running totals
        frameCount += 1
        totalProcessingTime += time
        
        // Add to recent times circular buffer
        if recentProcessingTimes.count >= maxRecentTimes {
            recentProcessingTimes.removeFirst()
        }
        recentProcessingTimes.append(time)
        
        // Also record in performance tracker
        performanceTracker.recordProcessingTime(time)
    }
    /// The audio data provider
    private weak var audioProvider: AudioDataProvider?
    
    /// Subscription to audio data
    private var audioDataSubscription: AnyCancellable?
    
    /// Processing queue
    private let processingQueue = DispatchQueue(label: "com.audiobloom.bridge.processing", qos: .userInteractive)
    
    /// Current connection state
    private var connectionState: ConnectionState = .disconnected
    
    /// Audio processing queue
    private let audioQueue = DispatchQueue(label: "com.audiobloom.bridge.audio", qos: .userInteractive)
    
    /// Audio ML processor for AI analysis
    private let mlProcessor: MLProcessorProtocol
    
    /// Performance tracker for monitoring
    private let performanceTracker = PerformanceTracker()
    
    /// Publisher for visualization data
    private let visualizationSubject = PassthroughSubject<VisualizationData, Never>()
    
    /// FFT setup for audio processing
    private var fftSetup: OpaquePointer?
    
    /// Window buffer for audio processing
    private var windowBuffer: [Float]?
    
    /// Standard FFT size for audio processing
    private let fftSize = 1024
    
    /// Standard frequency size for visualization
    private let standardFrequencySize = 128
    
    /// Lock for thread safety
    private let lock = NSLock()
    
    /// Input format for audio processing
    private var inputFormat = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 2)!
    
    /// Performance tracking start time
    private var trackingStartTime = CFAbsoluteTimeGetCurrent()
    
    /// Frame count for performance tracking
    private var frameCount: UInt = 0
    
    /// Total processing time for tracking
    private var totalProcessingTime: Double = 0
    
    /// Maximum number of recent times to track
    private let maxRecentTimes = 30
    
    /// Recent processing times for averaging
    private var recentProcessingTimes: [Double] = []
    
    /// Event count for performance tracking
    private var significantEventCount: UInt = 0
    
    /// Error count for performance tracking
    private var errorCount: UInt = 0
    
    /// Recent audio data for visualization
    private var recentAudioData: AudioData?
    
    /// Recent visualization data
    private var recentVisualizationData = [Float](repeating: 0, count: 128)
    
    /// Format converter for audio processing
    private let formatConverter: FormatConverter
    
    /// Publisher for visualization data
    public var visualizationPublisher: AnyPublisher<VisualizationData, Never> {
        return visualizationSubject.eraseToAnyPublisher()
    }
    
    // MARK: - Initialization
    /// Initializes the bridge with an ML processor
    /// - Parameter mlProcessor: The ML processor to use for analysis
    public init(mlProcessor: MLProcessorProtocol) {
        self.mlProcessor = mlProcessor
        self.formatConverter = FormatConverter()
        self.initializeFFT()
        self.setupMLSubscription()
        self.performanceTracker.reset()
    }
    
    /// Initialize FFT components
    private func initializeFFT() {
        // Create FFT setup
        fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.FORWARD
        )
        
        // Create Hann window for better frequency resolution
        windowBuffer = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(windowBuffer!, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
    }
    
    /// Setup subscription to ML processor
    private func setupMLSubscription() {
        // Implementation would go here
    }
    
    deinit {
        // Clean up subscriptions
        audioDataSubscription?.cancel()
        
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
        
        // Notify of state change
        NotificationCenter.default.post(
            name: .audioBridgeStateChanged,
            object: self,
            userInfo: ["state": newState]
        )
    }
    
    /// Records processing time for performance tracking
    private func recordProcessingTime(_
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
        
        Task {
            do {
                if newState == .active {
                    // Process code when connection becomes active
                }
            } catch {
                handleProcessingError(error)
            }
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
        
        // Also record in performance tracker
        performanceTracker.recordProcessingTime(time)
    }
            userInfo: ["metrics": metrics]
        )
    }
    
    /// Process audio data from the provider
    private func processAudioData(_ audioData: AudioData) {
        let processingStart = CFAbsoluteTimeGetCurrent()
        
        do {
            // Store recent audio data
            recentAudioData = audioData
            
            // Prepare data for ML processing if needed
            let processableData = try formatConverter.convertForProcessing(audioData)
            
            // Process with ML
            Task {
                do {
                    try await mlProcessor.processAudioData(processableData)
                    
                    // Create visualization data
                    let isSignificantEvent = false
                    if isSignificantEvent {
                        performanceTracker.recordSignificantEvent()
                    }
                    
                    let visualizationData = VisualizationData(
                        values: audioData.frequencyData,
                        isSignificantEvent: isSignificantEvent
                    )
                    // Record processing time
                    let processingTime = CFAbsoluteTimeGetCurrent() - processingStart
                    recordProcessingTime(processingTime)
                    
                    // Periodically update metrics
                    if frameCount % 30 == 0 {
                        updatePerformanceMetrics()
                    }
                } catch {
                    handleProcessingError(error)
                }
            }
        } catch {
            handleProcessingError(error)
        }
    }
    
    /// Gets the current performance metrics
    private func getCurrentMetrics() -> PerformanceMetrics {
        // Get metrics from performance tracker
        return performanceTracker.getCurrentMetrics()
    }
}
    /// Connects to an audio data provider
    /// - Parameter provider: The audio data provider
    public func connect(to provider: AudioDataProvider) {
        // Don't reconnect if already connected to this provider
        if audioProvider === provider && connectionState != .disconnected {
// MARK: - AudioBridge Extensions

extension AudioBridge {
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
        
        // Reset internal tracking as well
        frameCount = 0
        totalProcessingTime = 0
        recentProcessingTimes = []
        significantEventCount = 0
        errorCount = 0
        trackingStartTime = CFAbsoluteTimeGetCurrent()
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
// MARK: - Format Converter

/// Format converter for audio processing
class FormatConverter {
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

/// Protocol for audio data publishers
public protocol AudioDataPublisher: AnyObject, Sendable {
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
    public enum OptimizationLevel: Sendable {
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

