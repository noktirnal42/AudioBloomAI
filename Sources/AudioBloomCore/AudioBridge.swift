import Foundation
import AVFoundation
import Combine
import Accelerate
import CoreAudio
import os.log

// MARK: - Errors

/// Errors that can occur during audio bridge operations
public enum AudioBridgeError: Error, CustomStringConvertible {
    /// Failed to convert audio data
    case dataConversionFailed
    
    /// Failed to process audio data
    case processingFailed
    
    /// Failed to connect to audio provider
    case connectionFailed
    
    /// Required resources not available
    case resourcesUnavailable
    
    /// Invalid state for operation
    case invalidState
    
    /// Human-readable error description
    public var description: String {
        switch self {
        case .dataConversionFailed:
/// Bridge for connecting the AudioEngine to the ML processing pipeline
public class AudioBridge: ObservableObject {
    // MARK: - Published Properties
    /// Logger for diagnostic information
    private let logger = Logger(subsystem: "com.audiobloom", category: "AudioBridge")
    
    // MARK: - Initialization
    
    /// Initializes the bridge with an ML processor
    /// - Parameter mlProcessor: The ML processor to use for analysis
    public init(mlProcessor: MLProcessorProtocol) {
        self.mlProcessor = mlProcessor
        setupMLSubscription()
    }
    
    deinit {
        // Clean up subscriptions
        audioDataSubscription?.cancel()
        mlVisualizationSubscription?.cancel()
        
        // Ensure disconnection
        disconnect()
    }
    
    // MARK: - Public Methods
    
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
    
    // MARK: - Private Methods
    
    /// Processes audio data received from the provider
    /// - Parameter audioData: The audio data to process
    private func processAudioData(_ audioData: AudioData) {
        // Skip processing if not active
        guard connectionState == .active, let mlProcessor = mlProcessor else {
            return
        }
        
        // Track performance
        performanceTracker.beginProcessing()
        
        // Convert audio data to the format needed by the ML processor
        do {
            let processableData = try formatConverter.convertForProcessing(audioData)
            
            // Process the data
            Task {
                do {
                    // Process data through ML processor
                    try await mlProcessor.processAudioData(processableData)
                    
                    // Track performance
                    performanceTracker.endProcessing()
                    
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
            name: .audioBridgeStateChanged,
            object: self,
            userInfo: ["state": ConnectionState.inactive]
        )
    }
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
    
    // MARK: - Private Methods
    
    /// Processes audio data received from the provider
    /// - Parameter audioData: The audio data to process
    private func processAudioData(_ audioData: AudioData) {
        // Skip processing if not active
        guard connectionState == .active, let mlProcessor = mlProcessor else {
            return
        }
        
        // Track performance
        performanceTracker.beginProcessing()
        
        // Convert audio data to the format needed by the ML processor
        do {
            let processableData = try formatConverter.convertForProcessing(audioData)
            
            // Process the data
            Task {
                do {
                    // Process data through ML processor
                    try await mlProcessor.processAudioData(processableData)
                    
                    // Track performance
                    performanceTracker.endProcessing()
                    
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
    
        
        // Update performance tracking
        performanceTracker.recordError()
        
        // Log detailed error if it's a bridge error
        if let bridgeError = error as? AudioBridgeError {
            logger.error("Bridge error: \(bridgeError.description)")
        }
        
        // Notify of error
        NotificationCenter.default.post(
            name: .audioBridgeError,
            object: self,
            userInfo: ["error": error]
        )
        
        // Attempt to continue despite errors
    }
    
    /// Sets up subscription to ML processor visualization data
    private func setupMLSubscription() {
        // First, cancel any existing subscription
        mlVisualizationSubscription?.cancel()
        
        // Create new subscription if we have a processor
        if let mlProcessor = mlProcessor {
            mlVisualizationSubscription = mlProcessor.visualizationDataPublisher
                .receive(on: processingQueue)
                .sink { [weak self] visualizationData in
                    // Forward the visualization data
                    self?.visualizationSubject.send(visualizationData)
                    
                    // Track performance for significant events
                    if visualizationData.isSignificantEvent {
                        self?.performanceTracker.recordSignificantEvent()
                    }
                }
        }
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
    
    // MARK: - Initialization
    
    /// Initializes the bridge with an ML processor
    /// - Parameter mlProcessor: The ML processor to use for analysis
    public init(mlProcessor: MLProcessorProtocol) {
        self.mlProcessor = mlProcessor
        setupMLSubscription()
    }
    
    deinit {
        // Clean up subscriptions
        audioDataSubscription?.cancel()
        mlVisualizationSubscription?.cancel()
        
        // Ensure disconnection
        disconnect()
    }
    
    // MARK: - Public Methods
    
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
        for i in 0..<outputCount {
            let startIdx = Int(Double(i) * stride)
            let endIdx = min(Int(Double(i + 1) * stride), inputCount)
            
            // Calculate average for this segment
            var sum: Float = 0
            var count = 0
            
            for j in startIdx..<endIdx {
                sum += inputData[j]
                count += 1
            }
            
            outputData[i] = count > 0 ? sum / Float(count) : 0
        }
    }
    
    /// vDSP-based resampling for optimal performance
    /// - Parameters:
    ///   - inputData: Source data array
    ///   - outputData: Target data array (pre-allocated)
    ///   - upsample: Whether to upsample (true) or downsample (false)
    /// - Throws: Error if resampling fails
    private func vDSP_resample(inputData: [Float], outputData: inout [Float], upsample: Bool) throws {
        let inputCount = vDSP_Length(inputData.count)
        let outputCount = vDSP_Length(outputData.count)
        
        if upsample {
            // For upsampling, use vDSP_vgenp which handles non-integer stride
            vDSP_vgenp(
                inputData,
                1,
                &outputData,
                1,
                outputCount,
                inputCount
            )
        } else {
            // For downsampling, use a filter and decimate approach to avoid aliasing
            
            // Create temporary buffer for filtered data
            var tempBuffer = [Float](repeating: 0, count: inputData.count)
            
            // Apply low-pass filter first to avoid aliasing
            let filterSize = 5
            var filter = [Float](repeating: 1.0 / Float(filterSize), count: filterSize)
            
            // Convolve with the filter
            vDSP_conv(
                inputData,
                1,
                filter,
                1,
                &tempBuffer,
                1,
                vDSP_Length(inputData.count - filterSize + 1),
                vDSP_Length(filterSize)
            )
            
            // Then decimate with stride
            let stride = Float(inputCount) / Float(outputCount)
            var currentIndex: Float = 0
            
            for i in 0..<Int(outputCount) {
                let idx = min(Int(currentIndex), inputData.count - 1)
                outputData[i] = tempBuffer[idx]
                currentIndex += stride
            }
        }
    }
    
            if normalizedFreq > 0.1 && normalizedFreq < 0.8 {
                // Bell curve emphasis for mid-range (where most important features are)
                let emphasis = 1.0 + 0.5 * sin(Float.pi * (normalizedFreq - 0.1) / 0.7)
                data[i] *= emphasis
            }
        }
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
    
    /// Performs FFT on raw audio samples to get frequency data
    /// - Parameter samples: Raw audio samples
    /// - Returns: Frequency spectrum data
    /// - Throws: Error if FFT fails
    func performFFT(samples: [Float]) throws -> [Float] {
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
        vDSP_vsmul(magnitude, 1, &scale, &
        let processingTime = endTime - processingStartTime
        
        // Add to running totals
        frameCount += 1
        totalProcessingTime += processingTime
        
        // Add to recent times circular buffer
        if recentProcessingTimes.count >= maxRecentTimes {
            recentProcessingTimes.removeFirst()
        }
        recentProcessingTimes.append(processingTime)
    }
    
    /// Records a significant event
    func recordSignificantEvent() {
        significantEventCount += 1
    }
    
    /// Records an error
    func recordError() {
        errorCount += 1
    }
    
    /// Gets the current performance metrics
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
        
        // Calculate conversion efficiency (placeholder - would be based on actual timing in a real impl)
        metrics.conversionEfficiency = max(0, min(1, 1.0 - (metrics.averageProcessingTime / 16.0))) // Target is sub-16ms
        
        return metrics
    }
}

private class FormatConverter {
    /// Standard frequency data size for ML processing
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

