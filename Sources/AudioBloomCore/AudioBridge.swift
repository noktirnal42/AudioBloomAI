import Foundation
import AVFoundation
import Accelerate
import Combine
import CoreAudio
import CoreML
import QuartzCore
import MetalKit
import Logging

// Import modular components from the same module
@_implementationOnly import struct AudioBloomCore.AudioBridgeSettings
@_implementationOnly import struct AudioBloomCore.OptimizationLevel
@_implementationOnly import struct AudioBloomCore.AudioData
@_implementationOnly import struct AudioBloomCore.VisualizationData
@_implementationOnly import protocol AudioBloomCore.MLProcessorProtocol

/// Bridge connecting audio data providers to ML processors
public final class AudioBridge: @unchecked Sendable {
    // MARK: - Types
    
    /// Connection state of the bridge
    public enum ConnectionState: String, Sendable {
        case disconnected
        case connecting
        case connected
        case inactive
        case active
    }
    
    /// Error types that can occur in the bridge
    public enum AudioBridgeError: Error, CustomStringConvertible, Sendable {
        case connectionFailed
        case dataConversionFailed
        case processingFailed
        
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
    
    /// Performance metrics for audio processing
    public struct PerformanceMetrics: Sendable {
        public var framesPerSecond: Double = 0
        public var eventsPerMinute: Double = 0
        public var averageLatency: Double = 0
        public var errorRate: Double = 0
        public var averageProcessingTime: Double = 0
        public var conversionEfficiency: Double = 1.0
    }
    
    // MARK: - Properties
    
    /// Bridge version
    public static let version = "1.2.0"
    
    /// Logger instance
    private let logger = Logger(label: "com.audiobloom.bridge")
    
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
    
    deinit {
        // Ensure disconnection
        self.disconnect()
    }
    
    /// Creates empty processable data
    /// - Returns: Empty data array with zeros
    private func createEmptyProcessableData() -> [Float] {
        var result = [Float](repeating: 0, count: standardFrequencySize)
        // Add zero amplitude
        result.append(0)
        result.append(0)
        return result
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
    
    // MARK: - Private Methods
    
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
    
    /// Handles a processing error
    private func handleProcessingError(_ error: Error) {
        // Update error counter
        errorCount += 1
        performanceTracker.recordError()
        
        // Log detailed error
        if let bridgeError = error as? AudioBridgeError {
            logger.error("Bridge error: \(bridgeError.description)")
        } else {
            logger.error("Processing error: \(error.localizedDescription)")
        }
        
        // Notify listeners of the error
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
                    
                    // Publish visualization data
                    visualizationSubject.send(visualizationData)
                    
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
        return performanceTracker.getCurrentMetrics()
    }
}
