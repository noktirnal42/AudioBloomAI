    /// Current performance metrics
    public private(set) var performanceMetrics = PerformanceMetrics()
    
    /// Performance tracker for monitoring
    private let performanceTracker = PerformanceTracker()
    
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
public actor AudioBridge: Sendable {
    // MARK: - Types
    
    /// Connection state of the bridge
    public enum ConnectionState: String, Sendable {
        case disconnected
        case connecting
        case connected
        case active
        case inactive
        case error
    }
    
    /// Errors that can occur in the audio bridge
    public enum AudioBridgeError: Error, CustomStringConvertible, Sendable {
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
        /// Frames processed per second
        public var framesPerSecond: Double = 0
        /// Events detected per minute
        public var eventsPerMinute: Double = 0
        /// Errors per minute
        public var errorRate: Double = 0
        /// Average processing time per frame (ms)
        public var averageProcessingTime: Double = 0
        /// Efficiency of audio conversion (0-1)
        public var conversionEfficiency: Double = 0
        
        public init() {}
    }
    public private(set) var performanceMetrics = PerformanceMetrics()
    
    /// Performance tracker for monitoring
    private let performanceTracker = PerformanceTracker()
    
    /// Publisher for visualization data
    public var visualizationPublisher: AnyPublisher<VisualizationData, Never> {
        return visualizationSubject.eraseToAnyPublisher()
    }
            await disconnect()
        }
    }
        // Notify of error
        Task { @MainActor in
            NotificationCenter.default.post(
                name: .audioBridgeError,
                object: self,
                userInfo: ["error": error]
            )
        }
    }
        // Notify observers
        Task { @MainActor in
            NotificationCenter.default.post(
                name: .audioBridgePerformanceUpdate,
                object: self,
                userInfo: ["metrics": metrics]
            )
        }
    }
        // Notify of activation
        Task { @MainActor in
            NotificationCenter.default.post(
                name: .audioBridgeStateChanged,
                object: self,
                userInfo: ["state": ConnectionState.active]
            )
        }
    }
    /// - Parameter audioData: The audio data to process
    private let formatConverter: FormatConverter
    
    /// Audio ML processor for AI analysis
    private let mlProcessor: MLProcessorProtocol
    
    /// Subject for visualization data
    private let visualizationSubject = PassthroughSubject<VisualizationData, Never>()
    
    /// Publisher for visualization data
    public nonisolated var visualizationPublisher: AnyPublisher<VisualizationData, Never> {
        return visualizationSubject.eraseToAnyPublisher()
    }
    
    /// Current performance metrics
    public private(set) var performanceMetrics = PerformanceMetrics()
    
    /// Performance tracker for monitoring
    private let performanceTracker = PerformanceTracker()
    
    // MARK: - Initialization
    // Called after initialization
    public func setupSubscriptions() async {
        // Subscribe to visualization data from ML processor
        mlVisualizationSubscription = mlProcessor.visualizationDataPublisher
            .sink { [weak self] visualizationData in
                guard let self = self else { return }
                
                // Forward visualization data to our publisher
                self.visualizationSubject.send(visualizationData)
                
                // Record event if significant
                if visualizationData.isSignificantEvent {
                    self.performanceTracker.recordSignificantEvent()
                }
            }
    }
                name: .audioBridgeStateChanged,
                object: self,
                userInfo: ["state": ConnectionState.inactive]
            )
        }
    }
}
        Task {
            await self.disconnect()
        }
    }
    
    // MARK: - Connection Methods
    
    /// Initializes the bridge with an ML processor
    /// - Parameter mlProcessor: The ML processor to use for analysis
    public init(mlProcessor: MLProcessorProtocol) {
        self.mlProcessor = mlProcessor
        self.formatConverter = FormatConverter()
        
        // Initialize FFT
        self.fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.FORWARD
        )
        
        // Create window buffer
        self.windowBuffer = [Float](repeating    deinit {
        // Clean up subscriptions
        audioDataSubscription?.cancel()
        mlVisualizationSubscription?.cancel()
        
        // Clean up FFT resources
        if let fftSetup = fftSetup {
            vDSP_DFT_DestroySetup(fftSetup)
        }
    }
    
    /// Set up ML processor subscription
    public func setupSubscriptions() async {
        // Subscribe to visualization data from ML processor
        mlVisualizationSubscription = mlProcessor.visualizationDataPublisher
            return
        }
        
        logger.info("Disconnecting audio bridge")
        
        // Cancel subscriptions
        audioDataSubscription?.cancel()
        audioDataSubscription = nil
        
        // Clear provider reference
        audioProvider = nil
        
        // Update state
        updateConnectionState(.disconnected)
    }
    
    /// Activates the bridge to start processing
    public func activate() async {
        guard connectionState == .connected || connectionState == .inactive else {
            logger.warning("Cannot activate: bridge is not connected or is already active")
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
    public func deactivate() async {
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
