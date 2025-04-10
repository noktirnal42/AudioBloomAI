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
    // MARK: - Initialization
    
    // MARK: - Private Methods
    
    /// Updates the connection state and notifies listeners
    private func updateConnectionState(_ newState: ConnectionState) {
        // In an actor, we don't need explicit locking for internal state
        connectionState = newState
        
        // Notify observers of state change
        Task { @MainActor in
            NotificationCenter.default.post(
                name: .audioBridgeStateChanged,
                object: self,
                userInfo: ["state": newState]
            )
        }
        
        // Update metrics if appropriate
        if newState == .active {
            updatePerformanceMetrics()
        }
    }
        
        // Ensure dis    /// Subject for visualization data
    private let visualizationSubject = PassthroughSubject<VisualizationData, Never>()
    
    /// Publisher for visualization data
    public nonisolated var visualizationPublisher: AnyPublisher<VisualizationData, Never> {
        return visualizationSubject.eraseToAnyPublisher()
    }
    
    /// Current performance metrics
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
    private func processAudioData(_ audioData: AudioData) async {
        // Skip processing if not active
        guard connectionState == .active else { return }
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
        self.windowBuffer = [Float](repeating: 0, count: fftSize)
        var window = self.windowBuffer
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        self.windowBuffer = window
    }
    
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
    
    /// Connects the bridge to an audio provider and starts processing
    /// - Parameter provider: The audio data provider
    public func connect(to provider: AudioDataProvider) async {
        guard connectionState != .connected && connectionState != .active else {
            logger.warning("Already connected to audio provider")
            return
        }
        
        logger.info("Connecting to audio provider")
        updateConnectionState(.connecting)
        
        // Store provider reference
        audioProvider = provider
        
        // Subscribe to audio data
        audioDataSubscription = provider.audioDataPublisher
            .sink { [weak self] audioData in
                guard let self = self else { return }
                Task {
                    await self.processAudioData(audioData)
                }
            }
        
        updateConnectionState(.connected)
        
        // Activate immediately
        await activate()
    }
    
    /// Disconnects the bridge from the audio provider
    public func disconnect() async {
        guard connectionState != .disconnected else {
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
