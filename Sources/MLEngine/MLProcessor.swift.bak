import Foundation
import CoreML
import Combine
import AudioBloomCore
import AudioBloomCore.Audio
import AVFoundation
import SoundAnalysis
import Logging
import os.signpost
/// Errors that can occur during ML processing
public enum MLProcessorError: Error {
    case featureExtractionFailed(Error)
    case modelNotLoaded
    case processingFailed(Error)
    case audioFormatError
    case invalidConfiguration
    case neuralEngineError(String)
    
    var localizedDescription: String {
        switch self {
        case .featureExtractionFailed(let error):
            return "Feature extraction failed: \(error.localizedDescription)"
        case .modelNotLoaded:
            return "ML model is not loaded or ready"
        case .processingFailed(let error):
            return "Processing failed: \(error.localizedDescription)"
        case .audioFormatError:
            return "Audio format is incompatible with processing pipeline"
        case .invalidConfiguration:
            return "ML processor has invalid configuration"
        case .neuralEngineError(let message):
            return "Neural Engine error: \(message)"
        }
    }
}

/// Protocol for ML processing status updates
public protocol MLProcessingDelegate: AnyObject {
    /// Called when the processor's ready state changes
    func mlProcessorReadyStateChanged(isReady: Bool)
    
    /// Called when an error occurs during processing
    func mlProcessorDidEncounterError(_ error: Error)
    
    /// Called when performance metrics are updated
    func mlProcessorDidUpdateMetrics(_ metrics: MLProcessor.PerformanceMetrics)
}

/// ML Processor for analyzing audio data and generating visual effects

// Dummy implementation of AudioPipelineProtocol for backward compatibility
fileprivate class DummyAudioPipeline: AudioPipelineProtocol {
    func subscribe(bufferSize: Int, hopSize: Int, callback: @escaping (AudioBufferID, TimeInterval) -> Void) -> UUID {
        return UUID()
    }
    
    func unsubscribe(_ subscriptionID: UUID) {}
    
    func unsubscribeAll() {}
    
    func getBufferData(_ bufferID: AudioBufferID) -> [Float]? {
        return nil
    }
    
    func releaseBuffer(_ bufferID: AudioBufferID) {}
    
    var format: AVAudioFormat {
        return AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
    }
}

public class MLProcessor: ObservableObject, AudioFeatureExtractorDelegate, VisualizationDataReceiver, AudioDataConsumer {
    /// Published ML model output data for visualization
    @Published public private(set) var outputData: [Float] = []
    
    /// Published audio features from analysis
    @Published public private(set) var audioFeatures = AudioFeatures()
    
    /// Published performance metrics
    @Published public private(set) var performanceMetrics = PerformanceMetrics()
    
    /// Published processing state
    @Published public private(set) var state: ProcessingState = .inactive
    
    /// Audio feature extractor
    private var featureExtractor: AudioFeatureExtractor
    
    /// Model configuration
    private var modelConfig: ModelConfiguration
    
    /// Output transformer
    private var outputTransformer: OutputTransformer
    
    /// Core ML model
    private var mlModel: MLModel?
    
    /// Processing task for continuous processing
    private var processingTask: Task<Void, Never>?
    
    /// Logger for this class
    private let logger = Logger(label: "com.audiobloom.mlprocessor")
    
    /// Signpost log for performance tracking
    private let signposter = OSSignposter(subsystem: "com.audiobloom.mlprocessor", category: "Processing")
    
    /// Subject for publishing visualization data
    private let visualizationDataSubject = PassthroughSubject<VisualizationData, Never>()
    
    /// Publisher for visualization data
    public var visualizationDataPublisher: AnyPublisher<VisualizationData, Never> {
        visualizationDataSubject.eraseToAnyPublisher()
    }
    
    /// Delegate for ML processing status updates
    public weak var delegate: MLProcessingDelegate?
    
    /// Whether to use Neural Engine optimizations when available
    public var useNeuralEngine: Bool = true {
        didSet {
            if useNeuralEngine != oldValue {
                logger.info("Neural Engine usage \(useNeuralEngine ? "enabled" : "disabled")")
                configureModelForCurrentDevice()
            }
        }
    }
    
    /// The audio format used for processing
    private var audioFormat: AVAudioFormat?
    
    /// The optimization level for the processor
    public var optimizationLevel: OptimizationLevel = .balanced {
        didSet {
            if optimizationLevel != oldValue {
                logger.info("Optimization level changed to \(optimizationLevel)")
                modelConfig.applyOptimizationLevel(optimizationLevel)
            }
        }
    }
    
    /// Flag indicating if the ML pipeline is ready for processing
    public var isReady: Bool {
        return state == .ready || state == .processing
    }
    
    /// Processing state enum
    public enum ProcessingState {
        case inactive
        case preparing
        case ready
        case processing
        case error(Error)
    }
    
    /// Performance metrics for monitoring
    public struct PerformanceMetrics {
        /// Neural Engine utilization (0-1)
        public var neuralEngineUtilization: Double = 0
        
        /// Average processing time per frame (ms)
        public var averageProcessingTime: Double = 0
        
        /// CPU usage for processing (0-1)
        public var cpuUsage: Double = 0
        
        /// Memory usage in MB
        public var memoryUsage: Double = 0
        
        /// Current frames processed per second
        public var framesPerSecond: Double = 0
        
        /// Timestamp of last update
        public var lastUpdateTime: Date = Date()
    }
    
    /// Initializes a new MLProcessor with the specified configuration
    /// - Parameters:
    ///   - optimizationLevel: The optimization level to use
    ///   - useNeuralEngine: Whether to use Neural Engine optimizations
    public init(
        optimizationLevel: OptimizationLevel = .balanced,
        useNeuralEngine: Bool = true
    ) {
        self.optimizationLevel = optimizationLevel
        self.useNeuralEngine = useNeuralEngine
        
        // Initialize the components
        // Using default implementation since proper initialization would require AudioPipelineProtocol
        self.featureExtractor = AudioFeatureExtractor(
            audioPipeline: DummyAudioPipeline(),
            config: AudioFeatureExtractorConfiguration.defaultConfiguration(sampleRate: 44100)
        )
        self.modelConfig = ModelConfiguration(optimizationLevel: optimizationLevel)
        self.outputTransformer = OutputTransformer(configuration: .defaultConfiguration())
        // Complete initialization
        super.init()
        
        // Setup delegates and callbacks
        self.featureExtractor.delegate = self
        self.outputTransformer.delegate = self
        
        // Log initialization
        logger.info("MLProcessor initialized with optimization level: \(optimizationLevel)")
        logger.info("Neural Engine usage: \(useNeuralEngine ? "enabled" : "disabled")")
        
        // Start monitoring performance
        startPerformanceMonitoring()
    }
    
    /// Prepares the ML model for processing
    /// - Parameter audioFormat: The audio format to use for processing
    /// - Throws: MLProcessorError if preparation fails
    public func prepareMLModel(with audioFormat: AVAudioFormat? = nil) async throws {
        // Update state to preparing
        await setProcessingState(.preparing)
        
        logger.info("Preparing ML processor pipeline")
        
        do {
            // Configure the model configuration for the current device
            configureModelForCurrentDevice()
            
            // Get default audio format if none provided
            let format = audioFormat ?? AVAudioFormat(
                standardFormatWithSampleRate: 44100,
                channels: 1
            )
            
            guard let format else {
                throw MLProcessorError.audioFormatError
            }
            
            self.audioFormat = format
            
            // Prepare the feature extractor
            try await featureExtractor.prepare(with: format)
            
            // Configure the output transformer for real-time visualization
            let transformConfig = TransformationConfiguration.spectrumConfiguration(
                size: 64,
                smoothing: 0.3
            )
            outputTransformer.updateConfiguration(transformConfig)
            
            // In a real implementation, we would load a CoreML model here
            // For now, we'll create a placeholder that will be activated later
            // The feature extractor will provide the actual functionality
            
            // NOTE: In a production app, you would load your trained ML model here:
            // if let modelURL = Bundle.module.url(forResource: "AudioAnalysisModel", withExtension: "mlmodelc") {
            //     self.mlModel = try await modelConfig.compileModel(at: modelURL)
            // }
            
            // For this implementation, we'll primarily use the SoundAnalysis framework
            // via the feature extractor, but we're still setting up the structure to
            // incorporate a custom CoreML model in the future
            
            // Update the ready state
            await setProcessingState(.ready)
            
            logger.info("ML processor pipeline prepared successfully")
            
            // Notify delegate of ready state change
            delegate?.mlProcessorReadyStateChanged(isReady: true)
            
        } catch {
            logger.error("Failed to prepare ML pipeline: \(error.localizedDescription)")
            
            // Update state to error
            await setProcessingState(.error(error))
            
            // Notify delegate of error
            delegate?.mlProcessorDidEncounterError(error)
            
            throw MLProcessorError.featureExtractionFailed(error)
        }
    }
    
    /// Processes audio data through the ML pipeline
    /// - Parameter audioData: Raw audio sample data to process
    /// - Throws: MLProcessorError if processing fails
    public func processAudioData(_ audioData: [Float]) async throws {
        guard isReady else {
            throw MLProcessorError.modelNotLoaded
        }
        
        // Begin performance tracking
        let processID = signposter.makeSignpostID()
        let state = signposter.beginInterval("Process Audio Data", id: processID)
        
        // Update state to processing
        await setProcessingState(.processing)
        
        do {
            guard let format = audioFormat else {
                throw MLProcessorError.audioFormatError
            }
            
            // Convert Float array to AVAudioPCMBuffer
            let buffer = try createPCMBuffer(from: audioData, format: format)
            
            // Process through feature extractor
            try await featureExtractor.process(buffer: buffer)
            
            // If we have a custom ML model, also process through it
            if let model = mlModel {
                // In a real implementation, we would process data through CoreML here
                // For now, we'll skip this since we're using SoundAnalysis framework
            }
            
            // Update performance metrics
            updateProcessingTimeMetric()
            
            signposter.endInterval("Process Audio Data", state)
            
        } catch {
            logger.error("Failed to process audio data: \(error.localizedDescription)")
            
            // End performance tracking
            signposter.endInterval("Process Audio Data", state)
            
            // Update state to error
            await setProcessingState(.error(error))
            
            // Notify delegate
            delegate?.mlProcessorDidEncounterError(error)
            
            throw MLProcessorError.processingFailed(error)
        }
    }
    
    /// Processes a PCM buffer directly
    /// - Parameter buffer: The audio buffer to process
    /// - Throws: MLProcessorError if processing fails
    public func processAudioBuffer(_ buffer: AVAudioPCMBuffer) async throws {
        guard isReady else {
            throw MLProcessorError.modelNotLoaded
        }
        
        // Begin performance tracking
        let processID = signposter.makeSignpostID()
        let state = signposter.beginInterval("Process Audio Buffer", id: processID)
        
        // Update state to processing
        await setProcessingState(.processing)
        
        do {
            // Process buffer through feature extractor
            try await featureExtractor.process(buffer: buffer)
            
            // Update performance metrics
            updateProcessingTimeMetric()
            
            signposter.endInterval("Process Audio Buffer", state)
            
        } catch {
            logger.error("Failed to process audio buffer: \(error.localizedDescription)")
            
            // End performance tracking
            signposter.endInterval("Process Audio Buffer", state)
            
            // Update state to error
            await setProcessingState(.error(error))
            
            // Notify delegate
            delegate?.mlProcessorDidEncounterError(error)
            
            throw MLProcessorError.processingFailed(error)
        }
    }
    
    /// Starts continuous audio processing
    /// - Throws: MLProcessorError if starting fails
    public func startContinuousProcessing() throws {
        guard isReady else {
            throw MLProcessorError.modelNotLoaded
        }
        
        logger.info("Starting continuous processing")
        
        // Start the feature extractor
        featureExtractor.start()
        
        // Create a task for continuous processing
        processingTask = Task {
            // This task would typically integrate with an audio capture system
            // For this implementation, the audio buffers will be provided externally
            // via the processAudioBuffer method
        }
    }
    
    /// Stops continuous audio processing
    public func stopContinuousProcessing() {
        logger.info("Stopping continuous processing")
        
        // Stop the feature extractor
        featureExtractor.stop()
        
        // Cancel the processing task
        processingTask?.cancel()
        processingTask = nil
        
        // Set state to ready (if it was processing)
        if state == .processing {
            Task {
                await setProcessingState(.ready)
            }
        }
    }
    
    /// Cleanup resources
    public func cleanup() {
        logger.info("Cleaning up ML processor resources")
        
        // Stop processing if active
        if state == .processing {
            stopContinuousProcessing()
        }
        
        // Reset components
        mlModel = nil
        
        // Clean up audio data subscription
        audioDataSubscription?.cancel()
        audioDataSubscription = nil
        
        // Set state to inactive
        Task {
            await setProcessingState(.inactive)
        }
    }
    
    // MARK: - Audio Data Consumer
    
    /// The subscription to audio data updates
    private var audioDataSubscription: AnyCancellable?
    
    /// Processes audio data from an AudioDataProvider
    /// - Parameter audioData: The audio data to process
    public func processAudioData(_ audioData: AudioData) {
        // Skip processing if we're not ready
        guard isReady else { return }
        
        // Process on our dedicated queue to avoid blocking the main thread
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Extract frequency data for processing
            let frequencyData = audioData.frequencyData
            
            // Skip empty data
            guard !frequencyData.isEmpty else { return }
            
            // Process data asynchronously
            Task {
                do {
                    try await self.processAudioData(frequencyData)
                } catch {
                    // Silently handle processing errors during normal operation
                    if self.state != .error(error) {
                        self.logger.error("Audio data processing failed: \(error.localizedDescription)")
                    }
                }
            }
        }
    }
    
    /// Subscribes to audio data from the specified provider
    /// - Parameter provider: The audio data provider
    public func subscribeToAudioData(from provider: AudioDataProvider) {
        logger.info("Subscribing to audio data from provider")
        
        // Cancel any existing subscription
        audioDataSubscription?.cancel()
        
        // Create a new subscription
        audioDataSubscription = provider.audioDataPublisher
            .receive(on: processingQueue)
            .sink { [weak self] audioData in
                self?.processAudioData(audioData)
            }
    }
    
    /// Unsubscribes from audio data
    public func unsubscribeFromAudioData() {
        logger.info("Unsubscribing from audio data")
        
        audioDataSubscription?.cancel()
        audioDataSubscription = nil
    }
    
    // MARK: - Neural Engine Configuration
    
    /// Configures the model for the current device
    private func configureModelForCurrentDevice() {
        logger.debug("Configuring model for current device")
        
        // Apply basic optimization level
        modelConfig.applyOptimizationLevel(optimizationLevel)
        
        // Use device-specific optimizations
        if useNeuralEngine {
            modelConfig.optimizeForCurrentDevice()
            
            // Enable audio-specific optimizations for Neural Engine
            if modelConfig.neuralEngineCapabilities.isAvailable {
                logger.info("Applying Neural Engine optimizations for audio")
                
                // Add audio-specific parameters
                modelConfig.setParameter(MLParameterKey.frequencyBandCount, value: 128)
                modelConfig.setParameter(MLParameterKey.temporalResolution, value: 0.01) // 10ms
            }
        }
    }
    
    // MARK: - State Management
    
    /// Updates the processing state
    /// - Parameter newState: The new state
    @MainActor
    private func setProcessingState(_ newState: ProcessingState) {
        // Only update if state has changed
        guard state != newState else { return }
        
        let oldState = state
        state = newState
        
        logger.info("State changed from \(String(describing: oldState)) to \(String(describing: newState))")
        
        // Notify ready state changes
        if isReady != (oldState == .ready || oldState == .processing) {
            delegate?.mlProcessorReadyStateChanged(isReady: isReady)
        }
    }
    
    // MARK: - Performance Monitoring
    
    /// Starts performance monitoring
    private func startPerformanceMonitoring() {
        logger.debug("Starting performance monitoring")
        
        // Subscribe to Neural Engine utilization updates
        let cancellable = modelConfig.utilizationPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] utilization in
                self?.performanceMetrics.neuralEngineUtilization = utilization
                self?.updatePerformanceMetrics()
            }
        
        // In a real implementation, we would store the cancellable
        
        // Start periodic performance metric updates
        Task {
            while !Task.isCancelled {
                updatePerformanceMetrics()
                try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
            }
        }
    }
    
    /// Updates processing time metrics
    private func updateProcessingTimeMetric() {
        // In a real implementation, this would track actual processing times
        // For demonstration, we'll use simulated values
        let processingTime = Double.random(in: 1...5) // 1-5ms
        
        // Update metrics (simulated for demonstration)
        performanceMetrics.averageProcessingTime = processingTime
        performanceMetrics.framesPerSecond = 1000.0 / processingTime
    }
    
    /// Updates all performance metrics
    private func updatePerformanceMetrics() {
        // Update memory usage (simulated)
        performanceMetrics.memoryUsage = Double.random(in: 50...150) // 50-150MB
        
        // Update CPU usage (simulated)
        performanceMetrics.cpuUsage = Double.random(in: 0.1...0.3) // 10-30%
        
        // Update timestamp
        performanceMetrics.lastUpdateTime = Date()
        
        // Notify delegate
        DispatchQueue.main.async {
            self.delegate?.mlProcessorDidUpdateMetrics(self.performanceMetrics)
        }
    }
    
    // MARK: - Utilities
    
    /// Creates an audio PCM buffer from Float array data
    /// - Parameters:
    ///   - data: The Float array audio data
    ///   - format: The audio format to use
    /// - Returns: An AVAudioPCMBuffer containing the data
    /// - Throws: Error if buffer creation fails
    private func createPCMBuffer(from data: [Float], format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        // Create buffer with appropriate capacity
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(data.count)) else {
            throw MLProcessorError.audioFormatError
        }
        
        // Set the frame length to match data count
        buffer.frameLength = AVAudioFrameCount(data.count)
        
        // Copy data to the buffer
        if let bufferChannelData = buffer.floatChannelData {
            for i in 0..<data.count {
                bufferChannelData[0][i] = data[i]
            }
        }
        
        return buffer
    }
    
    // MARK: - AudioFeatureExtractorDelegate
    /// Called when new audio features are extracted
    /// - Parameter features: The extracted audio features
    public func featureExtractor(_ extractor: AudioFeatureExtractor, didExtract features: AudioFeatures) {
        // Update our stored audio features
        Task { @MainActor in
            self.audioFeatures = features
        }
        
        // Transform the features for visualization
        Task {
            do {
                // Transform the appropriate feature type
                if !features.frequencySpectrum.isEmpty {
                    let visualizationData = try outputTransformer.transform(
                        features: features,
                        type: .frequencySpectrum
                    )
                    
                    // Update our output data (used for bindings)
                    await MainActor.run {
                        self.outputData = visualizationData.values
                    }
                    
                    // Publish to subscribers
                    visualizationDataSubject.send(visualizationData)
                }
                
                // If beat detected, also transform rhythmic features
                if features.beatDetected {
                    let _ = try outputTransformer.transform(
                        features: features,
                        type: .rhythmicFeatures
                    )
                    // Note: The transformer's delegate will handle publishing this data
                }
            } catch {
                logger.error("Failed to transform features: \(error.localizedDescription)")
            }
        }
    }
    /// Called when an error occurs during feature extraction
    /// - Parameter error: The error that occurred
    public func featureExtractor(_ extractor: AudioFeatureExtractor, didFailExtractingFeature featureType: AudioFeatureType, withError error: Error) {
        logger.error("Feature extraction error: \(error.localizedDescription)")
        
        // Update state to error
        Task {
            await setProcessingState(.error(error))
        }
        
        // Notify delegate
        delegate?.mlProcessorDidEncounterError(error)
    }
    
    // MARK: - VisualizationDataReceiver
    
    /// Called when new visualization data is available
    /// - Parameters:
    ///   - data: The visualization data
    ///   - feature: The feature type
    public func didReceiveVisualizationData(_ data: VisualizationData, forFeature feature: AudioFeatureType) {
        // For most feature types, we'll just forward the data to subscribers
        if feature != .frequencySpectrum {
            visualizationDataSubject.send(data)
        }
        
        // For frequency data, we've already handled it in didExtractFeatures
    }
}
