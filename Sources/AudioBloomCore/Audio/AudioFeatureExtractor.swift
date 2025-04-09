//
// AudioFeatureExtractor.swift
// Unified audio feature extraction implementation
//

import Foundation
import Accelerate
import AVFoundation
import Combine
import Logging
import Metal
import SoundAnalysis
import CoreML
import os.log

/// Errors that can occur during audio feature extraction
@available(macOS 15.0, *)
public enum AudioFeatureExtractorError: Error, CustomStringConvertible, Sendable {
    // Core errors
    case extractionFailed(String)
    case configurationError(String)
    case metalComputeError(Error)
    case invalidParameters(String)
    case featureNotSupported(String)
    case insufficientData(String)
    
    // SoundAnalysis errors
    case analysisEngineCreationFailed
    case audioFormatError
    case modelNotFound
    case modelCompilationFailed
    case observerRegistrationFailed
    case soundAnalysisError(Error)
    case bufferProcessingError
    case neuralEngineUnavailable
    
    public var description: String {
        switch self {
        // Core errors
        case .extractionFailed(let details):
            return "Feature extraction failed: \(details)"
        case .configurationError(let details):
            return "Configuration error: \(details)"
        case .metalComputeError(let error):
            return "Metal compute error: \(error)"
        case .invalidParameters(let details):
            return "Invalid parameters: \(details)"
        case .featureNotSupported(let details):
            return "Feature not supported: \(details)"
        case .insufficientData(let details):
            return "Insufficient data: \(details)"
            
        // SoundAnalysis errors
        case .analysisEngineCreationFailed:
            return "Failed to create sound analysis engine"
        case .audioFormatError:
            return "Audio format is incompatible with analysis"
        case .modelNotFound:
            return "CoreML model not found"
        case .modelCompilationFailed:
            return "Failed to compile CoreML model"
        case .observerRegistrationFailed:
            return "Failed to register sound analysis observer"
        case .soundAnalysisError(let error):
            return "Sound analysis error: \(error.localizedDescription)"
        case .bufferProcessingError:
            return "Error processing audio buffer"
        case .neuralEngineUnavailable:
            return "Neural Engine is not available on this device"
        }
    }
}

/// Types of audio features that can be extracted
@available(macOS 15.0, *)
public enum AudioFeatureType: String, CaseIterable, Identifiable, Sendable {
    /// Fast Fourier Transform magnitude
    case fftMagnitude = "FFT Magnitude"
    /// Mel Frequency Cepstral Coefficients
    case mfcc = "MFCC"
    /// Spectral centroid
    case spectralCentroid = "Spectral Centroid"
    /// Spectral flatness
    case spectralFlatness = "Spectral Flatness"
    /// Spectral rolloff
    case spectralRolloff = "Spectral Rolloff"
    /// Spectral flux
    case spectralFlux = "Spectral Flux"
    /// Zero crossing rate
    case zeroCrossingRate = "Zero Crossing Rate"
    /// Root mean square energy
    case rmsEnergy = "RMS Energy"
    /// Frequency band energy
    case frequencyBandEnergy = "Frequency Band Energy"
    /// Chroma features
    case chroma = "Chroma"
    /// Onset detection
    case onset = "Onset Detection"
    /// Beat detection
    case beat = "Beat Detection"
    /// Pitch detection
    case pitch = "Pitch Detection"
    /// Tempo estimation
    case tempo = "Tempo Estimation"
    
    public var id: String { self.rawValue }
    
    /// Get the feature dimension (output size)
    public func featureDimension(config: AudioFeatureExtractorConfiguration) -> Int {
        switch self {
        case .fftMagnitude:
            return config.fftSize / 2
        case .mfcc:
            return config.mfccCoefficients
        case .spectralCentroid, .spectralFlatness, .spectralRolloff, .rmsEnergy, .tempo, .pitch:
            return 1
        case .zeroCrossingRate:
            return 1
        case .frequencyBandEnergy:
            return config.frequencyBands.count
        case .chroma:
            return 12 // 12 pitch classes
        case .spectralFlux:
            return 1
        case .onset, .beat:
            return 1
        }
    }
    
    /// Check if this feature requires FFT computation
    public var requiresFFT: Bool {
        switch self {
        case .fftMagnitude, .mfcc, .spectralCentroid, .spectralFlatness, .spectralRolloff, .spectralFlux,
             .frequencyBandEnergy, .chroma:
            return true
        case .zeroCrossingRate, .rmsEnergy, .onset, .beat, .pitch, .tempo:
            return false
        }
    }
    
    /// Check if this feature requires temporal processing
    public var requiresTemporalProcessing: Bool {
        switch self {
        case .spectralFlux, .onset, .beat, .tempo:
            return true
        default:
            return false
        }
    }
}

/// Audio feature configuration parameters
@available(macOS 15.0, *)
public struct AudioFeatureExtractorConfiguration: Sendable {
    /// Sample rate in Hz
    public var sampleRate: Double
    
    /// FFT size (must be a power of 2)
    public var fftSize: Int
    
    /// Hop size (frame advance) in samples
    public var hopSize: Int
    
    /// Window type for FFT
    public var windowType: WindowType
    
    /// Number of MFCC coefficients to compute
    public var mfccCoefficients: Int
    
    /// Number of Mel filter banks
    public var melFilterBanks: Int
    
    /// Minimum frequency for analysis in Hz
    public var minFrequency: Float
    
    /// Maximum frequency for analysis in Hz
    public var maxFrequency: Float
    
    /// Frequency bands for band energy calculation
    public var frequencyBands: [(min: Float, max: Float)]
    
    /// Whether to use GPU acceleration
    public var useGPU: Bool
    
    /// Whether to use Neural Engine
    public var useNeuralEngine: Bool
    
    /// Window overlap percentage (0.0-1.0)
    public var windowOverlap: Float
    
    /// Number of history frames to keep for temporal features
    public var historyFrameCount: Int
    
    /// Whether to normalize output features
    public var normalizeFeatures: Bool
    
    /// Whether to use SoundAnalysis for advanced analysis
    public var useSoundAnalysis: Bool
    
    /// Path to CoreML model for advanced analysis
    public var coreMLModelPath: String?
    
    /// Window function types
    public enum WindowType: String, CaseIterable, Sendable {
        case hann
        case hamming
        case blackman
        case rectangular
        
        /// Create a window function buffer of the specified size
        func createWindow(size: Int) -> [Float] {
            var window = [Float](repeating: 0, count: size)
            
            switch self {
            case .hann:
                vDSP_hann_window(&window, vDSP_Length(size), Int32(vDSP_HANN_NORM))
            case .hamming:
                vDSP_hamm_window(&window, vDSP_Length(size), 0)
            case .blackman:
                vDSP_blkman_window(&window, vDSP_Length(size), 0)
            case .rectangular:
                // Rectangular window is all 1s
                for i in 0..<size {
                    window[i] = 1.0
                }
            }
            
            return window
        }
    }
    
    /// Create a default configuration
    /// - Parameter sampleRate: Sample rate in Hz
    /// - Returns: Default configuration
    public static func defaultConfiguration(sampleRate: Double) -> AudioFeatureExtractorConfiguration {
        // Default frequency bands (in Hz): bass, low-mid, mid, high-mid, treble
        let bands: [(min: Float, max: Float)] = [
            (20, 250),    // Bass
            (250, 500),   // Low-mid
            (500, 2000),  // Mid
            (2000, 4000), // High-mid
            (4000, 20000) // Treble
        ]
        
        return AudioFeatureExtractorConfiguration(
            sampleRate: sampleRate,
            fftSize: 2048,
            hopSize: 512,
            windowType: .hann,
            mfccCoefficients: 13,
            melFilterBanks: 40,
            minFrequency: 20,
            maxFrequency: min(Float(sampleRate / 2), 20000),
            frequencyBands: bands,
            useGPU: true,
            useNeuralEngine: true,
            windowOverlap: 0.5,
            historyFrameCount: 8,
            normalizeFeatures: true,
            useSoundAnalysis: true,
            coreMLModelPath: nil
        )
    }
}

/// Extracted audio features result
@available(macOS 15.0, *)
public struct AudioFeatures: Sendable {
    /// Source sample timestamps
    public let timestamp: TimeInterval
    
    /// Feature type
    public let featureType: AudioFeatureType
    
    /// Feature values
    public let values: [Float]
    
    /// Additional metadata
    public let metadata: [String: Any]
    
    /// Duration of the audio frame used for extraction
    public let frameDuration: TimeInterval
    
    /// Create a new feature result
    /// - Parameters:
    ///   - timestamp: Source sample timestamps
    ///   - featureType: Feature type
    ///   - values: Feature values
    ///   - metadata: Additional metadata
    ///   - frameDuration: Duration of the audio frame used for extraction
    public init(
        timestamp: TimeInterval,
        featureType: AudioFeatureType,
        values: [Float],
        metadata: [String: Any] = [:],
        frameDuration: TimeInterval
    ) {
        self.timestamp = timestamp
        self.featureType = featureType
        self.values = values
        self.metadata = metadata
        self.frameDuration = frameDuration
    }
}

/// Protocol for receiving audio feature extraction results
@available(macOS 15.0, *)
public protocol AudioFeatureExtractorDelegate: AnyObject {
    /// Called when new features are extracted
    /// - Parameter features: Extracted features
    func featureExtractor(_ extractor: AudioFeatureExtractor, didExtract features: AudioFeatures)
    
    /// Called when a feature extraction error occurs
    /// - Parameters:
    ///   - featureType: Feature type that caused the error
    ///   - error: The error that occurred
    func featureExtractor(_ extractor: AudioFeatureExtractor, didFailExtractingFeature featureType: AudioFeatureType, withError error: Error)
}

/// Audio buffer identifier type
@available(macOS 15.0, *)
public typealias AudioBufferID = UUID

/// Protocol for audio pipeline integration
@available(macOS 15.0, *)
public protocol AudioPipelineProtocol: AnyObject {
    /// Subscribe to audio data with the specified parameters
    /// - Parameters:
    ///   - bufferSize: Size of buffers to receive
    ///   - hopSize: Hop size between buffers
    ///   - callback: Callback to receive buffer identifiers
    /// - Returns: Subscription identifier
    func subscribe(bufferSize: Int, hopSize: Int, callback: @escaping (AudioBufferID, TimeInterval) -> Void) -> UUID
    
    /// Unsubscribe from audio data
    /// - Parameter subscriptionID: Subscription identifier to cancel
    func unsubscribe(_ subscriptionID: UUID)
    
    /// Unsubscribe all subscriptions for this client
    func unsubscribeAll()
    
    /// Get data from a buffer
    /// - Parameter bufferID: Buffer identifier
    /// - Returns: Audio sample data, or nil if buffer is invalid
    func getBufferData(_ bufferID: AudioBufferID) -> [Float]?
    
    /// Release a buffer when done processing
    /// - Parameter bufferID: Buffer identifier
    func releaseBuffer(_ bufferID: AudioBufferID)
    
    /// Get the current audio format
    var format: AVAudioFormat { get }
}

/// Metal compute core for GPU-accelerated processing
@available(macOS 15.0, *)
public protocol MetalComputeProtocol: AnyObject {
    /// Create a Metal buffer
    /// - Parameters:
    ///   - size: Buffer size in bytes
    ///   - options: Resource options
    ///   - label: Debug label
    /// - Returns: Buffer identifier
    func createBuffer(size: Int, options: MTLResourceOptions, label: String) throws -> UInt64
    
    /// Create a buffer with initial data
    /// - Parameters:
    ///   - data: Initial data
    ///   - options: Resource options
    ///   - label: Debug label
    /// - Returns: Buffer identifier
    func createBufferWithData<T>(data: [T], options: MTLResourceOptions, label: String) throws -> UInt64
    
    /// Update buffer contents
    /// - Parameters:
    ///   - bufferID: Buffer identifier
    ///   - data: New data
    func updateBuffer<T>(_ bufferID: UInt64, data: [T]) throws
    
    /// Read data from a buffer
    /// - Parameters:
    ///   - bufferID: Buffer identifier
    ///   - count: Number of elements to read
    /// - Returns: Buffer contents
    func readBuffer<T>(_ bufferID: UInt64, count: Int) throws -> [T]
    
    /// Release a buffer
    /// - Parameter bufferID: Buffer identifier
    func releaseBuffer(_ bufferID: UInt64)
    
    /// Execute a Metal kernel function
    /// - Parameters:
    ///   - name: Kernel function name
    ///   - globalSize: Size of global execution grid
    ///   - arguments: Kernel arguments
    func executeKernel(name: String, globalSize: [Int], arguments: [Any]) throws
    
    /// Execute FFT operation
    /// - Parameters:
    ///   - inputBuffer: Input buffer identifier
    ///   - outputBuffer: Output buffer identifier
    ///   - magnitudeBuffer: Magnitude buffer identifier
    ///   - size: FFT size
    func executeFFT(inputBuffer: UInt64, outputBuffer: UInt64, magnitudeBuffer: UInt64, size: Int) throws
}

/// Unified audio feature extractor implementation
@available(macOS 15.0, *)
public final class AudioFeatureExtractor: @unchecked Sendable {
    // MARK: - Properties
    
    /// Logger instance for detailed logging
    private let logger = Logger(subsystem: "com.audiobloom.core", category: "feature-extractor")
    
    /// OS logger for system integration
    private let osLogger = os.Logger(subsystem: "com.audiobloom.core", category: "feature-extractor")
    
    /// Configuration parameters
    public private(set) var config: AudioFeatureExtractorConfiguration
    
    /// Audio pipeline for processing
    private let audioPipeline: AudioPipelineProtocol
    
    /// Metal compute core for GPU acceleration
    private let metalCore: MetalComputeProtocol?
    
    /// Delegate for receiving extraction results
    public weak var delegate: AudioFeatureExtractorDelegate?
    
    /// Feature types to extract
    public private(set) var activeFeatures: Set<AudioFeatureType> = []
    
    /// Current audio format
    private var audioFormat: AVAudioFormat
    
    /// Window function buffer
    private var window: [Float]
    
    /// Previous frame buffer for temporal processing
    private var previousFrames: [AudioFeatureType: [[Float]]] = [:]
    
    /// FFT buffer (reused for memory efficiency)
    private var fftBuffer: [Float]
    
    /// Feature history buffer for temporal features
    private var featureHistory: [AudioFeatureType: RingBuffer<[Float]>] = [:]
    
    /// Mel filter bank weights
    private var melFilterBank: [[Float]] = []
    
    /// DCT matrix for MFCC computation
    private var dctMatrix: [[Float]] = []
    
    /// GPU buffers for frequency domain processing
    private var gpuBuffers: [String: UInt64] = [:]
    
    /// Whether the extractor is running
    public private(set) var isRunning = false
    
    /// Audio subscription identifier
    private var audioSubscriptionID: UUID?
    
    /// Queue for buffer management and extraction
    private let extractionQueue = DispatchQueue(label: "com.audiobloom.feature-extraction", qos: .userInteractive)
    
    /// Currently processing audio buffer IDs
    private var processingBuffers: Set<AudioBufferID> = []
    
    /// Sound analysis engine for advanced analysis
    private var audioAnalyzer: SNAudioStreamAnalyzer?
    
    /// Neural Engine optimized ML model for audio analysis
    private var audioAnalysisModel: MLModel?
    
    /// Frequency data publisher for real-time updates
    private let frequencySubject = PassthroughSubject<[Float], Never>()
    
    /// Lock for thread safety
    private let lock = NSLock()
    
    /// Publisher for frequency data
    public var frequencyPublisher: AnyPublisher<[Float], Never> {
        frequencySubject.eraseToAnyPublisher()
    }
    
    /// Subject for publishing beat detection events
    private let beatSubject = PassthroughSubject<(confidence: Float, timestamp: TimeInterval), Never>()
    
    /// Publisher for beat detection events
    public var beatPublisher: AnyPublisher<(confidence: Float, timestamp: TimeInterval), Never> {
        beatSubject.eraseToAnyPublisher()
    }
    
    // MARK: - Lifecycle
    
    /// Initialize the feature extractor
    /// - Parameters:
    ///   - audioPipeline: Audio pipeline for processing
    ///   - config: Configuration parameters (optional)
    ///   - metalCore: Metal compute engine for GPU acceleration (optional)
    public init(
        audioPipeline: AudioPipelineProtocol,
        config: AudioFeatureExtractorConfiguration? = nil,
        metalCore: MetalComputeProtocol? = nil
    ) {
        self.audioPipeline = audioPipeline
        self.audioFormat = audioPipeline.format
        
        // Create configuration with defaults if none provided
        self.config = config ?? AudioFeatureExtractorConfiguration.defaultConfiguration(sampleRate: audioFormat.sampleRate)
        
        // Use provided Metal compute core or initialize if GPU is enabled
        if let providedCore = metalCore {
            self.metalCore = providedCore
            osLogger.debug("Using provided Metal compute core")
        } else if self.config.useGPU {
            // In a real implementation, this would create a default Metal compute core
            self.metalCore = nil
            osLogger.debug("Metal compute core not provided and default implementation not available")
        } else {
            self.metalCore = nil
            osLogger.debug("GPU acceleration disabled for feature extraction")
        }
        
        // Create window function buffer
        self.window = self.config.windowType.createWindow(size: self.config.fftSize)
        
        // Initialize FFT buffer
        self.fftBuffer = [Float](repeating: 0, count: self.config.fftSize)
        
        // Setup Mel filter banks and DCT matrix for MFCC
        setupMelFilterBanks()
        setupDCTMatrix()
        
        // Initialize SoundAnalysis if enabled
        if self.config.useSoundAnalysis {
            setupSoundAnalysis()
        }
        
        // Initialize Neural Engine model if enabled
        if self.config.useNeuralEngine {
            setupNeuralEngine()
        }
        
        osLogger.info("Audio feature extractor initialized: sampleRate=\(self.config.sampleRate), fftSize=\(self.config.fftSize)")
    }
    
    deinit {
        // Stop processing
        stop()
        
        // Clean up GPU resources
        releaseGPUBuffers()
        
        osLogger.debug("AudioFeatureExtractor deinitialized")
    }
    
    // MARK: - Configuration
    
    /// Configure the feature extractor
    /// - Parameter config: New configuration
    public func configure(with config: AudioFeatureExtractorConfiguration) throws {
        // Ensure we're not running
        guard config.useNeuralEngine else { return }
        
        #if ENABLE_NEURAL_ENGINE
        // Configure ML model with Neural Engine optimizations
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .all // Use Neural Engine when available
        
        // Check if custom model path is provided
        if let modelPath = config.coreMLModelPath, !modelPath.isEmpty {
            do {
                let modelURL = URL(fileURLWithPath: modelPath)
                self.audioAnalysisModel = try MLModel(contentsOf: modelURL, configuration: mlConfig)
                osLogger.info("Loaded custom Neural Engine model from: \(modelPath)")
            } catch {
                osLogger.error("Failed to load custom ML model: \(error)")
            }
        } else {
            // Check for bundled models
            osLogger.debug("No custom model specified, looking for bundled models")
        }
        #else
        osLogger.info("Neural Engine support disabled (ENABLE_NEURAL_ENGINE not defined)")
        #endif
    }
    
    // MARK: - GPU Buffer Management
    
    /// Allocate GPU buffers for feature extraction
    private func allocateGPUBuffers() throws {
        guard let metalCore = metalCore else { return }
        
        // Release any existing buffers
        releaseGPUBuffers()
        
        do {
            // Create buffer for FFT data
            let fftBufferID = try metalCore.createBuffer(
                size: config.fftSize * MemoryLayout<Float>.stride,
                options: .storageModeShared,
                label: "fftInputBuffer"
            )
            gpuBuffers["fftInput"] = fftBufferID
            
            // Create buffer for FFT output
            let fftOutputBufferID = try metalCore.createBuffer(
                size: config.fftSize * MemoryLayout<Float>.stride,
                options: .storageModeShared,
                label: "fftOutputBuffer"
            )
            gpuBuffers["fftOutput"] = fftOutputBufferID
            
            // Create buffer for spectrum magnitude
            let magnitudeBufferID = try metalCore.createBuffer(
                size: (config.fftSize / 2) * MemoryLayout<Float>.stride,
                options: .storageModeShared,
                label: "magnitudeBuffer"
            )
            gpuBuffers["magnitude"] = magnitudeBufferID
            
            // Create buffer for Mel filterbank weights
            if !melFilterBank.isEmpty {
                var flattenedMelBank: [Float] = []
                for row in melFilterBank {
                    flattenedMelBank.append(contentsOf: row)
                }
                
                let melBankBufferID = try metalCore.createBufferWithData(
                    data: flattenedMelBank,
                    options: .storageModeShared,
                    label: "melFilterBankBuffer"
                )
                gpuBuffers["melFilterBank"] = melBankBufferID
            }
            
            // Create buffer for MFCC computations
            if activeFeatures.contains(.mfcc) {
                let mfccBufferID = try metalCore.createBuffer(
                    size: config.melFilterBanks * MemoryLayout<Float>.stride,
                    options: .storageModeShared,
                    label: "mfccBuffer"
                )
                gpuBuffers["mfcc"] = mfccBufferID
                
                // Create buffer for DCT matrix
                var flattenedDCT: [Float] = []
                for row in dctMatrix {
                    flattenedDCT.append(contentsOf: row)
                }
                
                let dctBufferID = try metalCore.createBufferWithData(
                    data: flattenedDCT,
                    options: .storageModeShared,
                    label: "dctMatrixBuffer"
                )
                gpuBuffers["dctMatrix"] = dctBufferID
            }
            
            osLogger.debug("GPU buffers allocated successfully")
            
        } catch {
            osLogger.error("Failed to allocate GPU buffers: \(error)")
            throw AudioFeatureExtractorError.metalComputeError(error)
        }
    }
    
    /// Release GPU buffers
    private func releaseGPUBuffers() {
        guard let metalCore = metalCore else { return }
        
        for (_, bufferID) in gpuBuffers {
            metalCore.releaseBuffer(bufferID)
        }
        
        gpuBuffers.removeAll()
        osLogger.debug("GPU buffers released")
    }
    
    // MARK: - Filter Bank Setup
    
    /// Setup Mel filter banks for MFCC computation
    private func setupMelFilterBanks() {
        let fftBins = config.fftSize / 2 + 1
        let minMel = frequencyToMel(config.minFrequency)
        let maxMel = frequencyToMel(config.maxFrequency)
        
        // Create equally spaced points in the Mel scale
        let melPoints = (0..<config.melFilterBanks + 2).map { i in
            minMel + (maxMel - minMel) * Float(i) / Float(config.melFilterBanks + 1)
        }
        
        // Convert back to frequency
        let hzPoints = melPoints.map { melToFrequency($0) }
        
        // Convert to FFT bin indices
        let bins = hzPoints.map { hz -> Int in
            let bin = Int(floor((config.fftSize + 1) * Float(hz) / Float(config.sampleRate)))
            return min(max(0, bin), fftBins - 1) // Clamp to valid range
        }
        
        // Create filterbank matrix
        melFilterBank = Array(repeating: Array(repeating: 0.0, count: fftBins), count: config.melFilterBanks)
        
        for m in 0..<config.melFilterBanks {
            let filterStart = bins[m]
            let filterCenter = bins[m + 1]
            let filterEnd = bins[m + 2]
            
            // Triangular filter
            for k in filterStart..<filterCenter {
                if k < fftBins {
                    melFilterBank[m][k] = (Float(k) - Float(filterStart)) / (Float(filterCenter) - Float(filterStart))
                }
            }
            
            for k in filterCenter..<filterEnd {
                if k < fftBins {
                    melFilterBank[m][k] = (Float(filterEnd) - Float(k)) / (Float(filterEnd) - Float(filterCenter))
                }
            }
        }
    }
    
    /// Setup DCT matrix for MFCC computation
    private func setupDCTMatrix() {
        let M = config.melFilterBanks
        let N = config.mfccCoefficients
        
        dctMatrix = Array(repeating: Array(repeating: 0.0, count: M), count: N)
        
        // Create DCT matrix
        for n in 0..<N {
            for m in 0..<M {
                dctMatrix[n][m] = cos(Float.pi * Float(n) * (Float(m) + 0.5) / Float(M)) * (n == 0 ? sqrt(1.0 / Float(M)) : sqrt(2.0 / Float(M)))
            }
        }
    }
    
    /// Convert frequency to Mel scale
    private func frequencyToMel(_ frequency: Float) -> Float {
        return 2595 * log10(1 + frequency / 700)
    }
    
    /// Convert Mel scale to frequency
    private func melToFrequency(_ mel: Float) -> Float {
        return 700 * (pow(10, mel / 2595) - 1)
    }
    
    // MARK: - Feature Extraction Control
    
    /// Start feature extraction
    public func start() throws {
        guard !isRunning else { return }
        
        // Ensure we have at least one active feature
        guard !activeFeatures.isEmpty else {
            throw AudioFeatureExtractorError.configurationError("No active features to extract")
        }
        
        // Allocate GPU buffers if needed
        if config.useGPU && metalCore != nil {
            try allocateGPUBuffers()
        }
        
        // Subscribe to audio pipeline
        let subscription = audioPipeline.subscribe(
            bufferSize: config.fftSize,
            hopSize: config.hopSize,
            callback: { [weak self] buffer, timestamp in
                guard let self = self else { return }
                self.extractionQueue.async {
                    self.processAudioBuffer(buffer, timestamp: timestamp)
                }
            }
        )
        
        // Store subscription
        self.audioSubscriptionID = subscription
        
        // Update state
        lock.lock()
        isRunning = true
        lock.unlock()
        
        osLogger.info("Feature extraction started with \(activeFeatures.count) active features")
    }
    
    /// Stop feature extraction
    public func stop() {
        guard isRunning else { return }
        
        // Update state first to prevent new processing
        lock.lock()
        isRunning = false
        lock.unlock()
        
        // Unsubscribe from audio pipeline
        if let subscriptionID = audioSubscriptionID {
            audioPipeline.unsubscribe(subscriptionID)
            audioSubscriptionID = nil
        }
        
        // Clean up any remaining buffers
        lock.lock()
        for bufferID in processingBuffers {
            audioPipeline.releaseBuffer(bufferID)
        }
        processingBuffers.removeAll()
        lock.unlock()
        
        osLogger.info("Feature extraction stopped")
    }
    
    // MARK: - Audio Processing
    
    /// Process an audio buffer
    /// - Parameters:
    ///   - buffer: Audio buffer
    ///   - timestamp: Buffer timestamp
    private func processAudioBuffer(_ buffer: AudioBufferID, timestamp: TimeInterval) {
        // Check if we're still running
        guard isRunning else {
            audioPipeline.releaseBuffer(buffer)
            return
        }
        
        // Track this buffer as being processed
        lock.lock()
        processingBuffers.insert(buffer)
        lock.unlock()
        
        // Get audio data from buffer
        guard let samples = audioPipeline.getBufferData(buffer) else {
            lock.lock()
            processingBuffers.remove(buffer)
            lock.unlock()
            osLogger.warning("Failed to get data from buffer \(buffer)")
            return
        }
        
        // Use SoundAnalysis if enabled
        if config.useSoundAnalysis, let analyzer = audioAnalyzer {
            do {
                // Convert to AVAudioPCMBuffer
                if let pcmBuffer = createPCMBuffer(from: samples) {
                    try analyzer.analyze(pcmBuffer, atAudioFramePosition: 0)
                }
            } catch {
                osLogger.error("SoundAnalysis error: \(error)")
            }
        }
        
        // Extract features using our own implementation
        do {
            try extractFeaturesFromSamples(samples, timestamp: timestamp)
        } catch {
            osLogger.error("Feature extraction error: \(error)")
        }
        
        // Process with Neural Engine if enabled and model is available
        #if ENABLE_NEURAL_ENGINE
        if config.useNeuralEngine, let model = audioAnalysisModel {
            do {
                try processWithNeuralEngine(samples, timestamp: timestamp, model: model)
            } catch {
                osLogger.error("Neural Engine processing error: \(error)")
            }
        }
        #endif
        
        // Release the buffer
        audioPipeline.releaseBuffer(buffer)
        lock.lock()
        processingBuffers.remove(buffer)
        lock.unlock()
    }
    
    /// Create an AVAudioPCMBuffer from samples
    /// - Parameter samples: Sample data
    /// - Returns: PCM buffer, or nil if creation fails
    private func createPCMBuffer(from samples: [Float]) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
            return nil
        }
        
        buffer.frameLength = frameCount
        
        // Copy samples to buffer
        if let channelData = buffer.floatChannelData, channelData.count > 0 {
            let channelPtr = channelData[0]
            for i in 0..<samples.count {
                channelPtr[i] = samples[i]
            }
        }
        
        return buffer
    }
    
    /// Extract features from audio samples
    /// - Parameters:
    ///   - samples: Audio samples
    ///   - timestamp: Buffer timestamp
    private func extractFeaturesFromSamples(_ samples: [Float], timestamp: TimeInterval) throws {
        // Ensure we have enough samples
        guard samples.count == config.fftSize else {
            throw AudioFeatureExtractorError.insufficientData("Sample count mismatch: got \(samples.count), expected \(config.fftSize)")
        }
        
        // Apply window function
        var windowedSamples = [Float](repeating: 0, count: config.fftSize)
        vDSP_vmul(samples, 1, window, 1, &windowedSamples, 1, vDSP_Length(config.fftSize))
        
        // Compute FFT if needed for any active feature
        var magnitudeSpectrum: [Float]?
        
        if activeFeatures.contains(where: { $0.requiresFFT }) {
            // Calculate FFT
            magnitudeSpectrum = try computeFFT(windowedSamples)
            
            // Publish frequency data for real-time visualization
            if let spectrum = magnitudeSpectrum {
                frequencySubject.send(spectrum)
            }
        }
        
        // Calculate frame duration
        let frameDuration = TimeInterval(config.fftSize) / TimeInterval(config.sampleRate)
        
        // Extract each active feature
        for featureType in activeFeatures {
            do {
                switch featureType {
                case .fftMagnitude:
                    if let magnitudeSpectrum = magnitudeSpectrum {
                        let feature = AudioFeatures(
                            timestamp: timestamp,
                            featureType: featureType,
                            values: magnitudeSpectrum,
                            metadata: [:],
                            frameDuration: frameDuration
                        )
                        delegate?.featureExtractor(self, didExtract: feature)
                    }
                    
                case .mfcc:
                    if let magnitudeSpectrum = magnitu
        // Update Mel filter banks and DCT matrix
        setupMelFilterBanks()
        setupDCTMatrix()
        
        // Release and recreate GPU buffers if needed
        if self.config.useGPU && metalCore != nil {
            releaseGPUBuffers()
            try allocateGPUBuffers()
        }
        
        // Update SoundAnalysis setup if needed
        if self.config.useSoundAnalysis {
            setupSoundAnalysis()
        }
        
        // Update Neural Engine model if needed
        if self.config.useNeuralEngine {
            setupNeuralEngine()
        }
        
        osLogger.info("Feature extractor reconfigured: sampleRate=\(self.config.sampleRate), fftSize=\(self.config.fftSize)")
    }
    
    /// Set active features for extraction
    /// - Parameter features: Set of features to extract
    public func setActiveFeatures(_ features: Set<AudioFeatureType>) throws {
        // Ensure we're not running
        guard !isRunning else {
            throw AudioFeatureExtractorError.configurationError("Cannot change active features while running")
        }
        
        // Update active features
        self.activeFeatures = features
        
        // Initialize history buffers for temporal features
        resetHistory()
        
        // Log active features
        osLogger.info("Active features updated: \(features.map { $0.rawValue }.joined(separator: ", "))")
    }
    
    /// Reset history buffers
    private func resetHistory() {
        lock.lock()
        defer { lock.unlock() }
        
        // Clear previous frames
        previousFrames.removeAll()
        
        // Initialize history buffers for temporal features
        featureHistory.removeAll()
        
        for feature in activeFeatures {
            if feature.requiresTemporalProcessing {
                featureHistory[feature] = RingBuffer<[Float]>(capacity: config.historyFrameCount)
            }
        }
    }
    
    // MARK: - Neural Engine and SoundAnalysis Setup
    
    /// Set up SoundAnalysis components
    private func setupSoundAnalysis() {
        guard config.useSoundAnalysis else { return }
        
        // Create a new audio analyzer with the current format
        self.audioAnalyzer = SNAudioStreamAnalyzer(format: audioFormat)
        
        guard let analyzer = self.audioAnalyzer else {
            osLogger.warning("Failed to create audio analyzer")
            return
        }
        
        // Register sound analysis requests based on active features
        do {
            // Register frequency analysis if needed
            if activeFeatures.contains(.fftMagnitude) {
                if let request = try? SNSpectrumRequest(maximumFrequency: 22000) {
                    let observer = SpectrumAnalysisObserver(extractor: self)
                    try analyzer.add(request, withObserver: observer)
                    osLogger.debug("Registered spectrum analysis request")
                }
            }
            
            // Register tempo detection if needed
            if activeFeatures.contains(.tempo) || activeFeatures.contains(.beat) {
                if let request = try? SNTempoRequest() {
                    let observer = TempoAnalysisObserver(extractor: self)
                    try analyzer.add(request, withObserver: observer)
                    osLogger.debug("Registered tempo detection request")
                }
            }
        } catch {
            osLogger.error("Failed to register sound analysis requests: \(error)")
        }
    }
    
    /// Set up Neural Engine model
    private func setupNeuralEngine() {
        guard config.useNeuralEngine else { return }
        
        #if ENABLE_NEURAL_ENGINE
        // Configure ML model with Neural Engine optimizations
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .all // Use Neural Engine when available
        
        // Check if custom model path is provided
        if let modelPath = config.coreMLModelPath, !modelPath.isEmpty {
            do {
                let modelURL = URL(fileURLWithPath: modelPath)
                self.audioAnalysisModel = try MLModel(contentsOf: modelURL, configuration: mlConfig)
                osLogger.info("Loaded custom Neural Engine model from: \(modelPath)")
            } catch {
                osLogger.error("Failed to load custom ML model: \(error)")
            }

