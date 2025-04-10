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
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public enum AudioFeatureType: String, CaseIterable, Identifiable, Sendable {
    /// Fast Fourier Transform magnitude
/// Uses Swift 6 actor isolation for thread safety.
    case fftMagnitude = "FFT Magnitude"
    /// Mel Frequency Cepstral Coefficients
/// Uses Swift 6 actor isolation for thread safety.
    case mfcc = "MFCC"
    /// Spectral centroid
/// Uses Swift 6 actor isolation for thread safety.
    case spectralCentroid = "Spectral Centroid"
    /// Spectral flatness
/// Uses Swift 6 actor isolation for thread safety.
    case spectralFlatness = "Spectral Flatness"
    /// Spectral rolloff
/// Uses Swift 6 actor isolation for thread safety.
    case spectralRolloff = "Spectral Rolloff"
    /// Spectral flux
/// Uses Swift 6 actor isolation for thread safety.
    case spectralFlux = "Spectral Flux"
    /// Zero crossing rate
/// Uses Swift 6 actor isolation for thread safety.
    case zeroCrossingRate = "Zero Crossing Rate"
    /// Root mean square energy
/// Uses Swift 6 actor isolation for thread safety.
    case rmsEnergy = "RMS Energy"
    /// Frequency band energy
/// Uses Swift 6 actor isolation for thread safety.
    case frequencyBandEnergy = "Frequency Band Energy"
    /// Chroma features
/// Uses Swift 6 actor isolation for thread safety.
    case chroma = "Chroma"
    /// Onset detection
/// Uses Swift 6 actor isolation for thread safety.
    case onset = "Onset Detection"
    /// Beat detection
/// Uses Swift 6 actor isolation for thread safety.
    case beat = "Beat Detection"
    /// Pitch detection
/// Uses Swift 6 actor isolation for thread safety.
    case pitch = "Pitch Detection"
    /// Tempo estimation
/// Uses Swift 6 actor isolation for thread safety.
    case tempo = "Tempo Estimation"
    
    public var id: String { self.rawValue }
    
    /// Get the feature dimension (output size)
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct AudioFeatureExtractorConfiguration: Sendable: Sendable {
    /// Sample rate in Hz
/// Uses Swift 6 actor isolation for thread safety.
    public var sampleRate: Double
    
    /// FFT size (must be a power of 2)
/// Uses Swift 6 actor isolation for thread safety.
    public var fftSize: Int
    
    /// Hop size (frame advance) in samples
/// Uses Swift 6 actor isolation for thread safety.
    public var hopSize: Int
    
    /// Window type for FFT
/// Uses Swift 6 actor isolation for thread safety.
    public var windowType: WindowType
    
    /// Number of MFCC coefficients to compute
/// Uses Swift 6 actor isolation for thread safety.
    public var mfccCoefficients: Int
    
    /// Number of Mel filter banks
/// Uses Swift 6 actor isolation for thread safety.
    public var melFilterBanks: Int
    
    /// Minimum frequency for analysis in Hz
/// Uses Swift 6 actor isolation for thread safety.
    public var minFrequency: Float
    
    /// Maximum frequency for analysis in Hz
/// Uses Swift 6 actor isolation for thread safety.
    public var maxFrequency: Float
    
    /// Frequency bands for band energy calculation
/// Uses Swift 6 actor isolation for thread safety.
    public var frequencyBands: [(min: Float, max: Float)]
    
    /// Whether to use GPU acceleration
/// Uses Swift 6 actor isolation for thread safety.
    public var useGPU: Bool
    
    /// Whether to use Neural Engine
/// Uses Swift 6 actor isolation for thread safety.
    public var useNeuralEngine: Bool
    
    /// Window overlap percentage (0.0-1.0)
/// Uses Swift 6 actor isolation for thread safety.
    public var windowOverlap: Float
    
    /// Number of history frames to keep for temporal features
/// Uses Swift 6 actor isolation for thread safety.
    public var historyFrameCount: Int
    
    /// Whether to normalize output features
/// Uses Swift 6 actor isolation for thread safety.
    public var normalizeFeatures: Bool
    
    /// Whether to use SoundAnalysis for advanced analysis
/// Uses Swift 6 actor isolation for thread safety.
    public var useSoundAnalysis: Bool
    
    /// Path to CoreML model for advanced analysis
/// Uses Swift 6 actor isolation for thread safety.
    public var coreMLModelPath: String?
    
    /// Window function types
/// Uses Swift 6 actor isolation for thread safety.
    @available(macOS 15.0, *)
    public enum WindowType: String, CaseIterable, Sendable {
        case hann
        case hamming
        case blackman
        case rectangular
        
        /// Create a window function buffer of the specified size
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter sampleRate: Sample rate in Hz
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Default configuration
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct AudioFeatures: Sendable: Sendable {
    /// Source sample timestamps
/// Uses Swift 6 actor isolation for thread safety.
    public let timestamp: TimeInterval
    
    /// Feature type
/// Uses Swift 6 actor isolation for thread safety.
    public let featureType: AudioFeatureType
    
    /// Feature values
/// Uses Swift 6 actor isolation for thread safety.
    public let values: [Float]
    
    /// Additional metadata
/// Uses Swift 6 actor isolation for thread safety.
    public let metadata: [String: Any]
    
    /// Duration of the audio frame used for extraction
/// Uses Swift 6 actor isolation for thread safety.
    public let frameDuration: TimeInterval
    
    /// Create a new feature result
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - timestamp: Source sample timestamps
/// Uses Swift 6 actor isolation for thread safety.
    ///   - featureType: Feature type
/// Uses Swift 6 actor isolation for thread safety.
    ///   - values: Feature values
/// Uses Swift 6 actor isolation for thread safety.
    ///   - metadata: Additional metadata
/// Uses Swift 6 actor isolation for thread safety.
    ///   - frameDuration: Duration of the audio frame used for extraction
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol AudioFeatureExtractorDelegate: AnyObject {
    /// Called when new features are extracted
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter features: Extracted features
/// Uses Swift 6 actor isolation for thread safety.
    func featureExtractor(_ extractor: AudioFeatureExtractor, didExtract features: AudioFeatures)
    
    /// Called when a feature extraction error occurs
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - featureType: Feature type that caused the error
/// Uses Swift 6 actor isolation for thread safety.
    ///   - error: The error that occurred
/// Uses Swift 6 actor isolation for thread safety.
    func featureExtractor(_ extractor: AudioFeatureExtractor, didFailExtractingFeature featureType: AudioFeatureType, withError error: Error)
}

/// Audio buffer identifier type
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public typealias AudioBufferID = UUID

/// Protocol for audio pipeline integration
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol AudioPipelineProtocol: AnyObject {
    /// Subscribe to audio data with the specified parameters
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - bufferSize: Size of buffers to receive
/// Uses Swift 6 actor isolation for thread safety.
    ///   - hopSize: Hop size between buffers
/// Uses Swift 6 actor isolation for thread safety.
    ///   - callback: Callback to receive buffer identifiers
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Subscription identifier
/// Uses Swift 6 actor isolation for thread safety.
    func subscribe(bufferSize: Int, hopSize: Int, callback: @escaping (AudioBufferID, TimeInterval) -> Void) -> UUID
    
    /// Unsubscribe from audio data
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter subscriptionID: Subscription identifier to cancel
/// Uses Swift 6 actor isolation for thread safety.
    func unsubscribe(_ subscriptionID: UUID)
    
    /// Unsubscribe all subscriptions for this client
/// Uses Swift 6 actor isolation for thread safety.
    func unsubscribeAll()
    
    /// Get data from a buffer
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter bufferID: Buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Audio sample data, or nil if buffer is invalid
/// Uses Swift 6 actor isolation for thread safety.
    func getBufferData(_ bufferID: AudioBufferID) -> [Float]?
    
    /// Release a buffer when done processing
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter bufferID: Buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    func releaseBuffer(_ bufferID: AudioBufferID)
    
    /// Get the current audio format
/// Uses Swift 6 actor isolation for thread safety.
    var format: AVAudioFormat { get }
}

/// Metal compute core for GPU-accelerated processing
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol MetalComputeProtocol: AnyObject {
    /// Create a Metal buffer
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - size: Buffer size in bytes
/// Uses Swift 6 actor isolation for thread safety.
    ///   - options: Resource options
/// Uses Swift 6 actor isolation for thread safety.
    ///   - label: Debug label
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    func createBuffer(size: Int, options: MTLResourceOptions, label: String) throws -> UInt64
    
    /// Create a buffer with initial data
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - data: Initial data
/// Uses Swift 6 actor isolation for thread safety.
    ///   - options: Resource options
/// Uses Swift 6 actor isolation for thread safety.
    ///   - label: Debug label
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    func createBufferWithData<T>(data: [T], options: MTLResourceOptions, label: String) throws -> UInt64
    
    /// Update buffer contents
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - bufferID: Buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    ///   - data: New data
/// Uses Swift 6 actor isolation for thread safety.
    func updateBuffer<T>(_ bufferID: UInt64, data: [T]) throws
    
    /// Read data from a buffer
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - bufferID: Buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    ///   - count: Number of elements to read
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Buffer contents
/// Uses Swift 6 actor isolation for thread safety.
    func readBuffer<T>(_ bufferID: UInt64, count: Int) throws -> [T]
    
    /// Release a buffer
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter bufferID: Buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    func releaseBuffer(_ bufferID: UInt64)
    
    /// Execute a Metal kernel function
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - name: Kernel function name
/// Uses Swift 6 actor isolation for thread safety.
    ///   - globalSize: Size of global execution grid
/// Uses Swift 6 actor isolation for thread safety.
    ///   - arguments: Kernel arguments
/// Uses Swift 6 actor isolation for thread safety.
    func executeKernel(name: String, globalSize: [Int], arguments: [Any]) throws
    
    /// Execute FFT operation
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - inputBuffer: Input buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    ///   - outputBuffer: Output buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    ///   - magnitudeBuffer: Magnitude buffer identifier
/// Uses Swift 6 actor isolation for thread safety.
    ///   - size: FFT size
/// Uses Swift 6 actor isolation for thread safety.
    func executeFFT(inputBuffer: UInt64, outputBuffer: UInt64, magnitudeBuffer: UInt64, size: Int) throws
}

/// Unified audio feature extractor implementation
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public final actor AudioFeatureExtractor: @unchecked Sendable  {
    // Converted to actor in Swift 6 for thread safety
    // MARK: - Properties
    
    /// Logger instance for detailed logging
/// Uses Swift 6 actor isolation for thread safety.
    private let logger = Logger(subsystem: "com.audiobloom.core", category: "feature-extractor")
    
    /// OS logger for system integration
/// Uses Swift 6 actor isolation for thread safety.
    private let osLogger = os.Logger(subsystem: "com.audiobloom.core", category: "feature-extractor")
    
    /// Configuration parameters
/// Uses Swift 6 actor isolation for thread safety.
    public private(set) var config: AudioFeatureExtractorConfiguration
    
    /// Audio pipeline for processing
/// Uses Swift 6 actor isolation for thread safety.
    private let audioPipeline: AudioPipelineProtocol
    
    /// Metal compute core for GPU acceleration
/// Uses Swift 6 actor isolation for thread safety.
    private let metalCore: MetalComputeProtocol?
    
    /// Delegate for receiving extraction results
/// Uses Swift 6 actor isolation for thread safety.
    public weak var delegate: AudioFeatureExtractorDelegate?
    
    /// Feature types to extract
/// Uses Swift 6 actor isolation for thread safety.
    public private(set) var activeFeatures: Set<AudioFeatureType> = []
    
    /// Current audio format
/// Uses Swift 6 actor isolation for thread safety.
    private var audioFormat: AVAudioFormat
    
    /// Window function buffer
/// Uses Swift 6 actor isolation for thread safety.
    private var window: [Float]
    
    /// Previous frame buffer for temporal processing
/// Uses Swift 6 actor isolation for thread safety.
    private var previousFrames: [AudioFeatureType: [[Float]]] = [:]
    
    /// FFT buffer (reused for memory efficiency)
/// Uses Swift 6 actor isolation for thread safety.
    private var fftBuffer: [Float]
    
    /// Feature history buffer for temporal features
/// Uses Swift 6 actor isolation for thread safety.
    private var featureHistory: [AudioFeatureType: RingBuffer<[Float]>] = [:]
    
    /// Mel filter bank weights
/// Uses Swift 6 actor isolation for thread safety.
    private var melFilterBank: [[Float]] = []
    
    /// DCT matrix for MFCC computation
/// Uses Swift 6 actor isolation for thread safety.
    private var dctMatrix: [[Float]] = []
    
    /// GPU buffers for frequency domain processing
/// Uses Swift 6 actor isolation for thread safety.
    private var gpuBuffers: [String: UInt64] = [:]
    
    /// Whether the extractor is running
/// Uses Swift 6 actor isolation for thread safety.
    public private(set) var isRunning = false
    
    /// Audio subscription identifier
/// Uses Swift 6 actor isolation for thread safety.
    private var audioSubscriptionID: UUID?
    
    /// Queue for buffer management and extraction
/// Uses Swift 6 actor isolation for thread safety.
    private let extractionQueue = DispatchQueue(label: "com.audiobloom.feature-extraction", qos: .userInteractive)
    
    /// Currently processing audio buffer IDs
/// Uses Swift 6 actor isolation for thread safety.
    private var processingBuffers: Set<AudioBufferID> = []
    
    /// Sound analysis engine for advanced analysis
/// Uses Swift 6 actor isolation for thread safety.
    private var audioAnalyzer: SNAudioStreamAnalyzer?
    
    /// Neural Engine optimized ML model for audio analysis
/// Uses Swift 6 actor isolation for thread safety.
    private var audioAnalysisModel: MLModel?
    
    /// Frequency data publisher for real-time updates
/// Uses Swift 6 actor isolation for thread safety.
    private let frequencySubject = PassthroughSubject<[Float], Never>()
    
    /// Lock for thread safety
/// Uses Swift 6 actor isolation for thread safety.
    // Lock removed - actor provides automatic isolation
    
    /// Publisher for frequency data
/// Uses Swift 6 actor isolation for thread safety.
    public var frequencyPublisher: AnyPublisher<[Float], Never> {
        frequencySubject.eraseToAnyPublisher()
    }
    
    /// Subject for publishing beat detection events
/// Uses Swift 6 actor isolation for thread safety.
    private let beatSubject = PassthroughSubject<(confidence: Float, timestamp: TimeInterval), Never>()
    
    /// Publisher for beat detection events
/// Uses Swift 6 actor isolation for thread safety.
    public var beatPublisher: AnyPublisher<(confidence: Float, timestamp: TimeInterval), Never> {
        beatSubject.eraseToAnyPublisher()
    }
    
    // MARK: - Lifecycle
    
    /// Initialize the feature extractor
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - audioPipeline: Audio pipeline for processing
/// Uses Swift 6 actor isolation for thread safety.
    ///   - config: Configuration parameters (optional)
/// Uses Swift 6 actor isolation for thread safety.
    ///   - metalCore: Metal compute engine for GPU acceleration (optional)
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter config: New configuration
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
    private func frequencyToMel(_ frequency: Float) -> Float {
        return 2595 * log10(1 + frequency / 700)
    }
    
    /// Convert Mel scale to frequency
/// Uses Swift 6 actor isolation for thread safety.
    private func melToFrequency(_ mel: Float) -> Float {
        return 700 * (pow(10, mel / 2595) - 1)
    }
    
    // MARK: - Feature Extraction Control
    
    /// Start feature extraction
/// Uses Swift 6 actor isolation for thread safety.
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
        
        // Update state isRunning = true osLogger.info("Feature extraction started with \(activeFeatures.count) active features")
    }
    
    /// Stop feature extraction
/// Uses Swift 6 actor isolation for thread safety.
    public func stop() {
        guard isRunning else { return }
        
        // Update state first to prevent new processing isRunning = false // Unsubscribe from audio pipeline
        if let subscriptionID = audioSubscriptionID {
            audioPipeline.unsubscribe(subscriptionID)
            audioSubscriptionID = nil
        }
        
        // Clean up any remaining buffers for bufferID in processingBuffers {
            audioPipeline.releaseBuffer(bufferID)
        }
        processingBuffers.removeAll() osLogger.info("Feature extraction stopped")
    }
    
    // MARK: - Audio Processing
    
    /// Process an audio buffer
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - buffer: Audio buffer
/// Uses Swift 6 actor isolation for thread safety.
    ///   - timestamp: Buffer timestamp
/// Uses Swift 6 actor isolation for thread safety.
    private func processAudioBuffer(_ buffer: AudioBufferID, timestamp: TimeInterval) {
        // Check if we're still running
        guard isRunning else {
            audioPipeline.releaseBuffer(buffer)
            return
        }
        
        // Track this buffer as being processed processingBuffers.insert(buffer) // Get audio data from buffer
        guard let samples = audioPipeline.getBufferData(buffer) else { processingBuffers.remove(buffer) osLogger.warning("Failed to get data from buffer \(buffer)")
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
        audioPipeline.releaseBuffer(buffer) processingBuffers.remove(buffer) }
    
    /// Create an AVAudioPCMBuffer from samples
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter samples: Sample data
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: PCM buffer, or nil if creation fails
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - samples: Audio samples
/// Uses Swift 6 actor isolation for thread safety.
    ///   - timestamp: Buffer timestamp
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter features: Set of features to extract
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
    private func resetHistory() { defer { }
        
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
/// Uses Swift 6 actor isolation for thread safety.
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
/// Uses Swift 6 actor isolation for thread safety.
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

