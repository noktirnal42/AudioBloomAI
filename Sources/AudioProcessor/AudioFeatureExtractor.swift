import Foundation
import Accelerate
import AVFoundation
import Combine
import Logging
import Metal
import AudioBloomCore

/// Errors that can occur during audio feature extraction
@available(macOS 15.0, *)
public enum AudioFeatureExtractorError: Error {
    /// Feature extraction failed
    case extractionFailed(String)
    /// Configuration error
    case configurationError(String)
    /// Metal compute error
    case metalComputeError(Error)
    /// Invalid feature parameters
    case invalidParameters(String)
    /// Feature not supported
    case featureNotSupported(String)
    /// Insufficient data for extraction
    case insufficientData(String)
}

/// Types of audio features that can be extracted
@available(macOS 15.0, *)
public enum AudioFeatureType: String, CaseIterable, Identifiable {
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
public struct AudioFeatureExtractorConfiguration {
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
    
    /// Window overlap percentage (0.0-1.0)
    public var windowOverlap: Float
    
    /// Number of history frames to keep for temporal features
    public var historyFrameCount: Int
    
    /// Whether to normalize output features
    public var normalizeFeatures: Bool
    
    /// Window function types
    public enum WindowType: String, CaseIterable {
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
            windowOverlap: 0.5,
            historyFrameCount: 8,
            normalizeFeatures: true
        )
    }
}

/// Extracted audio features result
@available(macOS 15.0, *)
public struct AudioFeatures {
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

/// Audio feature extractor for real-time audio analysis
@available(macOS 15.0, *)
public class AudioFeatureExtractor {
    // MARK: - Properties
    
    /// Logger instance
    private let logger = Logger(label: "com.audiobloom.feature-extractor")
    
    /// Configuration parameters
    public private(set) var config: AudioFeatureExtractorConfiguration
    
    /// Audio pipeline for processing
    private let audioPipeline: AudioPipelineCore
    
    /// Metal compute core for GPU acceleration
    private let metalCore: MetalComputeCore?
    
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
    
    /// Queue for buffer management
    private let extractionQueue = DispatchQueue(label: "com.audiobloom.feature-extraction", qos: .userInteractive)
    
    /// Currently processing audio buffer IDs
    private var processingBuffers: Set<AudioBufferID> = []
    
    // MARK: - Lifecycle
    
    /// Initialize the feature extractor
    /// - Parameters:
    ///   - config: Configuration parameters
    ///   - audioPipeline: Audio pipeline for processing
    public init(config: AudioFeatureExtractorConfiguration? = nil, audioPipeline: AudioPipelineCore) throws {
        self.audioPipeline = audioPipeline
        self.audioFormat = audioPipeline.format
        
        // Create configuration with defaults if none provided
        self.config = config ?? AudioFeatureExtractorConfiguration.defaultConfiguration(sampleRate: audioFormat.sampleRate)
        
        // Initialize Metal compute core if GPU is enabled
        if self.config.useGPU {
            do {
                self.metalCore = try MetalComputeCore(maxConcurrentOperations: 3)
                logger.info("GPU acceleration enabled for feature extraction")
            } catch {
                logger.warning("Failed to initialize Metal compute core: \(error). Falling back to CPU processing.")
                self.metalCore = nil
            }
        } else {
            self.metalCore = nil
            logger.info("GPU acceleration disabled for feature extraction")
        }
        
        // Create window function buffer
        self.window = self.config.windowType.createWindow(size: self.config.fftSize)
        
        // Initialize FFT buffer
        self.fftBuffer = [Float](repeating: 0, count: self.config.fftSize)
        
        // Setup Mel filter banks and DCT matrix for MFCC
        setupMelFilterBanks()
        setupDCTMatrix()
        
        logger.info("Audio feature extractor initialized: sampleRate=\(self.config.sampleRate), fftSize=\(self.config.fftSize)")
    }
    
    deinit {
        // Clean up GPU resources
        releaseGPUBuffers()
        
        logger.debug("AudioFeatureExtractor deinitialized")
    }
    
    // MARK: - Configuration
    
    /// Configure the feature extractor
    /// - Parameter config: New configuration
    public func configure(with config: AudioFeatureExtractorConfiguration) throws {
        // Ensure we're not running
        guard !isRunning else {
            throw AudioFeatureExtractorError.configurationError("Cannot reconfigure while running")
        }
        
        // Update configuration
        self.config = config
        
        // Update window function
        self.window = self.config.windowType.createWindow(size: self.config.fftSize)
        
        // Resize FFT buffer
        self.fftBuffer = [Float](repeating: 0, count: self.config.fftSize)
        
        // Reset history buffers
        resetHistory()
        
        // Update Mel filter banks and DCT matrix
        setupMelFilterBanks()
        setupDCTMatrix()
        
        // Release and recreate GPU buffers if needed
        if self.config.useGPU && metalCore != nil {
            releaseGPUBuffers()
            try allocateGPUBuffers()
        }
        
        logger.info("Feature extractor reconfigured: sampleRate=\(self.config.sampleRate), fftSize=\(self.config.fftSize)")
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
        logger.info("Active features updated: \(features.map { $0.rawValue }.joined(separator: ", "))")
    }
    
    /// Reset history buffers
    private func resetHistory() {
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
    
    // MARK: - GPU Buffer Management
    
    /// Allocate GPU buffers for feature extraction
    private

