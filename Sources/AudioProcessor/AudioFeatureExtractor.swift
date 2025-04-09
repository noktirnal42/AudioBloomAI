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
            
            logger.debug("GPU buffers allocated successfully")
            
        } catch {
            logger.error("Failed to allocate GPU buffers: \(error)")
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
        logger.debug("GPU buffers released")
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
        
        // Update state
        isRunning = true
        logger.info("Feature extraction started with \(activeFeatures.count) active features")
    }
    
    /// Stop feature extraction
    public func stop() {
        guard isRunning else { return }
        
        // Unsubscribe from audio pipeline
        audioPipeline.unsubscribeAll()
        
        // Update state
        isRunning = false
        logger.info("Feature extraction stopped")
    }
    
    // MARK: - Audio Processing
    
    /// Process an audio buffer
    /// - Parameters:
    ///   - buffer: Audio buffer
    ///   - timestamp: Buffer timestamp
    private func processAudioBuffer(_ buffer: AudioBufferID, timestamp: TimeInterval) {
        // Track this buffer as being processed
        processingBuffers.insert(buffer)
        
        // Get audio data from buffer
        guard let samples = audioPipeline.getBufferData(buffer) else {
            processingBuffers.remove(buffer)
            logger.warning("Failed to get data from buffer \(buffer)")
            return
        }
        
        // Extract features
        do {
            try extractFeaturesFromSamples(samples, timestamp: timestamp)
        } catch {
            logger.error("Feature extraction error: \(error)")
        }
        
        // Release the buffer
        audioPipeline.releaseBuffer(buffer)
        processingBuffers.remove(buffer)
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
                    if let magnitudeSpectrum = magnitudeSpectrum {
                        let mfccValues = try computeMFCC(magnitudeSpectrum)
                        let feature = AudioFeatures(
                            timestamp: timestamp,
                            featureType: featureType,
                            values: mfccValues,
                            metadata: [:],
                            frameDuration: frameDuration
                        )
                        delegate?.featureExtractor(self, didExtract: feature)
                    }
                    
                case .spectralCentroid:
                    if let magnitudeSpectrum = magnitudeSpectrum {
                        let centroid = try computeSpectralCentroid(magnitudeSpectrum)
                        let feature = AudioFeatures(
                            timestamp: timestamp,
                            featureType: featureType,
                            values: [centroid],
                            metadata: [:],
                            frameDuration: frameDuration
                        )
                        delegate?.featureExtractor(self, didExtract: feature)
                    }
                    
                case .spectralFlatness:
                    if let magnitudeSpectrum = magnitudeSpectrum {
                        let flatness = try computeSpectralFlatness(magnitudeSpectrum)
                        let feature = AudioFeatures(
                            timestamp: timestamp,
                            featureType: featureType,
                            values: [flatness],
                            metadata: [:],
                            frameDuration: frameDuration
                        )
                        delegate?.featureExtractor(self, didExtract: feature)
                    }
                    
                case .spectralRolloff:
                    if let magnitudeSpectrum = magnitudeSpectrum {
                        let rolloff = try computeSpectralRolloff(magnitudeSpectrum)
                        let feature = AudioFeatures(
                            timestamp: timestamp,
                            featureType: featureType,
                            values: [rolloff],
                            metadata: [:],
                            frameDuration: frameDuration
                        )
                        delegate?.featureExtractor(self, didExtract: feature)
                    }
                    
                case .spectralFlux:
                    if let magnitudeSpectrum = magnitudeSpectrum {
                        // Get previous spectrum from history if available
                        if let historyBuffer = featureHistory[.spectralFlux],
                           !historyBuffer.isEmpty {
                            let flux = try computeSpectralFlux(
                                current: magnitudeSpectrum,
                                previous: historyBuffer.latest() ?? Array(repeating: 0, count: magnitudeSpectrum.count)
                            )
                            let feature = AudioFeatures(
                                timestamp: timestamp,
                                featureType: featureType,
                                values: [flux],
                                metadata: [:],
                                frameDuration: frameDuration
                            )
                            delegate?.featureExtractor(self, didExtract: feature)
                        }
                        
                        // Store current spectrum for next time
                        featureHistory[.spectralFlux]?.add(magnitudeSpectrum)
                    }
                    
                case .zeroCrossingRate:
                    let zcr = computeZeroCrossingRate(samples)
                    let feature = AudioFeatures(
                        timestamp: timestamp,
                        featureType: featureType,
                        values: [zcr],
                        metadata: [:],
                        frameDuration: frameDuration
                    )
                    delegate?.featureExtractor(self, didExtract: feature)
                    
                case .rmsEnergy:
                    let rms = computeRMSEnergy(samples)
                    let feature = AudioFeatures(
                        timestamp: timestamp,
                        featureType: featureType,
                        values: [rms],
                        metadata: [:],
                        frameDuration: frameDuration
                    )
                    delegate?.featureExtractor(self, didExtract: feature)
                    
                case .frequencyBandEnergy:
                    if let magnitudeSpectrum = magnitudeSpectrum {
                        let bandEnergies = try computeFrequencyBandEnergies(magnitudeSpectrum)
                        let feature = AudioFeatures(
                            timestamp: timestamp,
                            featureType: featureType,
                            values: bandEnergies,
                            metadata: [:],
                            frameDuration: frameDuration
                        )
                        delegate?.featureExtractor(self, didExtract: feature)
                    }
                    
                case .chroma:
                    if let magnitudeSpectrum = magnitudeSpectrum {
                        let chromaFeatures = try computeChromaFeatures(magnitudeSpectrum)
                        let feature = AudioFeatures(
                            timestamp: timestamp,
                            featureType: featureType,
                            values: chromaFeatures,
                            metadata: [:],
                            frameDuration: frameDuration
                        )
                        delegate?.featureExtractor(self, didExtract: feature)
                    }
                    
                case .onset:
                    if let magnitudeSpectrum = magnitudeSpectrum,
                       let historyBuffer = featureHistory[.onset] {
                        let onsetDetected = try detectOnset(
                            magnitudeSpectrum: magnitudeSpectrum,
                            spectrumHistory: historyBuffer.allItems()
                        )
                        let feature = AudioFeatures(
                            timestamp: timestamp,
                            featureType: featureType,
                            values: [onsetDetected ? 1.0 : 0.0],
                            metadata: ["detected": onsetDetected],
                            frameDuration: frameDuration
                        )
                        delegate?.featureExtractor(self, didExtract: feature)
                        
                        // Store current spectrum for next time
                        featureHistory[.onset]?.add(magnitudeSpectrum)
                    }
                    
                case .beat:
                    // Beat detection requires energy information
                    let energy = computeRMSEnergy(samples)
                    
                    if let historyBuffer = featureHistory[.beat] {
                        // Add current energy to history
                        featureHistory[.beat]?.add([energy])
                        
                        // Only attempt beat detection when we have enough history
                        if historyBuffer.count >= config.historyFrameCount / 2 {
                            let beatDetected = try detectBeat(
                                currentEnergy: energy,
                                energyHistory: historyBuffer.allItems().map { $0[0] }
                            )
                            let feature = AudioFeatures(
                                timestamp: timestamp,
                                featureType: featureType,
                                values: [beatDetected ? 1.0 : 0.0],
                                metadata: ["detected": beatDetected],
                                frameDuration: frameDuration
                            )
                            delegate?.featureExtractor(self, didExtract: feature)
                        }
                    }
                    
                case .pitch:
                    let pitchInfo = try detectPitch(samples)
                    let feature = AudioFeatures(
                        timestamp: timestamp,
                        featureType: featureType,
                        values: [pitchInfo.frequency],
                        metadata: [
                            "confidence": pitchInfo.confidence,
                            "clarity": pitchInfo.clarity
                        ],
                        frameDuration: frameDuration
                    )
                    delegate?.featureExtractor(self, didExtract: feature)
                    
                case .tempo:
                    // Tempo estimation requires enough audio history
                    if let beatHistory = featureHistory[.beat],
                       beatHistory.count >= config.historyFrameCount {
                        
                        let tempo = try estimateTempo(
                            beatHistory: beatHistory.allItems().map { $0[0] },
                            hopSize: config.hopSize,
                            sampleRate: config.sampleRate
                        )
                        
                        let feature = AudioFeatures(
                            timestamp: timestamp,
                            featureType: featureType,
                            values: [tempo],
                            metadata: ["bpm": tempo],
                            frameDuration: frameDuration
                        )
                        delegate?.featureExtractor(self, didExtract: feature)
                    }
                }
            } catch {
                logger.error("Failed to extract \(featureType.rawValue): \(error)")
                delegate?.featureExtractor(self, didFailExtractingFeature: featureType, withError: error)
            }
        }
    }
    
    // MARK: - Feature Computation Methods
    
    /// Compute FFT on input samples
    /// - Parameter samples: Audio samples
    /// - Returns: Magnitude spectrum
    private func computeFFT(_ samples: [Float]) throws -> [Float] {
        let fftSize = config.fftSize
        let outputSize = fftSize / 2
        
        // Use GPU if available and enabled
        if config.useGPU, let metalCore = metalCore,
           let inputBufferID = gpuBuffers["fftInput"],
           let outputBufferID = gpuBuffers["fftOutput"],
           let magnitudeBufferID = gpuBuffers["magnitude"] {
            
            // Copy audio data to GPU
            try metalCore.updateBuffer(inputBufferID, data: samples)
            
            // Perform FFT on GPU
            try metalCore.executeFFT(
                inputBuffer: inputBufferID,
                outputBuffer: outputBufferID,
                magnitudeBuffer: magnitudeBufferID,
                size: fftSize
            )
            
            // Get magnitude spectrum back from GPU
            let magnitudeSpectrum = try metalCore.readBuffer(magnitudeBufferID, count: outputSize)
            
            return magnitudeSpectrum
        } else {
            // Perform FFT using Accelerate framework
            let halfSize = fftSize / 2
            
            // Set up FFT
            let log2n = vDSP_Length(log2(Float(fftSize)))
            let fftSetup = vDSP_create_fftsetup(log2n, FFT_RADIX2)!
            defer { vDSP_destroy_fftsetup(fftSetup) }
            
            // Prepare split complex buffer
            var realp = [Float](repeating: 0, count: halfSize)
            var imagp = [Float](repeating: 0, count: halfSize)
            var splitComplex = DSPSplitComplex(realp: &realp, imagp: &imagp)
            
            // Pack samples into split complex format
            samples.withUnsafeBytes { samplePtr in
                vDSP_ctoz(samplePtr.bindMemory(to: DSPComplex.self).baseAddress!, 2, &splitComplex, 1, vDSP_Length(halfSize))
            }
            
            // Perform forward FFT
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFT_FORWARD)
            
            // Scale the results
            let scale = 1.0 / Float(fftSize)
            vDSP_vsmul(splitComplex.realp, 1, &scale, splitComplex.realp, 1, vDSP_Length(halfSize))
            vDSP_vsmul(splitComplex.imagp, 1, &scale, splitComplex.imagp, 1, vDSP_Length(halfSize))
            
            // Compute magnitude
            var magnitudeSpectrum = [Float](repeating: 0, count: halfSize)
            vDSP_zvmags(&splitComplex, 1, &magnitudeSpectrum, 1, vDSP_Length(halfSize))
            
            // Take square root to get amplitude spectrum
            vvsqrtf(&magnitudeSpectrum, &magnitudeSpectrum, [Int32(halfSize)])
            
            return magnitudeSpectrum
        }
    }
    
    /// Compute MFCC from magnitude spectrum
    /// - Parameter magnitudeSpectrum: Magnitude spectrum from FFT
    /// - Returns: MFCC coefficients
    private func computeMFCC(_ magnitudeSpectrum: [Float]) throws -> [Float] {
        let fftSize = config.fftSize
        let halfSize = fftSize / 2
        
        // Use GPU if available and enabled
        if config.useGPU, let metalCore = metalCore,
           let magnitudeBufferID = gpuBuffers["magnitude"],
           let melBankBufferID = gpuBuffers["melFilterBank"],
           let mfccBufferID = gpuBuffers["mfcc"],
           let dctBufferID = gpuBuffers["dctMatrix"] {
            
            // Magnitude spectrum already in GPU memory from FFT computation
            
            // Apply Mel filterbank
            try metalCore.executeKernel(
                name: "applyMelFilterbank",
                globalSize: [config.melFilterBanks],
                arguments: [
                    magnitudeBufferID,
                    melBankBufferID,
                    mfccBufferID,
                    Int32(halfSize),
                    Int32(config.melFilterBanks)
                ]
            )
            
            // Apply log to Mel spectrum
            try metalCore.executeKernel(
                name: "applyLog",
                globalSize: [config.melFilterBanks],
                arguments: [
                    mfccBufferID,
                    mfccBufferID,
                    Int32(config.melFilterBanks)
                ]
            )
            
            // Apply DCT to get MFCCs
            try metalCore.executeKernel(
                name: "applyDCT",
                globalSize: [config.mfccCoefficients],
                arguments: [
                    mfccBufferID,
                    dctBufferID,
                    magnitudeBufferID, // Reuse as output buffer
                    Int32(config.melFilterBanks),
                    Int32(config.mfccCoefficients)
                ]
            )
            
            // Read back MFCC coefficients from GPU
            return try metalCore.readBuffer(magnitudeBufferID, count: config.mfccCoefficients)
            
        } else {
            // CPU implementation
            
            // Apply Mel filterbank
            var melSpectrum = [Float](repeating: 0, count: config.melFilterBanks)
            
            // For each mel filter bank, compute the weighted sum
            for m in 0..<config.melFilterBanks {
                for k in 0..<magnitudeSpectrum.count {
                    melSpectrum[m] += magnitudeSpectrum[k] * melFilterBank[m][k]
                }
                
                // Apply floor to prevent log of zero
                melSpectrum[m] = max(melSpectrum[m], 1e-10)
                
                // Apply log
                melSpectrum[m] = log10(melSpectrum[m])
            }
            
            // Apply DCT to get MFCCs
            var mfccs = [Float](repeating: 0, count: config.mfccCoefficients)
            
            for n in 0..<config.mfccCoefficients {
                for m in 0..<config.melFilterBanks {
                    mfccs[n] += melSpectrum[m] * dctMatrix[n][m]
                }
            }
            
            return mfccs
        }
    }
    
    /// Compute spectral centroid from magnitude spectrum
    /// - Parameter magnitudeSpectrum: Magnitude spectrum from FFT
    /// - Returns: Spectral centroid in Hz
    private func computeSpectralCentroid(_ magnitudeSpectrum: [Float]) throws -> Float {
        let nyquist = Float(config.sampleRate) / 2.0
        let fftSize = config.fftSize
        let binCount = magnitudeSpectrum.count
        
        // Compute frequency bins
        let freqPerBin = nyquist / Float(binCount)
        
        var numerator: Float = 0.0
        var denominator: Float = 0.0
        
        // Calculate weighted sum of bin frequencies
        for i in 0..<binCount {
            let binFrequency = Float(i) * freqPerBin
            let magnitude = magnitudeSpectrum[i]
            
            numerator += binFrequency * magnitude
            denominator += magnitude
        }
        
        // Avoid division by zero
        if denominator < 1e-10 {
            return 0.0
        }
        
        return numerator / denominator
    }
    
    /// Compute spectral flatness from magnitude spectrum
    /// - Parameter magnitudeSpectrum: Magnitude spectrum from FFT
    /// - Returns: Spectral flatness (0-1)
    private func computeSpectralFlatness(_ magnitudeSpectrum: [Float]) throws -> Float {
        let binCount = magnitudeSpectrum.count
        
        // Ensure we have valid data
        if binCount < 2 {
            throw AudioFeatureExtractorError.insufficientData("Magnitude spectrum too small for flatness calculation")
        }
        
        // Copy spectrum and apply floor to prevent log of zero
        var spectrum = magnitudeSpectrum
        for i in 0..<binCount {
            spectrum[i] = max(spectrum[i], 1e-10)
        }
        
        // Calculate geometric mean (using log for numerical stability)
        var logSum: Float = 0.0
        for val in spectrum {
            logSum += log(val)
        }
        let geometricMean = exp(logSum / Float(binCount))
        
        // Calculate arithmetic mean
        var arithmeticMean: Float = 0.0
        vDSP_meanv(spectrum, 1, &arithmeticMean, vDSP_Length(binCount))
        
        // Calculate flatness
        if arithmeticMean < 1e-10 {
            return 0.0
        }
        
        return geometricMean / arithmeticMean
    }
    
    /// Compute spectral rolloff from magnitude spectrum
    /// - Parameter magnitudeSpectrum: Magnitude spectrum from FFT
    /// - Returns: Frequency (in Hz) below which 85% of the spectrum energy is contained
    private func computeSpectralRolloff(_ magnitudeSpectrum: [Float]) throws -> Float {
        let binCount = magnitudeSpectrum.count
        let nyquist = Float(config.sampleRate) / 2.0
        let freqPerBin = nyquist / Float(binCount)
        
        // Calculate total energy
        var totalEnergy: Float = 0.0
        for val in magnitudeSpectrum {
            totalEnergy += val * val // Square for energy
        }
        
        // Threshold is 85% of total energy
        let threshold = totalEnergy * 0.85
        
        // Find the frequency bin where we exceed the threshold
        var cumulativeEnergy: Float = 0.0
        
        for i in 0..<binCount {
            cumulativeEnergy += magnitudeSpectrum[i] * magnitudeSpectrum[i]
            
            if cumulativeEnergy >= threshold {
                return Float(i) * freqPerBin
            }
        }
        
        // If we get here, return the Nyquist frequency
        return nyquist
    }
    
    /// Compute spectral flux between two consecutive frames
    /// - Parameters:
    ///   - current: Current magnitude spectrum
    ///   - previous: Previous magnitude spectrum
    /// - Returns: Spectral flux value
    private func computeSpectralFlux(current: [Float], previous: [Float]) throws -> Float {
        // Ensure both spectra have the same length
        guard current.count == previous.count else {
            throw AudioFeatureExtractorError.invalidParameters("Spectral flux requires equal length spectra")
        }
        
        let binCount = current.count
        var totalDifference: Float = 0.0
        
        // Calculate the sum of the squared differences
        for i in 0..<binCount {
            // Use half-wave rectification (only positive changes)
            let diff = current[i] - previous[i]
            let rectified = max(0.0, diff)
            totalDifference += rectified * rectified
        }
        
        // Normalize by the number of bins
        return totalDifference / Float(binCount)
    }
    
    /// Compute zero crossing rate
    /// - Parameter samples: Audio samples
    /// - Returns: Zero crossing rate (0-1)
    private func computeZeroCrossingRate(_ samples: [Float]) -> Float {
        let count = samples.count
        var crossings = 0
        
        // Count sign changes
        for i in 1..<count {
            if (samples[i-1] < 0 && samples[i] >= 0) || (samples[i-1] >= 0 && samples[i] < 0) {
                crossings += 1
            }
        }
        
        // Normalize by frame length
        return Float(crossings) / Float(count - 1)
    }
    
    /// Compute RMS energy
    /// - Parameter samples: Audio samples
    /// - Returns: RMS energy value (0-1)
    private func computeRMSEnergy(_ samples: [Float]) -> Float {
        let count = samples.count
        var meanSquare: Float = 0.0
        
        // Use vDSP for efficient computation
        vDSP_measqv(samples, 1, &meanSquare, vDSP_Length(count))
        
        // Take square root for RMS
        return sqrt(meanSquare)
    }
    
    /// Compute energy in frequency bands
    /// - Parameter magnitudeSpectrum: Magnitude spectrum from FFT
    /// - Returns: Energy in each frequency band
    private func computeFrequencyBandEnergies(_ magnitudeSpectrum: [Float]) throws -> [Float] {
        let binCount = magnitudeSpectrum.count
        let nyquist = Float(config.sampleRate) / 2.0
        let freqPerBin = nyquist / Float(binCount)
        
        // Initialize band energies
        var bandEnergies = [Float](repeating: 0.0, count: config.frequencyBands.count)
        
        // Calculate the energy in each frequency band
        for (bandIndex, band) in config.frequencyBands.enumerated() {
            // Convert frequency range to bin indices
            let startBin = Int(band.min / freqPerBin)
            let endBin = min(Int(band.max / freqPerBin) + 1, binCount)
            
            // Calculate energy in this band
            var energy: Float = 0.0
            for i in startBin..<endBin {
                if i < binCount {
                    energy += magnitudeSpectrum[i] * magnitudeSpectrum[i]
                }
            }
            
            // Normalize by number of bins
            let binCount = max(1, endBin - startBin)
            bandEnergies[bandIndex] = energy / Float(binCount)
        }
        
        // Optionally normalize across bands
        if config.normalizeFeatures {
            let maxEnergy = bandEnergies.max() ?? 1.0
            if maxEnergy > 1e-10 {
                for i in 0..<bandEnergies.count {
                    bandEnergies[i] /= maxEnergy
                }
            }
        }
        
        return bandEnergies
    }
    
    /// Compute chroma features (pitch class profile)
    /// - Parameter magnitudeSpectrum: Magnitude spectrum from FFT
    /// - Returns: 12-dimensional chroma vector
    private func computeChromaFeatures(_ magnitudeSpectrum: [Float]) throws -> [Float] {
        let binCount = magnitudeSpectrum.count
        let nyquist = Float(config.sampleRate) / 2.0
        let freqPerBin = nyquist / Float(binCount)
        
        // Create 12-dimensional chroma vector (one for each pitch class)
        var chroma = [Float](repeating: 0.0, count: 12)
        
        // Calculate chroma features
        for i in 0..<binCount {
            let frequency = Float(i) * freqPerBin
            
            // Skip very low frequencies
            if frequency < 20.0 {
                continue
            }
            
            // Convert frequency to MIDI note number
            let midiNote = 69.0 + 12.0 * log2(frequency / 440.0)
            
            // Skip if out of reasonable MIDI range
            if midiNote < 0 || midiNote > 127 {
                continue
            }
            
            // Calculate pitch class (C=0, C#=1, ..., B=11)
            let pitchClass = Int(round(midiNote).truncatingRemainder(dividingBy: 12))
            
            // Add magnitude to corresponding pitch class
            chroma[pitchClass] += magnitudeSpectrum[i]
        }
        
        // Normalize chroma vector
        if config.normalizeFeatures {
            let maxValue = chroma.max() ?? 1.0
            if maxValue > 1e-10 {
                for i in 0..<chroma.count {
                    chroma[i] /= maxValue
                }
            }
        }
        
        return chroma
    }
    
    /// Detect onset in audio signal
    /// - Parameters:
    ///   - magnitudeSpectrum: Current magnitude spectrum
    ///   - spectrumHistory: Recent spectrum history
    /// - Returns: Boolean indicating whether an onset was detected
    private func detectOnset(magnitudeSpectrum: [Float], spectrumHistory: [[Float]]) throws -> Bool {
        // Ensure we have enough history
        guard !spectrumHistory.isEmpty else {
            return false // Not enough data to detect onset
        }
        
        // Calculate spectral flux (positive changes only)
        var totalFlux: Float = 0.0
        let previousSpectrum = spectrumHistory.last!
        
        // Ensure spectra match in size
        guard magnitudeSpectrum.count == previousSpectrum.count else {
            throw AudioFeatureExtractorError.invalidParameters("Spectrum size mismatch for onset detection")
        }
        
        // Calculate spectral flux with half-wave rectification
        for i in 0..<magnitudeSpectrum.count {
            let diff = magnitudeSpectrum[i] - previousSpectrum[i]
            if diff > 0 {
                totalFlux += diff
            }
        }
        
        // Normalize flux
        totalFlux /= Float(magnitudeSpectrum.count)
        
        // Calculate adaptive threshold from history
        let historyCount = min(spectrumHistory.count, 5)
        var fluxHistory: [Float] = []
        
        // Calculate flux for recent history frames
        for i in 0..<historyCount-1 {
            if i+1 < spectrumHistory.count {
                let curr = spectrumHistory[i+1]
                let prev = spectrumHistory[i]
                var frameFlux: Float = 0.0
                
                for j in 0..<curr.count {
                    let diff = curr[j] - prev[j]
                    if diff > 0 {
                        frameFlux += diff
                    }
                }
                
                fluxHistory.append(frameFlux / Float(curr.count))
            }
        }
        
        // Calculate adaptive threshold
        var threshold: Float = 0.0
        if !fluxHistory.isEmpty {
            var sum: Float = 0.0
            vDSP_meanv(fluxHistory, 1, &sum, vDSP_Length(fluxHistory.count))
            threshold = sum * 1.5 // Threshold is 1.5x the mean flux
        } else {
            threshold = 0.1 // Default threshold
        }
        
        // Detect onset if flux exceeds threshold
        return totalFlux > threshold
    }
    
    /// Detect beat in audio signal based on energy
    /// - Parameters:
    ///   - currentEnergy: Current frame energy
    ///   - energyHistory: Recent energy history
    /// - Returns: Boolean indicating whether a beat was detected
    private func detectBeat(currentEnergy: Float, energyHistory: [Float]) throws -> Bool {
        // Ensure we have enough history
        guard energyHistory.count >= 4 else {
            return false // Not enough data to detect beat
        }
        
        // Calculate local energy average
        var localAverage: Float = 0.0
        vDSP_meanv(energyHistory, 1, &localAverage, vDSP_Length(energyHistory.count))
        
        // Calculate variance
        var variance: Float = 0.0
        for energy in energyHistory {
            let diff = energy - localAverage
            variance += diff * diff
        }
        variance /= Float(energyHistory.count)
        
        // Calculate standard deviation
        let stdDev = sqrt(variance)
        
        // Dynamic threshold: mean + 1.5*stddev
        let threshold = localAverage + 1.5 * stdDev
        
        // Check for beat
        let isBeat = currentEnergy > threshold
        
        // Prevent beats too close together (debounce)
        if isBeat && energyHistory.count >= 8 {
            // Check if we had a recent beat in the last 3 frames
            for i in 0..<min(3, energyHistory.count) {
                if energyHistory[energyHistory.count - 1 - i] > threshold {
                    return false // Too soon after previous beat
                }
            }
        }
        
        return isBeat
    }
    
    /// Information about detected pitch
    struct PitchInfo {
        /// Detected frequency in Hz
        var frequency: Float
        /// Confidence level (0-1)
        var confidence: Float
        /// Clarity of the pitch (0-1)
        var clarity: Float
    }
    
    /// Detect pitch in audio signal
    /// - Parameter samples: Audio samples
    /// - Returns: Pitch information
    private func detectPitch(_ samples: [Float]) throws -> PitchInfo {
        let count = samples.count
        
        // Use autocorrelation for pitch detection
        var autocorrelation = [Float](repeating: 0, count: count)
        
        // Compute autocorrelation
        vDSP_auto_correlation(samples, 1, &autocorrelation, 1, vDSP_Length(count))
        
        // Normalize autocorrelation
        let zerothLag = autocorrelation[0]
        if zerothLag > 0 {
            for i in 0..<count {
                autocorrelation[i] /= zerothLag
            }
        }
        
        // Find peaks in autocorrelation
        var peakIndices: [Int] = []
        var peakValues: [Float] = []
        
        // Skip first few lags to avoid detecting sub-harmonics
        let minLag = Int(Float(config.sampleRate) / 2000.0) // Corresponds to 2000 Hz (upper bound)
        let maxLag = Int(Float(config.sampleRate) / 50.0)   // Corresponds to 50 Hz (lower bound)
        
        for i in minLag..<min(maxLag, count-1) {
            if autocorrelation[i] > autocorrelation[i-1] && autocorrelation[i] > autocorrelation[i+1] {
                peakIndices.append(i)
                peakValues.append(autocorrelation[i])
            }
        }
        
        // If no peaks found, return no pitch
        if peakIndices.isEmpty {
            return PitchInfo(frequency: 0, confidence: 0, clarity: 0)
        }
        
        // Find the highest peak
        var maxPeakValue: Float = 0
        var maxPeakIndex = 0
        
        for i in 0..<peakValues.count {
            if peakValues[i] > maxPeakValue {
                maxPeakValue = peakValues[i]
                maxPeakIndex = i
            }
        }
        
        // Calculate frequency from lag
        let lag = peakIndices[maxPeakIndex]
        let frequency = Float(config.sampleRate) / Float(lag)
        
        // Calculate clarity as ratio between highest and second highest peak
        var clarity: Float = 1.0
        if peakValues.count > 1 {
            var secondHighest: Float = 0
            for value in peakValues {
                if value < maxPeakValue && value > secondHighest {
                    secondHighest = value
                }
            }
            
            if secondHighest > 0 {
                clarity = (maxPeakValue - secondHighest) / maxPeakValue
            }
        }
        
        // Calculate confidence based on peak value
        let confidence = maxPeakValue
        
        return PitchInfo(
            frequency: frequency,
            confidence: confidence,
            clarity: clarity
        )
    }
    
    /// Estimate tempo from beat history
    /// - Parameters:
    ///   - beatHistory: History of beat detection values (1.0 = beat, 0.0 = no beat)
    ///   - hopSize: Analysis hop size in samples
    ///   - sampleRate: Sample rate in Hz
    /// - Returns: Estimated tempo in BPM
    private func estimateTempo(beatHistory: [Float], hopSize: Int, sampleRate: Double) throws -> Float {
        // Ensure we have enough history
        guard beatHistory.count >= config.historyFrameCount else {
            throw AudioFeatureExtractorError.insufficientData("Not enough history for tempo estimation")
        }
        
        // Find beat onsets (indices where the value changes from 0 to 1)
        var beatIndices: [Int] = []
        for i in 1..<beatHistory.count {
            if beatHistory[i] > 0.5 && beatHistory[i-1] <= 0.5 {
                beatIndices.append(i)
            }
        }
        
        // If we don't have at least 2 beats, use default tempo
        if beatIndices.count < 2 {
            return 120.0 // Default tempo
        }
        
        // Calculate inter-beat intervals in frames
        var intervals: [Int] = []
        for i in 1..<beatIndices.count {
            intervals.append(beatIndices[i] - beatIndices[i-1])
        }
        
        // Calculate average interval
        var sum = 0
        for interval in intervals {
            sum += interval
        }
        let averageInterval = Float(sum) / Float(intervals.count)
        
        // Calculate time between frames
        let frameTime = Float(hopSize) / Float(sampleRate)
        
        // Convert to seconds
        let beatInterval = averageInterval * frameTime
        
        // Convert to BPM
        let bpm = 60.0 / beatInterval
        
        // Clamp to reasonable range
        return min(max(bpm, 60.0), 200.0)
    }
}

/// A FIFO buffer with fixed capacity
@available(macOS 15.0, *)
fileprivate class RingBuffer<T> {
    /// The internal storage
    private var buffer: [T] = []
    
    /// The maximum capacity
    private let capacity: Int
    
    /// The current count of items
    public var count: Int {
        return buffer.count
    }
    
    /// Whether the buffer is empty
    public var isEmpty: Bool {
        return buffer.isEmpty
    }
    
    /// Initialize a new ring buffer
    /// - Parameter capacity: The maximum capacity
    init(capacity: Int) {
        self.capacity = capacity
    }
    
    /// Add an item to the buffer
    /// - Parameter item: The item to add
    func add(_ item: T) {
        buffer.append(item)
        
        // If we exceed capacity, remove oldest items
        if buffer.count > capacity {
            buffer.removeFirst(buffer.count - capacity)
        }
    }
    
    /// Get the latest added item
    /// - Returns: The most recently added item, or nil if empty
    func latest() -> T? {
        return buffer.last
    }
    
    /// Get an item at a specific index (0 is oldest)
    /// - Parameter index: Index of the item to get
    /// - Returns: The item at the index, or nil if out of bounds
    func item(at index: Int) -> T? {
        guard index >= 0 && index < buffer.count else { return nil }
        return buffer[index]
    }
    
    /// Get all items in the buffer
    /// - Returns: Array of all items (oldest first)
    func allItems() -> [T] {
        return buffer
    }
    
    /// Clear the buffer
    func clear() {
        buffer.removeAll()
    }
}
