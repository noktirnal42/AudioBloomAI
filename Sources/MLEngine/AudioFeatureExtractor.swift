import Foundation
import SoundAnalysis
import CoreML
import CoreAudio
import AVFoundation
import Combine
import Logging

/// Errors that can occur during audio feature extraction
public enum AudioFeatureExtractorError: Error {
    case analysisEngineCreationFailed
    case audioFormatError
    case modelNotFound
    case modelCompilationFailed
    case observerRegistrationFailed
    case soundAnalysisError(Error)
    case bufferProcessingError
    case neuralEngineUnavailable
    
    var localizedDescription: String {
        switch self {
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

/// Audio features extracted from audio input
public struct AudioFeatures {
    /// Frequency spectrum data (normalized 0-1)
    public var frequencySpectrum: [Float] = []
    
    /// Detected tempo in BPM
    public var tempo: Float = 0.0
    
    /// Beat confidence (0-1)
    public var beatConfidence: Float = 0.0
    
    /// Whether a beat occurred in this frame
    public var beatDetected: Bool = false
    
    /// Audio energy level (0-1)
    public var energyLevel: Float = 0.0
    
    /// Detected audio patterns or classifications
    public var classifications: [String: Float] = [:]
    
    /// Timestamp of the analysis
    public var timestamp: TimeInterval = 0.0
}

/// Protocol for receiving audio feature extraction events
public protocol AudioFeatureExtractorDelegate: AnyObject {
    /// Called when new audio features are extracted
    func didExtractFeatures(_ features: AudioFeatures)
    
    /// Called when an error occurs during extraction
    func didEncounterError(_ error: AudioFeatureExtractorError)
}

/// Handles audio feature extraction using SoundAnalysis and CoreML
public class AudioFeatureExtractor {
    /// The logger for this class
    private let logger = Logger(label: "com.audiobloom.audiofeatureextractor")
    
    /// The sound analysis engine
    private var audioAnalyzer: SNAudioStreamAnalyzer?
    
    /// Neural Engine optimized ML model for audio analysis
    private var audioAnalysisModel: MLModel?
    
    /// The current audio format
    private var audioFormat: AVAudioFormat?
    
    /// Delegate to receive extraction events
    public weak var delegate: AudioFeatureExtractorDelegate?
    
    /// Whether the extractor is currently running
    private(set) public var isRunning = false
    
    /// Subject for streaming frequency data
    private let frequencySubject = PassthroughSubject<[Float], Never>()
    
    /// Publisher for frequency data
    public var frequencyPublisher: AnyPublisher<[Float], Never> {
        frequencySubject.eraseToAnyPublisher()
    }
    
    /// Dispatch queue for audio processing
    private let processingQueue = DispatchQueue(label: "com.audiobloom.featureextraction", qos: .userInteractive)
    
    /// Task to manage async audio analysis
    private var analysisTask: Task<Void, Never>?
    
    /// Initializes a new AudioFeatureExtractor
    public init() {
        logger.info("AudioFeatureExtractor initialized")
    }
    
    /// Prepares the audio analysis engine with the specified audio format
    /// - Parameter format: The audio format to analyze
    /// - Throws: AudioFeatureExtractorError if preparation fails
    public func prepare(with format: AVAudioFormat) async throws {
        logger.info("Preparing audio feature extractor with format: \(format)")
        
        self.audioFormat = format
        
        // Create a new audio analyzer with the specified format
        self.audioAnalyzer = SNAudioStreamAnalyzer(format: format)
        
        guard let analyzer = self.audioAnalyzer else {
            logger.error("Failed to create audio analyzer")
            throw AudioFeatureExtractorError.analysisEngineCreationFailed
        }
        
        // Attempt to load and compile the audio analysis model
        try await loadAudioAnalysisModel()
        
        // Register sound analysis requests
        try registerSoundAnalysisRequests(with: analyzer)
        
        logger.info("Audio feature extractor prepared successfully")
    }
    
    /// Loads and compiles the audio analysis CoreML model
    /// - Throws: AudioFeatureExtractorError if model loading or compilation fails
    private func loadAudioAnalysisModel() async throws {
        logger.info("Loading audio analysis model")
        
        // In a real implementation, you would load your specific CoreML model
        // For now, we'll use a placeholder approach
        
        #if ENABLE_NEURAL_ENGINE
        // Configure ML model with Neural Engine optimizations
        let config = MLModelConfiguration()
        config.computeUnits = .all // Use Neural Engine when available
        
        // Check for model file - in a real app, you would bundle your .mlmodel file
        // and use Bundle.module.url(forResource:withExtension:) to locate it
        guard let modelURL = Bundle.module.url(forResource: "AudioAnalysisModel", withExtension: "mlmodelc") else {
            // For testing, we'll allow falling back to a simulated model
            logger.warning("Model file not found, using fallback")
            return
        }
        
        do {
            // Compile and load the model
            self.audioAnalysisModel = try MLModel(contentsOf: modelURL, configuration: config)
            logger.info("Audio analysis model loaded successfully with Neural Engine optimizations")
        } catch {
            logger.error("Failed to compile model: \(error)")
            throw AudioFeatureExtractorError.modelCompilationFailed
        }
        #else
        logger.info("Neural Engine optimizations disabled")
        #endif
    }
    
    /// Registers sound analysis requests with the analyzer
    /// - Parameter analyzer: The audio stream analyzer
    /// - Throws: AudioFeatureExtractorError if registration fails
    private func registerSoundAnalysisRequests(with analyzer: SNAudioStreamAnalyzer) throws {
        logger.debug("Registering sound analysis requests")
        
        // Register frequency analysis
        if let request = try? SNSpectrumRequest(maximumFrequency: 22000) {
            let observer = SpectrumAnalysisObserver(extractor: self)
            do {
                try analyzer.add(request, withObserver: observer)
                logger.debug("Registered spectrum analysis request")
            } catch {
                logger.error("Failed to add spectrum request: \(error)")
                throw AudioFeatureExtractorError.observerRegistrationFailed
            }
        }
        
        // Register tempo detection
        if let request = try? SNTempoRequest() {
            let observer = TempoAnalysisObserver(extractor: self)
            do {
                try analyzer.add(request, withObserver: observer)
                logger.debug("Registered tempo detection request")
            } catch {
                logger.error("Failed to add tempo request: \(error)")
                throw AudioFeatureExtractorError.observerRegistrationFailed
            }
        }
    }
    
    /// Processes an audio buffer for feature extraction
    /// - Parameter buffer: The audio buffer to analyze
    /// - Throws: AudioFeatureExtractorError if processing fails
    public func process(buffer: AVAudioPCMBuffer) async throws {
        guard isRunning, let analyzer = audioAnalyzer else { return }
        
        do {
            // Process the buffer through the sound analyzer
            try analyzer.analyze(buffer, atAudioFramePosition: 0)
            
            // If we have a CoreML model, also process through it
            if let model = audioAnalysisModel {
                // Process with CoreML model would go here
                // This would be implemented based on your specific model
            }
        } catch {
            logger.error("Error processing audio buffer: \(error)")
            throw AudioFeatureExtractorError.bufferProcessingError
        }
    }
    
    /// Starts the audio feature extraction
    public func start() {
        guard !isRunning else { return }
        
        logger.info("Starting audio feature extraction")
        isRunning = true
        
        // Create an async task to manage the extraction process
        analysisTask = Task {
            // Continuous processing would happen here
            // In a real implementation, this would be driven by incoming audio buffers
        }
    }
    
    /// Stops the audio feature extraction
    public func stop() {
        guard isRunning else { return }
        
        logger.info("Stopping audio feature extraction")
        isRunning = false
        
        // Cancel the analysis task
        analysisTask?.cancel()
        analysisTask = nil
        
        // Reset the analyzer
        audioAnalyzer = nil
    }
    
    /// Called when frequency spectrum data is received
    /// - Parameters:
    ///   - frequencies: The frequency data
    ///   - timestamp: The timestamp of the analysis
    internal func didReceiveFrequencyData(_ frequencies: [Float], at timestamp: TimeInterval) {
        var features = AudioFeatures()
        features.frequencySpectrum = frequencies
        features.timestamp = timestamp
        
        // Publish to combine pipeline
        frequencySubject.send(frequencies)
        
        // Notify delegate
        delegate?.didExtractFeatures(features)
    }
    
    /// Called when tempo data is received
    /// - Parameters:
    ///   - tempo: The detected tempo in BPM
    ///   - timestamp: The timestamp of the analysis
    internal func didReceiveTempo(_ tempo: Float, at timestamp: TimeInterval) {
        var features = AudioFeatures()
        features.tempo = tempo
        features.timestamp = timestamp
        
        // Notify delegate
        delegate?.didExtractFeatures(features)
    }
    
    /// Called when a beat is detected
    /// - Parameters:
    ///   - confidence: The confidence of the beat detection (0-1)
    ///   - timestamp: The timestamp of the analysis
    internal func didDetectBeat(confidence: Float, at timestamp: TimeInterval) {
        var features = AudioFeatures()
        features.beatDetected = true
        features.beatConfidence = confidence
        features.timestamp = timestamp
        
        // Notify delegate
        delegate?.didExtractFeatures(features)
    }
}

// MARK: - Sound Analysis Observers

/// Observer for spectrum analysis results
private class SpectrumAnalysisObserver: NSObject, SNSpectrumAnalyzing {
    weak var extractor: AudioFeatureExtractor?
    
    init(extractor: AudioFeatureExtractor) {
        self.extractor = extractor
        super.init()
    }
    
    func spectrum(_ spectrum: SNSpectrogram) {
        // Convert the spectrum data to our format
        let frequencies = spectrum.significantFrequencies.map { Float($0) }
        extractor?.didReceiveFrequencyData(frequencies, at: spectrum.frameDuration)
    }
}

/// Observer for tempo analysis results
private class TempoAnalysisObserver: NSObject, SNTempoObserver {
    weak var extractor: AudioFeatureExtractor?
    
    init(extractor: AudioFeatureExtractor) {
        self.extractor = extractor
        super.init()
    }
    
    func tempo(_ tempo: SNTempoResult) {
        extractor?.didReceiveTempo(Float(tempo.tempo), at: tempo.timeRange.start.seconds)
        
        // Check if this is a beat onset
        if tempo.confidence > 0.5 {
            extractor?.didDetectBeat(confidence: Float(tempo.confidence), at: tempo.timeRange.start.seconds)
        }
    }
}

