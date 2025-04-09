import Foundation
import Combine
import SwiftUI
import Accelerate
import AVFoundation
import AudioBloomCore
import AudioBloomCore.Audio
/// Protocol for providing audio data to visualizers
public protocol AudioDataProvider {
    /// Setup audio session for capture
    func setupAudioSession() async throws
    
    /// Start audio capture
    func startCapture() throws
    
    /// Stop audio capture
    func stopCapture()
}

/// Struct for containing audio data to be visualized
public struct AudioVisualizationData {
    /// Frequency data normalized to 0.0-1.0 range
    public var frequencyData: [Float]
    
    /// Audio levels for left and right channels
    public var levels: (left: Float, right: Float)
    
    /// Create a default instance with empty data
    public static var empty: AudioVisualizationData {
        return AudioVisualizationData(
            frequencyData: Array(repeating: 0.0, count: 512),
            levels: (left: 0.0, right: 0.0)
        )
    }
}

/// Processor that connects the AudioFeatureExtractor to visualization systems
@available(macOS 15.0, *)
public class AudioVisualizerProcessor: NSObject, AudioDataProvider, AudioFeatureExtractorDelegate {
    // MARK: - Properties
    
    /// The audio feature extractor
    private let featureExtractor: AudioFeatureExtractor
    
    /// The audio pipeline core
    private let audioPipeline: AudioPipelineCore
    
    /// Configuration for visualization data
    private let config: VisualizationConfig
    
    /// Audio session
    private let audioSession = AVAudioSession.sharedInstance()
    
    /// Subject for publishing audio data to visualizers
    private let audioDataSubject = CurrentValueSubject<AudioVisualizationData, Never>(.empty)
    
    /// Flag indicating whether audio capture is active
    private var isCapturing = false
    
    /// Lock for thread-safe operations
    private let lock = NSLock()
    
    /// Frequency band energy data
    private var bandEnergies: [Float] = []
    
    /// FFT magnitude spectrum data
    private var fftMagnitude: [Float] = []
    
    /// Beat detection status
    private var beatDetected: Bool = false
    
    /// Audio levels
    private var audioLevels: (left: Float, right: Float) = (0, 0)
    
    /// Configuration for visualization processing
    public struct VisualizationConfig {
        /// Number of frequency bands for visualization (default 512)
        public var frequencyBandCount: Int
        
        /// Smoothing factor for visualization data (0.0-1.0)
        public var smoothingFactor: Float
        
        /// Minimum frequency to visualize
        public var minFrequency: Float
        
        /// Maximum frequency to visualize
        public var maxFrequency: Float
        
        /// Default configuration
        public static var standard: VisualizationConfig {
            return VisualizationConfig(
                frequencyBandCount: 512,
                smoothingFactor: 0.3,
                minFrequency: 20,
                maxFrequency: 20000
            )
        }
    }
    
    // MARK: - Initialization
    
    /// Initialize with existing audio pipeline and feature extractor
    /// - Parameters:
    ///   - audioPipeline: The audio pipeline core
    ///   - featureExtractor: The audio feature extractor
    ///   - config: Configuration for visualization
    public init(audioPipeline: AudioPipelineCore, 
                featureExtractor: AudioFeatureExtractor,
                config: VisualizationConfig = .standard) {
        self.audioPipeline = audioPipeline
        self.featureExtractor = featureExtractor
        self.config = config
        
        // Initialize storage arrays
        self.fftMagnitude = Array(repeating: 0.0, count: config.frequencyBandCount)
        self.bandEnergies = Array(repeating: 0.0, count: 5) // Bass, low-mid, mid, high-mid, treble
        
        super.init()
        
        // Set self as delegate for feature extractor
        self.featureExtractor.delegate = self
        
        // Configure feature extractor
        setupFeatureExtractor()
    }
    
    /// Initialize with new audio pipeline and feature extractor
    /// - Parameter config: Configuration for visualization
    public convenience init(config: VisualizationConfig = .standard) throws {
        // Create audio pipeline
        let audioPipeline = try AudioPipelineCore()
        
        // Create feature extractor with default configuration
        let featureExtractorConfig = AudioFeatureExtractorConfiguration.defaultConfiguration(
            sampleRate: audioPipeline.format.sampleRate
        )
        
        let featureExtractor = try AudioFeatureExtractor(
            config: featureExtractorConfig, 
            audioPipeline: audioPipeline
        )
        
        self.init(
            audioPipeline: audioPipeline,
            featureExtractor: featureExtractor,
            config: config
        )
    }
    
    // MARK: - Setup
    
    /// Configure the feature extractor for visualization
    private func setupFeatureExtractor() {
        do {
            // Set active features for visualization
            try featureExtractor.setActiveFeatures([
                .fftMagnitude,         // For spectrum visualization
                .frequencyBandEnergy,  // For band energy visualization
                .beat,                 // For beat detection
                .rmsEnergy             // For overall audio level
            ])
        } catch {
            print("Failed to set active features: \(error.localizedDescription)")
        }
    }
    
    // MARK: - AudioDataProvider Protocol Implementation
    
    /// Setup audio session for capture
    public func setupAudioSession() async throws {
        try await audioSession.setCategory(.playAndRecord, mode: .default, options: [.mixWithOthers, .defaultToSpeaker])
        try await audioSession.setActive(true)
    }
    
    /// Start audio capture and feature extraction
    public func startCapture() throws {
        guard !isCapturing else { return }
        
        // Start feature extraction
        try featureExtractor.start()
        
        // Start audio pipeline if needed
        if !audioPipeline.isRunning {
            try audioPipeline.start()
        }
        
        isCapturing = true
    }
    
    /// Stop audio capture
    public func stopCapture() {
        guard isCapturing else { return }
        
        // Stop feature extraction
        featureExtractor.stop()
        
        // Reset data
        lock.lock()
        fftMagnitude = Array(repeating: 0.0, count: config.frequencyBandCount)
        bandEnergies = Array(repeating: 0.0, count: 5)
        beatDetected = false
        audioLevels = (0, 0)
        lock.unlock()
        
        // Publish empty data
        audioDataSubject.send(.empty)
        
        isCapturing = false
    }
    
    // MARK: - AudioFeatureExtractorDelegate Implementation
    
    /// Handle extracted audio features
    public func featureExtractor(_ extractor: AudioFeatureExtractor, didExtract features: AudioFeatures) {
        lock.lock()
        defer { lock.unlock() }
        
        switch features.featureType {
        case .fftMagnitude:
            // Process FFT magnitude data for visualization
            processFFTMagnitude(features.values)
            
        case .frequencyBandEnergy:
            // Store band energies
            bandEnergies = features.values
            
            // Update audio levels based on band energies
            updateAudioLevels()
            
        case .beat:
            // Update beat detection status
            beatDetected = features.values.first ?? 0 > 0.5
            
        case .rmsEnergy:
            // Use RMS energy for overall level
            let energy = features.values.first ?? 0
            // Simple stereo simulation for visualization
            audioLevels = (energy, energy * (0.8 + 0.4 * Float.random(in: 0...1)))
            
        default:
            break
        }
        
        // After processing new data, update the visualization data
        updateVisualizationData()
    }
    
    /// Handle feature extraction errors
    public func featureExtractor(_ extractor: AudioFeatureExtractor, didFailExtractingFeature featureType: AudioFeatureType, withError error: Error) {
        print("Feature extraction error for \(featureType.rawValue): \(error.localizedDescription)")
    }
    
    // MARK: - Data Processing
    
    /// Process FFT magnitude data for visualization
    private func processFFTMagnitude(_ rawMagnitude: [Float]) {
        // Scale the raw FFT data to the desired visualization band count
        let sourceCount = rawMagnitude.count
        let targetCount = config.frequencyBandCount
        
        if sourceCount == targetCount {
            // No scaling needed
            fftMagnitude = rawMagnitude
        } else {
            // Resample to match the desired frequency band count
            let tempMagnitude = Array(repeating: Float(0), count: targetCount)
            
            for i in 0..<targetCount {
                // Calculate the corresponding index in the source data
                let sourceIndex = Int(Float(i) * Float(sourceCount) / Float(targetCount))
                if sourceIndex < sourceCount {
                    // Apply logarithmic scaling to emphasize lower frequencies
                    let logFactor: Float = 1.0 - (log10(Float(i+1)) / log10(Float(targetCount)))
                    let scaledValue = rawMagnitude[sourceIndex] * logFactor
                    
                    // Apply smoothing with previous value if enabled
                    if config.smoothingFactor > 0 && i < fftMagnitude.count {
                        let smoothFactor = config.smoothingFactor
                        tempMagnitude[i] = (smoothFactor * fftMagnitude[i]) + ((1 - smoothFactor) * scaledValue)
                    } else {
                        tempMagnitude[i] = scaledValue
                    }
                }
            }
            
            // Update the magnitude array
            fftMagnitude = tempMagnitude
        }
        
        // Ensure normalization to 0.0-1.0 range
        if let maxValue = fftMagnitude.max(), maxValue > 0 {
            for i in 0..<fftMagnitude.count {
                fftMagnitude[i] /= maxValue
            }
        }
    }
    
    /// Update audio levels based on band energies
    private func updateAudioLevels() {
        guard bandEnergies.count >= 5 else { return }
        
        // Calculate rough average for visualization
        let bass = bandEnergies[0]
        let lowMid = bandEnergies[1]
        let mid = bandEnergies[2]
        let highMid = bandEnergies[3]
        let treble = bandEnergies[4]
        
        // Create stereo effect for visualization
        let leftEmphasis = bass * 0.7 + mid * 0.2 + treble * 0.1
        let rightEmphasis = bass * 0.1 + mid * 0.2 + treble * 0.7
        
        // Update audio levels
        audioLevels = (
            left: max(audioLevels.left, leftEmphasis),
            right: max(audioLevels.right, rightEmphasis)
        )
    }
    
    /// Update visualization data and publish to subscribers
    private func updateVisualizationData() {
        // Create visualization data
        let visualizationData = AudioVisualizationData(
            frequencyData: fftMagnitude,
            levels: audioLevels
        )
        
        // Publish data
        audioDataSubject.send(visualizationData)
    }
    
    // MARK: - Public Interface
    
    /// Get publisher for audio data
    /// - Returns: AudioVisualizationData publisher
    public func getAudioDataPublisher() -> CurrentValueSubject<AudioVisualizationData, Never> {
        return audioDataSubject
    }
    
    /// Check if a beat was detected
    /// - Returns: True if a beat was detected
    public var isBeatDetected: Bool {
        lock.lock()
        defer { lock.unlock() }
        return beatDetected
    }
    
    /// Get the current frequency band energy values
    /// - Returns: Array of band energy values
    public var frequencyBandEnergies: [Float] {
        lock.lock()
        defer { lock.unlock() }
        return bandEnergies
    }
    
    /// Get the current audio levels
    /// - Returns: Audio levels for left and right channels
    public var currentAudioLevels: (left: Float, right: Float) {
        lock.lock()
        defer { lock.unlock() }
        return audioLevels
    }
}

/// Extension to AudioVisualizerProcessor to conform to AudioEngine interface
/// for backward compatibility with existing code
@available(macOS 15.0, *)
extension AudioVisualizerProcessor {
    /// Get the AudioEngine-compatible publisher
    /// - Returns: Publisher for audio data
    public func getAudioDataPublisher() -> (CurrentValueSubject<AudioVisualizationData, Never>, VisualizationConfig) {
        return (audioDataSubject, config)
    }
}

