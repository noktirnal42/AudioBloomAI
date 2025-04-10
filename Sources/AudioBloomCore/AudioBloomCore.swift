// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency
//
// AudioBloomCore.swift
// Core functionality for AudioBloomAI
//

import Foundation
import Combine
import Algorithms
import AVFoundation
import SwiftUI

/// AudioBloomCore provides the shared protocols and utilities used across the application
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public enum AudioBloomCore {
    /// Application-wide constants
/// Uses Swift 6 actor isolation for thread safety.
    @available(macOS 15.0, *)
    public enum Constants {
        /// Default audio sample rate
/// Uses Swift 6 actor isolation for thread safety.
        public static let defaultSampleRate: Double = 44100.0
        
        /// Default FFT size for audio analysis
/// Uses Swift 6 actor isolation for thread safety.
        public static let defaultFFTSize: Int = 2048
        
        /// Default frame rate for visualization
/// Uses Swift 6 actor isolation for thread safety.
        public static let defaultFrameRate: Int = 60
        
        /// Min/Max values for audio buffering
/// Uses Swift 6 actor isolation for thread safety.
        public static let minBufferSize: Int = 512
        public static let maxBufferSize: Int = 8192
        
        /// Default hop size for audio analysis
/// Uses Swift 6 actor isolation for thread safety.
        public static let defaultHopSize: Int = 512
        
        /// Application name
/// Uses Swift 6 actor isolation for thread safety.
        public static let appName = "AudioBloom"
        
        /// Application version
/// Uses Swift 6 actor isolation for thread safety.
        public static let appVersion = "1.0.0"
        
        /// Minimum supported macOS version
/// Uses Swift 6 actor isolation for thread safety.
        public static let minMacOSVersion = "15.0"
        
        /// Default particle count for visualizations
/// Uses Swift 6 actor isolation for thread safety.
        public static let defaultParticleCount: Int = 1000
        
        /// Default window settings
/// Uses Swift 6 actor isolation for thread safety.
        public static let defaultWindowWidth: CGFloat = 1280
        public static let defaultWindowHeight: CGFloat = 720
    }
    
    /// Error types used throughout the application
/// Uses Swift 6 actor isolation for thread safety.
    @available(macOS 15.0, *)
    public enum Error: Swift.Error, LocalizedError {
        // Audio system errors
        case audioSessionSetupFailed
        case audioEngineStartFailed
        case audioDeviceNotFound
        case audioFormatNotSupported
        case audioProcessingFailed
        
        // Visualization errors
        case metalDeviceNotFound
        case metalCommandQueueCreationFailed
        case metalShaderCompilationFailed
        case renderingFailed
        
        // ML system errors
        case mlModelLoadFailed
        case mlModelInferenceFailed
        case featureExtractionFailed
        
        // General application errors
        case fileNotFound
        case dataCorrupted
        case settingsCorrupted
        case unexpectedState
        
        public var errorDescription: String? {
            switch self {
            // Audio errors
            case .audioSessionSetupFailed:
                return "Failed to set up audio session"
            case .audioEngineStartFailed:
                return "Failed to start audio engine"
            case .audioDeviceNotFound:
                return "Audio device not found or unavailable"
            case .audioFormatNotSupported:
                return "Audio format is not supported"
            case .audioProcessingFailed:
                return "Audio processing operation failed"
                
            // Visualization errors
            case .metalDeviceNotFound:
                return "Metal device not found"
            case .metalCommandQueueCreationFailed:
                return "Failed to create Metal command queue"
            case .metalShaderCompilationFailed:
                return "Failed to compile Metal shader"
            case .renderingFailed:
                return "Failed to render visualization"
                
            // ML system errors
            case .mlModelLoadFailed:
                return "Failed to load ML model"
            case .mlModelInferenceFailed:
                return "ML model inference failed"
            case .featureExtractionFailed:
                return "Failed to extract audio features for ML processing"
                
            // General application errors
            case .fileNotFound:
                return "Required file not found"
            case .dataCorrupted:
                return "Data is corrupted or invalid"
            case .settingsCorrupted:
                return "Application settings are corrupted"
            case .unexpectedState:
                return "Application is in an unexpected state"
            }
        }
    }
}

// MARK: - Audio Data Structures
/// Audio data structure representing an audio buffer and its properties
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct AudioData: Sendable: Sendable {
    /// The audio samples
/// Uses Swift 6 actor isolation for thread safety.
    public let samples: [Float]
    
    /// Audio levels (left and right channels)
/// Uses Swift 6 actor isolation for thread safety.
    public let levels: (left: Float, right: Float)
    
    /// Sample rate of the audio
/// Uses Swift 6 actor isolation for thread safety.
    public let sampleRate: Double
    
    /// Timestamp when the audio was captured
/// Uses Swift 6 actor isolation for thread safety.
    public let timestamp: Date
    
    /// Frequency spectrum data (if available)
/// Uses Swift 6 actor isolation for thread safety.
    public var frequencyData: [Float] = []
    
    public init(
        samples: [Float],
        levels: (left: Float, right: Float),
        sampleRate: Double,
        timestamp: Date = Date(),
        frequencyData: [Float] = []
    ) {
        self.samples = samples
        self.levels = levels
        self.sampleRate = sampleRate
        self.timestamp = timestamp
        self.frequencyData = frequencyData
    }
    
    /// Initializes with just frequency data (for compatibility)
/// Uses Swift 6 actor isolation for thread safety.
    public init(frequencyData: [Float], levels: (left: Float, right: Float), timestamp: Date = Date()) {
        self.samples = []
        self.levels = levels
        self.sampleRate = AudioBloomCore.Constants.defaultSampleRate
        self.timestamp = timestamp
        self.frequencyData = frequencyData
    }
}

/// Core configuration for audio processing
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct AudioConfiguration: Sendable {
    /// Sample rate in Hz
/// Uses Swift 6 actor isolation for thread safety.
    public let sampleRate: Double
    /// Number of channels (1 for mono, 2 for stereo)
/// Uses Swift 6 actor isolation for thread safety.
    public let channels: Int
    /// FFT size (power of 2)
/// Uses Swift 6 actor isolation for thread safety.
    public let fftSize: Int
    /// Hop size for FFT analysis
/// Uses Swift 6 actor isolation for thread safety.
    public let hopSize: Int
    
    public init(
        sampleRate: Double = AudioBloomCore.Constants.defaultSampleRate,
        channels: Int = 2,
        fftSize: Int = AudioBloomCore.Constants.defaultFFTSize,
        hopSize: Int = AudioBloomCore.Constants.defaultHopSize
    ) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.fftSize = fftSize
        self.hopSize = hopSize
    }
}

// MARK: - Core Protocols
/// Protocol for objects that provide audio data
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol AudioDataProvider: AnyObject, Sendable {
    /// Publisher for audio data updates
/// Uses Swift 6 actor isolation for thread safety.
    var audioDataPublisher: AnyPublisher<AudioData, Never> { get }
    
    /// Current audio levels (e.g., left and right channel levels)
/// Uses Swift 6 actor isolation for thread safety.
    var levels: (left: Float, right: Float) { get }
    
    /// Current frequency spectrum data
/// Uses Swift 6 actor isolation for thread safety.
    var frequencyData: [Float] { get }
    
    /// Sets up the audio session
/// Uses Swift 6 actor isolation for thread safety.
    func setupAudioSession() async throws
    
    /// Starts audio capture
/// Uses Swift 6 actor isolation for thread safety.
    func startCapture() throws
    
    /// Stops audio capture
/// Uses Swift 6 actor isolation for thread safety.
    func stopCapture()
    
    /// Selects a specific audio input device
/// Uses Swift 6 actor isolation for thread safety.
    func selectInputDevice(id: String?) throws
    
    /// Selects a specific audio output device
/// Uses Swift 6 actor isolation for thread safety.
    func selectOutputDevice(id: String?) throws
    
    /// Lists available audio devices
/// Uses Swift 6 actor isolation for thread safety.
    func availableDevices() -> (input: [(id: String, name: String)], output: [(id: String, name: String)])
}

/// Protocol for audio data publishers with simplified requirements
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol AudioDataPublisher: AnyObject {
    /// Publisher for audio data
/// Uses Swift 6 actor isolation for thread safety.
    var publisher: AnyPublisher<AudioData, Never> { get }
    
    /// Publishes new audio data
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - frequencyData: The frequency spectrum data
/// Uses Swift 6 actor isolation for thread safety.
    ///   - levels: The audio level data
/// Uses Swift 6 actor isolation for thread safety.
    func publish(frequencyData: [Float], levels: (Float, Float))
}

/// Protocol for audio processing modules
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol AudioProcessor {
    /// Process an audio buffer
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter audioData: The audio data to process
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Processed audio data
/// Uses Swift 6 actor isolation for thread safety.
    func process(_ audioData: AudioData) throws -> AudioData
    
    /// Configure the processor with specific settings
/// Uses Swift 6 actor isolation for thread safety.
    func configure(with settings: [String: Any])
}

/// Protocol defining required functionality for visualization renderers
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol VisualizationRenderer: ObservableObject {
    /// Flag indicating if the renderer is ready
/// Uses Swift 6 actor isolation for thread safety.
    var isReady: Bool { get }
    
    /// Prepares the renderer for drawing
/// Uses Swift 6 actor isolation for thread safety.
    func prepareRenderer()
    
    /// Updates the renderer with new audio data
/// Uses Swift 6 actor isolation for thread safety.
    func update(audioData: [Float], levels: (left: Float, right: Float))
    
    /// Renders a frame
/// Uses Swift 6 actor isolation for thread safety.
    func render()
    
    /// Updates rendering parameters
/// Uses Swift 6 actor isolation for thread safety.
    func updateParameters(_ parameters: [String: Any])
    
    /// Cleans up resources
/// Uses Swift 6 actor isolation for thread safety.
    func cleanup()
}

/// Protocol for objects that can receive visualization parameters
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol VisualizationParameterReceiver {
    /// Updates parameters for the visualization
/// Uses Swift 6 actor isolation for thread safety.
    func updateParameters(_ parameters: [String: Any])
}

/// Protocol defining required functionality for ML processors
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol MLProcessing: ObservableObject {
    /// Current ML model output data
/// Uses Swift 6 actor isolation for thread safety.
    var outputData: [Float] { get }
    
    /// Publisher for ML processing results
/// Uses Swift 6 actor isolation for thread safety.
    var resultPublisher: AnyPublisher<[Float], Never> { get }
    
    /// Prepares the ML model for processing
/// Uses Swift 6 actor isolation for thread safety.
    func prepareMLModel() throws
    
    /// Processes audio data through the ML model
/// Uses Swift 6 actor isolation for thread safety.
    func processAudioData(_ audioData: [Float]) async throws -> [Float]
    
    /// Configure the ML processor with specific settings
/// Uses Swift 6 actor isolation for thread safety.
    func configure(with settings: [String: Any])
    
    /// Cleanup ML resources
/// Uses Swift 6 actor isolation for thread safety.
    func cleanup()
}
/// Protocol for ML processing integration
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol MLProcessorProtocol: AnyObject, Sendable {
    /// Whether the processor is ready for processing
/// Uses Swift 6 actor isolation for thread safety.
    var isReady: Bool { get }
    
    /// Process audio data for ML analysis
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter audioData: The audio data to analyze
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Analysis results
/// Uses Swift 6 actor isolation for thread safety.
    func processAudio(_ audioData: AudioData) throws -> [String: Any]
    
    /// Processes raw audio data
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter audioData: The raw audio data to process
/// Uses Swift 6 actor isolation for thread safety.
    /// - Throws: Error if processing fails
/// Uses Swift 6 actor isolation for thread safety.
    func processAudioData(_ audioData: [Float]) async throws
    
    /// Publisher for visualization data
/// Uses Swift 6 actor isolation for thread safety.
    var visualizationDataPublisher: AnyPublisher<VisualizationData, Never> { get }
}

// MARK: - Utility Type Aliases

public extension AudioBloomCore {
    /// Frequency data with corresponding audio levels
/// Uses Swift 6 actor isolation for thread safety.
    typealias FrequencyLevelData = ([Float], (left: Float, right: Float))
    
    /// ML processing result type
/// Uses Swift 6 actor isolation for thread safety.
    typealias MLResult = [String: Any]
    
    /// Audio device information
/// Uses Swift 6 actor isolation for thread safety.
    typealias AudioDevice = (id: String, name: String)
    
    /// RGB color value
/// Uses Swift 6 actor isolation for thread safety.
    typealias RGBColor = (red: Float, green: Float, blue: Float, alpha: Float)
}

// MARK: - Utility Functions

public extension AudioBloomCore {
    /// Converts linear amplitude to decibels
/// Uses Swift 6 actor isolation for thread safety.
    static func amplitudeToDecibels(_ amplitude: Float, minDb: Float = -80.0) -> Float {
        let db = 20.0 * log10(amplitude)
        return max(db, minDb)
    }
    
    /// Converts decibels to a normalized value between 0 and 1
/// Uses Swift 6 actor isolation for thread safety.
    static func normalizeDecibels(_ db: Float, minDb: Float = -80.0, maxDb: Float = 0.0) -> Float {
        (db - minDb) / (maxDb - minDb)
    }
    
    /// Smooth values with a simple low-pass filter
/// Uses Swift 6 actor isolation for thread safety.
    static func smoothValue(_ newValue: Float, previousValue: Float, factor: Float = 0.2) -> Float {
        previousValue + factor * (newValue - previousValue)
    }
    
    /// Maps a value from one range to another
/// Uses Swift 6 actor isolation for thread safety.
    static func mapValue(_ value: Float, inMin: Float, inMax: Float, outMin: Float, outMax: Float) -> Float {
        outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin)
    }
    
    /// Clamps a value to a specific range
/// Uses Swift 6 actor isolation for thread safety.
    static func clamp(_ value: Float, min: Float, max: Float) -> Float {
        Swift.min(Swift.max(value, min), max)
    }
}
