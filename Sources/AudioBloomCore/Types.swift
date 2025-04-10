// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency
// Types.swift
// Shared type definitions for AudioBloomAI
//

import Foundation
import Combine
import AVFoundation
import CoreAudio
import CoreML
import QuartzCore
import MetalKit
import SwiftUI
import Accelerate
// MARK: - AudioBridge Settings

/// Settings for audio bridge configuration
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct AudioBridgeSettings: Sendable: Sendable {
    /// Whether to apply Neural Engine optimizations
/// Uses Swift 6 actor isolation for thread safety.
    public var useNeuralEngine: Bool = true
    
    /// Optimization level for processing
/// Uses Swift 6 actor isolation for thread safety.
    public var optimizationLevel: OptimizationLevel = .balanced
    
    /// Buffer size for audio processing
/// Uses Swift 6 actor isolation for thread safety.
    public var bufferSize: Int = 
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
        self.sampleRate = 44100 // default
        self.timestamp = timestamp
        self.frequencyData = frequencyData
    }
}

// MARK: - Provider Protocols

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
}

/// Protocol for audio data publishers with simplified requirements
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public protocol AudioDataPublisher: AnyObject, Sendable {
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

// MARK: - Visualization Types

/// Data structure for visualization
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct VisualizationData: Sendable: Sendable, Equatable {
    /// Values to visualize (typically frequency spectrum)
/// Uses Swift 6 actor isolation for thread safety.
    public let values: [Float]
    
    /// Whether this data represents a significant event (like a beat)
/// Uses Swift 6 actor isolation for thread safety.
    public let isSignificantEvent: Bool
    
    /// Timestamp of the data
/// Uses Swift 6 actor isolation for thread safety.
    public let timestamp: Date
    
    /// Initializes a new visualization data instance
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - values: Values to visualize
/// Uses Swift 6 actor isolation for thread safety.
    ///   - isSignificantEvent: Whether this is a significant event
/// Uses Swift 6 actor isolation for thread safety.
    ///   - timestamp: Timestamp of the data
/// Uses Swift 6 actor isolation for thread safety.
    public init(values: [Float], isSignificantEvent: Bool = false, timestamp: Date = Date()) {
        self.values = values
        self.isSignificantEvent = isSignificantEvent
        self.timestamp = timestamp
    }
}

/// Enumeration of available visualization modes
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public enum VisualizationMode: String, CaseIterable, Identifiable, Sendable {
    case spectrum = "Spectrum"
    case waveform = "Waveform"
    case particles = "Particles"
    case neural = "Neural"
    
    public var id: String { rawValue }
    
    /// Get a user-friendly description of the mode
/// Uses Swift 6 actor isolation for thread safety.
    public var description: String {
        switch self {
        case .spectrum:
            return "Frequency Spectrum"
        case .waveform:
            return "Waveform Oscilloscope"
        case .particles:
            return "Particle System"
        case .neural:
            return "Neural Pattern"
        }
    }
}

// MARK: - ML Processing Types

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

/// Optimization level for audio processing
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public enum OptimizationLevel: String, Sendable {
    /// Balance efficiency and quality
/// Uses Swift 6 actor isolation for thread safety.
    case balanced
    
    /// Prioritize quality (higher power usage)
/// Uses Swift 6 actor isolation for thread safety.
    case quality
    
    /// Prioritize efficiency (lower power usage)
/// Uses Swift 6 actor isolation for thread safety.
    case efficiency
}

/// Neural processor response for visualization
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct NeuralProcessingResponse: Sendable: Equatable, Sendable {
    /// Energy level (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var energy: Float = 0.0
    
    /// Pleasantness (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var pleasantness: Float = 0.5
    
    /// Complexity (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var complexity: Float = 0.5
    
    /// Pattern type (0-3)
/// Uses Swift 6 actor isolation for thread safety.
    public var patternType: Int = 0
    
    /// Beat confidence (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var beatConfidence: Float = 0.0
    
    /// Default initialization
/// Uses Swift 6 actor isolation for thread safety.
    public init() {}
    
    /// Full initialization
/// Uses Swift 6 actor isolation for thread safety.
    public init(energy: Float, pleasantness: Float, complexity: Float, patternType: Int = 0, beatConfidence: Float = 0.0) {
        self.energy = energy
        self.pleasantness = pleasantness
        self.complexity = complexity
        self.patternType = patternType
        self.beatConfidence = beatConfidence
    }
}

// MARK: - Configuration Types

/// Framework configuration
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct Configuration: Sendable: Equatable, Sendable {
    /// Audio processor configuration
/// Uses Swift 6 actor isolation for thread safety.
    public var audioProcessor: AudioProcessorConfiguration
    
    /// ML engine configuration
/// Uses Swift 6 actor isolation for thread safety.
    public var mlEngine: MLEngineConfiguration
    
    /// Visualization configuration
/// Uses Swift 6 actor isolation for thread safety.
    public var visualization: VisualizationConfiguration
    
    /// Performance configuration
/// Uses Swift 6 actor isolation for thread safety.
    public var performance: PerformanceConfiguration
    
    /// Creates a default configuration
/// Uses Swift 6 actor isolation for thread safety.
    public init() {
        self.audioProcessor = AudioProcessorConfiguration()
        self.mlEngine = MLEngineConfiguration()
        self.visualization = VisualizationConfiguration()
        self.performance = PerformanceConfiguration()
    }
    
    /// Creates a custom configuration
/// Uses Swift 6 actor isolation for thread safety.
    public init(
        audioProcessor: AudioProcessorConfiguration,
        mlEngine: MLEngineConfiguration,
        visualization: VisualizationConfiguration,
        performance: PerformanceConfiguration
    ) {
        self.audioProcessor = audioProcessor
        self.mlEngine = mlEngine
        self.visualization = visualization
        self.performance = performance
    }
}

/// Audio processor configuration
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct AudioProcessorConfiguration: Sendable: Equatable, Sendable {
    /// Sample rate for audio processing
/// Uses Swift 6 actor isolation for thread safety.
    public var sampleRate: Double = 44100
    
    /// Buffer size for audio processing
/// Uses Swift 6 actor isolation for thread safety.
    public var bufferSize: Int = 1024
    
    /// Number of channels (typically 1 or 2)
/// Uses Swift 6 actor isolation for thread safety.
    public var channels: Int = 2
    
    /// FFT size for frequency analysis
/// Uses Swift 6 actor isolation for thread safety.
    public var fftSize: Int = 1024
    
    /// Whether to use hardware acceleration
/// Uses Swift 6 actor isolation for thread safety.
    public var useHardwareAcceleration: Bool = true
    
    /// Creates a default configuration
/// Uses Swift 6 actor isolation for thread safety.
    public init() {}
}

/// ML engine configuration
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct MLEngineConfiguration: Sendable: Equatable, Sendable {
    /// Whether the ML engine is enabled
/// Uses Swift 6 actor isolation for thread safety.
    public var enabled: Bool = true
    
    /// Whether to use the Neural Engine
/// Uses Swift 6 actor isolation for thread safety.
    public var useNeuralEngine: Bool = true
    
    /// Optimization level for processing
/// Uses Swift 6 actor isolation for thread safety.
    public var optimizationLevel: OptimizationLevel = .balanced
    
    /// Beat detection sensitivity (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var beatSensitivity: Float = 0.65
    
    /// Pattern recognition sensitivity (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var patternSensitivity: Float = 0.7
    
    /// Emotional content analysis sensitivity (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var emotionalSensitivity: Float = 0.5
    
    /// Pattern history duration in seconds
/// Uses Swift 6 actor isolation for thread safety.
    public var patternHistoryDuration: Float = 10.0
    
    /// Creates a default configuration
/// Uses Swift 6 actor isolation for thread safety.
    public init() {}
}

/// Visualization configuration
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct VisualizationConfiguration: Sendable: Equatable, Sendable {
    /// Default visualization mode
/// Uses Swift 6 actor isolation for thread safety.
    public var defaultMode: VisualizationMode = .spectrum
    
    /// Spectrum smoothing factor (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var spectrumSmoothing: Double = 0.5
    
    /// Color intensity (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var colorIntensity: Double = 0.8
    
    /// Motion intensity (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var motionIntensity: Double = 0.8
    
    /// Audio sensitivity (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    public var audioSensitivity: Double = 0.75
    
    /// Show beat indicator
/// Uses Swift 6 actor isolation for thread safety.
    public var showBeatIndicator: Bool = true
    
    /// Creates a default configuration
/// Uses Swift 6 actor isolation for thread safety.
    public init() {}
}

/// Performance configuration
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public struct PerformanceConfiguration: Sendable: Equatable, Sendable {
    /// Target frame rate
/// Uses Swift 6 actor isolation for thread safety.
    public var frameRateTarget: Int = 60
    
    /// Whether to show FPS counter
/// Uses Swift 6 actor isolation for thread safety.
    public var showFPS: Bool = false
    
    /// Maximum CPU usage percentage
/// Uses Swift 6 actor isolation for thread safety.
    public var maxCPUUsage: Double = 0.5
    
    /// Power efficiency mode
/// Uses Swift 6 actor isolation for thread safety.
    public var powerEfficiencyMode: Bool = false
    
    /// Creates a default configuration
/// Uses Swift 6 actor isolation for thread safety.
    public init() {}
}

// MARK: - Constants

/// Framework constants
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public enum Constants {
    /// Default frame rate
/// Uses Swift 6 actor isolation for thread safety.
    public static let defaultFrameRate = 60
    
    /// Default sample rate
/// Uses Swift 6 actor isolation for thread safety.
    public static let defaultSampleRate = 44100.0
    
    /// Default FFT size
/// Uses Swift 6 actor isolation for thread safety.
    public static let defaultFFTSize = 1024
    
    /// Maximum buffer size
/// Uses Swift 6 actor isolation for thread safety.
    public static let maxBufferSize = 4096
}

