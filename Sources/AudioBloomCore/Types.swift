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

// MARK: - Audio Data Types

/// Audio data structure representing an audio buffer and its properties
public struct AudioData: Sendable {
    /// The audio samples
    public let samples: [Float]
    
    /// Audio levels (left and right channels)
    public let levels: (left: Float, right: Float)
    
    /// Sample rate of the audio
    public let sampleRate: Double
    
    /// Timestamp when the audio was captured
    public let timestamp: Date
    
    /// Frequency spectrum data (if available)
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
public protocol AudioDataProvider: AnyObject, Sendable {
    /// Publisher for audio data updates
    var audioDataPublisher: AnyPublisher<AudioData, Never> { get }
    
    /// Current audio levels (e.g., left and right channel levels)
    var levels: (left: Float, right: Float) { get }
    
    /// Current frequency spectrum data
    var frequencyData: [Float] { get }
}

/// Protocol for audio data publishers with simplified requirements
public protocol AudioDataPublisher: AnyObject, Sendable {
    /// Publisher for audio data
    var publisher: AnyPublisher<AudioData, Never> { get }
    
    /// Publishes new audio data
    /// - Parameters:
    ///   - frequencyData: The frequency spectrum data
    ///   - levels: The audio level data
    func publish(frequencyData: [Float], levels: (Float, Float))
}

// MARK: - Visualization Types

/// Data structure for visualization
public struct VisualizationData: Sendable, Equatable {
    /// Values to visualize (typically frequency spectrum)
    public let values: [Float]
    
    /// Whether this data represents a significant event (like a beat)
    public let isSignificantEvent: Bool
    
    /// Timestamp of the data
    public let timestamp: Date
    
    /// Initializes a new visualization data instance
    /// - Parameters:
    ///   - values: Values to visualize
    ///   - isSignificantEvent: Whether this is a significant event
    ///   - timestamp: Timestamp of the data
    public init(values: [Float], isSignificantEvent: Bool = false, timestamp: Date = Date()) {
        self.values = values
        self.isSignificantEvent = isSignificantEvent
        self.timestamp = timestamp
    }
}

/// Enumeration of available visualization modes
public enum VisualizationMode: String, CaseIterable, Identifiable, Sendable {
    case spectrum = "Spectrum"
    case waveform = "Waveform"
    case particles = "Particles"
    case neural = "Neural"
    
    public var id: String { rawValue }
    
    /// Get a user-friendly description of the mode
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
public protocol MLProcessorProtocol: AnyObject, Sendable {
    /// Whether the processor is ready for processing
    var isReady: Bool { get }
    
    /// Process audio data for ML analysis
    /// - Parameter audioData: The audio data to analyze
    /// - Returns: Analysis results
    func processAudio(_ audioData: AudioData) throws -> [String: Any]
    
    /// Processes raw audio data
    /// - Parameter audioData: The raw audio data to process
    /// - Throws: Error if processing fails
    func processAudioData(_ audioData: [Float]) async throws
    
    /// Publisher for visualization data
    var visualizationDataPublisher: AnyPublisher<VisualizationData, Never> { get }
}

/// Optimization level for audio processing
public enum OptimizationLevel: String, Sendable {
    /// Balance efficiency and quality
    case balanced
    
    /// Prioritize quality (higher power usage)
    case quality
    
    /// Prioritize efficiency (lower power usage)
    case efficiency
}

/// Neural processor response for visualization
public struct NeuralProcessingResponse: Equatable, Sendable {
    /// Energy level (0-1)
    public var energy: Float = 0.0
    
    /// Pleasantness (0-1)
    public var pleasantness: Float = 0.5
    
    /// Complexity (0-1)
    public var complexity: Float = 0.5
    
    /// Pattern type (0-3)
    public var patternType: Int = 0
    
    /// Beat confidence (0-1)
    public var beatConfidence: Float = 0.0
    
    /// Default initialization
    public init() {}
    
    /// Full initialization
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
public struct Configuration: Equatable, Sendable {
    /// Audio processor configuration
    public var audioProcessor: AudioProcessorConfiguration
    
    /// ML engine configuration
    public var mlEngine: MLEngineConfiguration
    
    /// Visualization configuration
    public var visualization: VisualizationConfiguration
    
    /// Performance configuration
    public var performance: PerformanceConfiguration
    
    /// Creates a default configuration
    public init() {
        self.audioProcessor = AudioProcessorConfiguration()
        self.mlEngine = MLEngineConfiguration()
        self.visualization = VisualizationConfiguration()
        self.performance = PerformanceConfiguration()
    }
    
    /// Creates a custom configuration
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
public struct AudioProcessorConfiguration: Equatable, Sendable {
    /// Sample rate for audio processing
    public var sampleRate: Double = 44100
    
    /// Buffer size for audio processing
    public var bufferSize: Int = 1024
    
    /// Number of channels (typically 1 or 2)
    public var channels: Int = 2
    
    /// FFT size for frequency analysis
    public var fftSize: Int = 1024
    
    /// Whether to use hardware acceleration
    public var useHardwareAcceleration: Bool = true
    
    /// Creates a default configuration
    public init() {}
}

/// ML engine configuration
public struct MLEngineConfiguration: Equatable, Sendable {
    /// Whether the ML engine is enabled
    public var enabled: Bool = true
    
    /// Whether to use the Neural Engine
    public var useNeuralEngine: Bool = true
    
    /// Optimization level for processing
    public var optimizationLevel: OptimizationLevel = .balanced
    
    /// Beat detection sensitivity (0-1)
    public var beatSensitivity: Float = 0.65
    
    /// Pattern recognition sensitivity (0-1)
    public var patternSensitivity: Float = 0.7
    
    /// Emotional content analysis sensitivity (0-1)
    public var emotionalSensitivity: Float = 0.5
    
    /// Pattern history duration in seconds
    public var patternHistoryDuration: Float = 10.0
    
    /// Creates a default configuration
    public init() {}
}

/// Visualization configuration
public struct VisualizationConfiguration: Equatable, Sendable {
    /// Default visualization mode
    public var defaultMode: VisualizationMode = .spectrum
    
    /// Spectrum smoothing factor (0-1)
    public var spectrumSmoothing: Double = 0.5
    
    /// Color intensity (0-1)
    public var colorIntensity: Double = 0.8
    
    /// Motion intensity (0-1)
    public var motionIntensity: Double = 0.8
    
    /// Audio sensitivity (0-1)
    public var audioSensitivity: Double = 0.75
    
    /// Show beat indicator
    public var showBeatIndicator: Bool = true
    
    /// Creates a default configuration
    public init() {}
}

/// Performance configuration
public struct PerformanceConfiguration: Equatable, Sendable {
    /// Target frame rate
    public var frameRateTarget: Int = 60
    
    /// Whether to show FPS counter
    public var showFPS: Bool = false
    
    /// Maximum CPU usage percentage
    public var maxCPUUsage: Double = 0.5
    
    /// Power efficiency mode
    public var powerEfficiencyMode: Bool = false
    
    /// Creates a default configuration
    public init() {}
}

// MARK: - Constants

/// Framework constants
public enum Constants {
    /// Default frame rate
    public static let defaultFrameRate = 60
    
    /// Default sample rate
    public static let defaultSampleRate = 44100.0
    
    /// Default FFT size
    public static let defaultFFTSize = 1024
    
    /// Maximum buffer size
    public static let maxBufferSize = 4096
}

