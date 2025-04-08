import Foundation
import AVFoundation
import Combine

/// AudioBloomCore provides the shared protocols and utilities used across the application
public enum AudioBloomCore {
    /// Application-wide constants
    public enum Constants {
        /// Default audio sample rate
        public static let defaultSampleRate: Double = 44100.0
        
        /// Default FFT size for audio analysis
        public static let defaultFFTSize: Int = 2048
        
        /// Default frame rate for visualization
        public static let defaultFrameRate: Int = 60
        
        /// Application name
        public static let appName = "AudioBloom"
        
        /// Application version
        public static let appVersion = "0.1.0"
    }
    
    /// Error types used throughout the application
    public enum Error: Swift.Error, LocalizedError {
        case audioSessionSetupFailed
        case audioEngineStartFailed
        case metalDeviceNotFound
        case metalCommandQueueCreationFailed
        case mlModelLoadFailed
        
        public var errorDescription: String? {
            switch self {
            case .audioSessionSetupFailed:
                return "Failed to set up audio session"
            case .audioEngineStartFailed:
                return "Failed to start audio engine"
            case .metalDeviceNotFound:
                return "Metal device not found"
            case .metalCommandQueueCreationFailed:
                return "Failed to create Metal command queue"
            case .mlModelLoadFailed:
                return "Failed to load ML model"
            }
        }
    }
    
    /// Visual theme options
    public enum VisualTheme: String, CaseIterable, Identifiable {
        case classic = "Classic"
        case neon = "Neon"
        case monochrome = "Monochrome"
        case cosmic = "Cosmic"
        
        public var id: String { self.rawValue }
        
        /// Returns color parameters for the theme
        public var colorParameters: [String: Any] {
            switch self {
            case .classic:
                return [
                    "primaryColor": [0.0, 0.5, 1.0, 1.0],
                    "secondaryColor": [1.0, 0.0, 0.5, 1.0],
                    "backgroundColor": [0.0, 0.0, 0.1, 1.0]
                ]
            case .neon:
                return [
                    "primaryColor": [0.0, 1.0, 0.8, 1.0],
                    "secondaryColor": [1.0, 0.0, 1.0, 1.0],
                    "backgroundColor": [0.05, 0.0, 0.05, 1.0]
                ]
            case .monochrome:
                return [
                    "primaryColor": [0.9, 0.9, 0.9, 1.0],
                    "secondaryColor": [0.6, 0.6, 0.6, 1.0],
                    "backgroundColor": [0.1, 0.1, 0.1, 1.0]
                ]
            case .cosmic:
                return [
                    "primaryColor": [0.5, 0.0, 0.8, 1.0],
                    "secondaryColor": [0.0, 0.8, 0.8, 1.0],
                    "backgroundColor": [0.0, 0.0, 0.2, 1.0]
                ]
            }
        }
    }
}

/// Protocol defining required functionality for audio data providers
public protocol AudioDataProvider: ObservableObject {
    /// Current audio levels (e.g., left and right channel levels)
    var levels: (left: Float, right: Float) { get }
    
    /// Current frequency spectrum data
    var frequencyData: [Float] { get }
    
    /// Sets up the audio session
    func setupAudioSession() async throws
    
    /// Starts audio capture
    func startCapture() throws
    
    /// Stops audio capture
    func stopCapture()
}

/// Protocol defining required functionality for visualization renderers
public protocol VisualizationRenderer: ObservableObject {
    /// Flag indicating if the renderer is ready
    var isReady: Bool { get }
    
    /// Prepares the renderer for drawing
    func prepareRenderer()
    
    /// Updates the renderer with new audio data
    func update(audioData: [Float], levels: (left: Float, right: Float))
    
    /// Renders a frame
    func render()
}

/// Protocol defining required functionality for ML processors
public protocol MLProcessing: ObservableObject {
    /// Current ML model output data
    var outputData: [Float] { get }
    
    /// Prepares the ML model for processing
    func prepareMLModel()
    
    /// Processes audio data through the ML model
    func processAudioData(_ audioData: [Float]) async
}

/// Protocol for objects that can receive visualization parameters
public protocol VisualizationParameterReceiver {
    /// Updates parameters for the visualization
    func updateParameters(_ parameters: [String: Any])
}

/// A publisher that emits audio data at regular intervals
public class AudioDataPublisher {
    /// The subject that publishes audio data
    private let subject = PassthroughSubject<([Float], (left: Float, right: Float)), Never>()
    
    /// Public initializer
    public init() {}
    
    /// The publisher that emits audio data
    public var publisher: AnyPublisher<([Float], (left: Float, right: Float)), Never> {
        subject.eraseToAnyPublisher()
    }
    
    /// Publishes new audio data
    public func publish(frequencyData: [Float], levels: (left: Float, right: Float)) {
        subject.send((frequencyData, levels))
    }
}

/// Application settings manager
public class AudioBloomSettings: ObservableObject {
    /// Current visual theme
    @Published public var currentTheme: AudioBloomCore.VisualTheme = .classic
    
    /// Audio sensitivity (0.0 - 1.0)
    @Published public var audioSensitivity: Double = 0.75
    
    /// Motion intensity (0.0 - 1.0)
    @Published public var motionIntensity: Double = 0.8
    
    /// Neural Engine enabled
    @Published public var neuralEngineEnabled: Bool = true
    
    /// Frame rate target
    @Published public var frameRateTarget: Int = AudioBloomCore.Constants.defaultFrameRate
    
    /// Audio quality
    @Published public var audioQuality: Int = Int(AudioBloomCore.Constants.defaultSampleRate)
    
    /// Default initialization with standard settings
    public init() {}
    
    /// Get full parameter dictionary for visualizers
    public func visualizationParameters() -> [String: Any] {
        var params = currentTheme.colorParameters
        
        // Add additional parameters
        params["sensitivity"] = Float(audioSensitivity)
        params["motionIntensity"] = Float(motionIntensity)
        params["neuralEngineEnabled"] = neuralEngineEnabled
        
        return params
    }
}
