//
// AudioBloomSettings.swift
// Settings manager for AudioBloomAI
//

import Foundation
import SwiftUI
import Combine

/// Application settings manager with persistence
public class AudioBloomSettings: ObservableObject {
    // MARK: - UserDefaults Keys
    
    private enum Keys {
        static let currentTheme = "ABSettingsCurrentTheme"
        static let audioSensitivity = "ABSettingsAudioSensitivity"
        static let motionIntensity = "ABSettingsMotionIntensity"
        static let neuralEngineEnabled = "ABSettingsNeuralEngineEnabled"
        static let frameRateTarget = "ABSettingsFrameRateTarget"
        static let audioQuality = "ABSettingsAudioQuality"
        static let visualizationMode = "ABSettingsVisualizationMode"
        static let colorIntensity = "ABSettingsColorIntensity"
        static let spectrumSmoothing = "ABSettingsSpectrumSmoothing"
        static let autoStart = "ABSettingsAutoStart"
        static let showFPS = "ABSettingsShowFPS"
        static let lastInputDevice = "ABSettingsLastInputDevice"
        static let lastOutputDevice = "ABSettingsLastOutputDevice"
    }
    
    // MARK: - Defaults
    
    private static let defaultTheme: AudioBloomCore.VisualTheme = .classic
    private static let defaultAudioSensitivity: Double = 0.75
    private static let defaultMotionIntensity: Double = 0.8
    private static let defaultNeuralEngineEnabled: Bool = true
    private static let defaultFrameRateTarget: Int = AudioBloomCore.Constants.defaultFrameRate
    private static let defaultAudioQuality: Int = Int(AudioBloomCore.Constants.defaultSampleRate)
    private static let defaultVisualizationMode: VisualizationMode = .spectrum
    private static let defaultColorIntensity: Double = 0.8
    private static let defaultSpectrumSmoothing: Double = 0.5
    private static let defaultAutoStart: Bool = true
    private static let defaultShowFPS: Bool = false
    
    // MARK: - Properties with UserDefaults Persistence
    
    /// Current visual theme
    @Published public var currentTheme: AudioBloomCore.VisualTheme {
        didSet {
            UserDefaults.standard.set(currentTheme.rawValue, forKey: Keys.currentTheme)
            notifyVisualizationUpdate()
        }
    }
    
    /// Audio sensitivity (0.0 - 1.0)
    @Published public var audioSensitivity: Double {
        didSet {
            UserDefaults.standard.set(audioSensitivity, forKey: Keys.audioSensitivity)
            notifyVisualizationUpdate()
        }
    }
    
    /// Motion intensity (0.0 - 1.0)
    @Published public var motionIntensity: Double {
        didSet {
            UserDefaults.standard.set(motionIntensity, forKey: Keys.motionIntensity)
            notifyVisualizationUpdate()
        }
    }
    
    /// Neural Engine enabled
    @Published public var neuralEngineEnabled: Bool {
        didSet {
            UserDefaults.standard.set(neuralEngineEnabled, forKey: Keys.neuralEngineEnabled)
            notifyVisualizationUpdate()
        }
    }
    
    /// Frame rate target
    @Published public var frameRateTarget: Int {
        didSet {
            UserDefaults.standard.set(frameRateTarget, forKey: Keys.frameRateTarget)
            notifyPerformanceUpdate()
        }
    }
    
    /// Audio quality (sample rate)
    @Published public var audioQuality: Int {
        didSet {
            UserDefaults.standard.set(audioQuality, forKey: Keys.audioQuality)
            notifyAudioConfigUpdate()
        }
    }
    
    /// Visualization mode
    @Published public var visualizationMode: VisualizationMode {
        didSet {
            UserDefaults.standard.set(visualizationMode.rawValue, forKey: Keys.visualizationMode)
            notifyVisualizationUpdate()
        }
    }
    
    /// Color intensity (0.0 - 1.0)
    @Published public var colorIntensity: Double {
        didSet {
            UserDefaults.standard.set(colorIntensity, forKey: Keys.colorIntensity)
            notifyVisualizationUpdate()
        }
    }
    
    /// Spectrum smoothing (0.0 - 1.0)
    @Published public var spectrumSmoothing: Double {
        didSet {
            UserDefaults.standard.set(spectrumSmoothing, forKey: Keys.spectrumSmoothing)
            notifyVisualizationUpdate()
        }
    }
    
    /// Auto-start audio capture on launch
    @Published public var autoStart: Bool {
        didSet {
            UserDefaults.standard.set(autoStart, forKey: Keys.autoStart)
        }
    }
    
    /// Show FPS counter
    @Published public var showFPS: Bool {
        didSet {
            UserDefaults.standard.set(showFPS, forKey: Keys.showFPS)
            notifyPerformanceUpdate()
        }
    }
    
    /// Last used input device ID
    @Published public var lastInputDeviceID: String? {
        didSet {
            if let deviceID = lastInputDeviceID {
                UserDefaults.standard.set(deviceID, forKey: Keys.lastInputDevice)
            } else {
                UserDefaults.standard.removeObject(forKey: Keys.lastInputDevice)
            }
            notifyAudioConfigUpdate()
        }
    }
    
    /// Last used output device ID
    @Published public var lastOutputDeviceID: String? {
        didSet {
            if let deviceID = lastOutputDeviceID {
                UserDefaults.standard.set(deviceID, forKey: Keys.lastOutputDevice)
            } else {
                UserDefaults.standard.removeObject(forKey: Keys.lastOutputDevice)
            }
            notifyAudioConfigUpdate()
        }
    }
    
    /// Visualization modes available in the app
    public enum VisualizationMode: String, CaseIterable, Identifiable {
        case spectrum = "Spectrum"
        case waveform = "Waveform"
        case particles = "Particles"
        case neural = "Neural"
        
        public var id: String { self.rawValue }
        
        /// Description of the visualization mode
        public var description: String {
            switch self {
            case .spectrum:
                return "Frequency spectrum visualization"
            case .waveform:
                return "Audio waveform visualization"
            case .particles:
                return "Particle-based visualization"
            case .neural:
                return "Neural network-driven visualization"
            }
        }
    }
    
    // MARK: - Publishers for Observed Changes
    
    /// Publisher for visualization parameter updates
    public let visualizationUpdatePublisher = PassthroughSubject<[String: Any], Never>()
    
    /// Publisher for audio configuration updates
    public let audioConfigUpdatePublisher = PassthroughSubject<Void, Never>()
    
    /// Publisher for performance configuration updates
    public let performanceUpdatePublisher = PassthroughSubject<Void, Never>()
    
    // MARK: - Initialization
    
    /// Default initialization with standard settings
    public init() {
        // Load theme from UserDefaults or use default
        if let themeName = UserDefaults.standard.string(forKey: Keys.currentTheme),
           let theme = AudioBloomCore.VisualTheme(rawValue: themeName) {
            self.currentTheme = theme
        } else {
            self.currentTheme = Self.defaultTheme
        }
        
        // Load visualization mode from UserDefaults or use default
        if let modeName = UserDefaults.standard.string(forKey: Keys.visualizationMode),
           let mode = VisualizationMode(rawValue: modeName) {
            self.visualizationMode = mode
        } else {
            self.visualizationMode = Self.defaultVisualizationMode
        }
        
        // Load numeric values with defaults
        self.audioSensitivity = UserDefaults.standard.double(forKey: Keys.audioSensitivity)
        if self.audioSensitivity == 0 { self.audioSensitivity = Self.defaultAudioSensitivity }
        
        self.motionIntensity = UserDefaults.standard.double(forKey: Keys.motionIntensity)
        if self.motionIntensity == 0 { self.motionIntensity = Self.defaultMotionIntensity }
        
        self.neuralEngineEnabled = UserDefaults.standard.bool(forKey: Keys.neuralEngineEnabled)
        if !UserDefaults.standard.object(forKey: Keys.neuralEngineEnabled) {
            self.neuralEngineEnabled = Self.defaultNeuralEngineEnabled
        }
        
        self.frameRateTarget = UserDefaults.standard.integer(forKey: Keys.frameRateTarget)
        if self.frameRateTarget == 0 { self.frameRateTarget = Self.defaultFrameRateTarget }
        
        self.audioQuality = UserDefaults.standard.integer(forKey: Keys.audioQuality)
        if self.audioQuality == 0 { self.audioQuality = Self.defaultAudioQuality }
        
        self.colorIntensity = UserDefaults.standard.double(forKey: Keys.colorIntensity)
        if self.colorIntensity == 0 { self.colorIntensity = Self.defaultColorIntensity }
        
        self.spectrumSmoothing = UserDefaults.standard.double(forKey: Keys.spectrumSmoothing)
        if self.spectrumSmoothing == 0 { self.spectrumSmoothing = Self.defaultSpectrumSmoothing }
        
        self.autoStart = UserDefaults.standard.bool(forKey: Keys.autoStart)
        if !UserDefaults.standard.object(forKey: Keys.autoStart) {
            self.autoStart = Self.defaultAutoStart
        }
        
        self.showFPS = UserDefaults.standard.bool(forKey: Keys.showFPS)
        if !UserDefaults.standard.object(forKey: Keys.showFPS) {
            self.showFPS = Self.defaultShowFPS
        }
        
        // Load device IDs if available
        self.lastInputDeviceID = UserDefaults.standard.string(forKey: Keys.lastInputDevice)
        self.lastOutputDeviceID = UserDefaults.standard.string(forKey: Keys.lastOutputDevice)
    }
    
    // MARK: - Methods
    
    /// Reset all settings to default values
    public func resetToDefaults() {
        currentTheme = Self.defaultTheme
        audioSensitivity = Self.defaultAudioSensitivity
        motionIntensity = Self.defaultMotionIntensity
        neuralEngineEnabled = Self.defaultNeuralEngineEnabled
        frameRateTarget = Self.defaultFrameRateTarget
        audioQuality = Self.defaultAudioQuality
        visualizationMode = Self.defaultVisualizationMode
        colorIntensity = Self.defaultColorIntensity
        spectrumSmoothing = Self.defaultSpectrumSmoothing
        autoStart = Self.defaultAutoStart
        showFPS = Self.defaultShowFPS
        
        // Clear device selections
        lastInputDeviceID = nil
        lastOutputDeviceID = nil
        
        // Notify subscribers about the reset
        notifyVisualizationUpdate()
        notifyAudioConfigUpdate()
        notifyPerformanceUpdate()
    }
    
    /// Get full parameter dictionary for visualizers
    public func visualizationParameters() -> [String: Any] {
        var params = currentTheme.colorParameters
        
        // Add additional parameters
        params["sensitivity"] = Float(audioSensitivity)
        params["motionIntensity"] = Float(motionIntensity)
        params["neuralEngineEnabled"] = neuralEngineEnabled
        params["visualizationMode"] = visualizationMode.rawValue
        params["colorIntensity"] = Float(colorIntensity)
        params["spectrumSmoothing"] = Float(spectrumSmoothing)
        
        return params
    }
    
    /// Get audio configuration
    public func audioConfiguration() -> [String: Any] {
        [
            "sampleRate": Double(audioQuality),
            "fftSize": AudioBloomCore.Constants.defaultFFTSize,
            "channels": 2,
            "inputDeviceID": lastInputDeviceID as Any,
            "outputDeviceID": lastOutputDeviceID as Any
        ]
    }
    
    /// Imports settings from a dictionary
    public func importSettings(from dict: [String: Any]) {
        // Handle theme
        if let themeName = dict[Keys.currentTheme] as? String,
           let theme = AudioBloomCore.VisualTheme(rawValue: themeName) {
            currentTheme = theme
        }
        
        // Handle visualization mode
        if let modeName = dict[Keys.visualizationMode] as? String,
           let mode = VisualizationMode(rawValue: modeName) {
            visualizationMode = mode
        }
        
        // Handle numeric values
        if let value = dict[Keys.audioSensitivity] as? Double {
            audioSensitivity = min(max(value, 0), 1)
        }
        
        if let value = dict[Keys.motionIntensity] as? Double {
            motionIntensity = min(max(value, 0), 1)
        }
        
        if let value = dict[Keys.neuralEngineEnabled] as? Bool {
            neuralEngineEnabled = value
        }
        
        if let value = dict[Keys.frameRateTarget] as? Int {
            frameRateTarget = max(value, 30)
        }
        
        if let value = dict[Keys.audioQuality] as? Int {
            audioQuality = max(value, 22050)
        }
        
        if let value = dict[Keys.colorIntensity] as? Double {
            colorIntensity = min(max(value, 0), 1)
        }
        
        if let value = dict[Keys.spectrumSmoothing] as? Double {
            spectrumSmoothing = min(max(value, 0), 1)
        }
        
        if let value = dict[Keys.autoStart] as? Bool {
            autoStart = value
        }
        
        if let value = dict[Keys.showFPS] as? Bool {
            showFPS = value
        }
        
        // Handle device IDs
        lastInputDeviceID = dict[Keys.lastInputDevice] as? String
        lastOutputDeviceID = dict[Keys.lastOutputDevice] as? String
        
        // Notify subscribers about all updates
        notifyVisualizationUpdate()
        notifyAudioConfigUpdate()
        notifyPerformanceUpdate()
    }
    
    /// Exports settings to a dictionary
    public func exportSettings() -> [String: Any] {
        [
            Keys.currentTheme: currentTheme.rawValue,
            Keys.visualizationMode: visualizationMode.rawValue,
            Keys.audioSensitivity: audioSensitivity,
            Keys.motionIntensity: motionIntensity,
            Keys.neuralEngineEnabled: neuralEngineEnabled,
            Keys.frameRateTarget: frameRateTarget,
            Keys.audioQuality: audioQuality,
            Keys.colorIntensity: colorIntensity,
            Keys.spectrumSmoothing: spectrumSmoothing,
            Keys.autoStart: autoStart,
            Keys.showFPS: showFPS,
            Keys.lastInputDevice: lastInputDeviceID as Any,
            Keys.lastOutputDevice: lastOutputDeviceID as Any
        ]
    }
    
    // MARK: - Notification Methods
    
    /// Notifies subscribers of visualization parameter changes
    private func notifyVisualizationUpdate() {
        visualizationUpdatePublisher.send(visualizationParameters())
    }
    
    /// Notifies subscribers of audio configuration changes
    private func notifyAudioConfigUpdate() {
        audioConfigUpdatePublisher.send()
    }
    
    /// Notifies subscribers of performance configuration changes
    private func notifyPerformanceUpdate() {
        performanceUpdatePublisher.send()
    }
}
