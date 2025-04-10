//
// AudioBloomSettings.swift
// Settings manager for AudioBloomAI
//

import Foundation
import SwiftUI
import Combine
import os.log
/// Application settings manager with persistence
///
/// This class manages all user configurable settings for the AudioBloom application.
/// It provides persistence through UserDefaults and publishes changes to interested subscribers.
///
/// - Note: This class is marked as `@unchecked Sendable` because it contains `@Published` properties
///         that are not themselves `Sendable`. However, all mutations are performed on the main thread
///         through SwiftUI's state management system, making it thread-safe in practice.
public final class AudioBloomSettings: ObservableObject, @unchecked Sendable {
    /// Logger instance for settings
    private let logger = Logger(subsystem: "com.audiobloom.settings", category: "settings")
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
        static let microphoneVolume = "ABSettingsMicrophoneVolume"
        static let systemAudioVolume = "ABSettingsSystemAudioVolume"
        static let mixAudioInputs = "ABSettingsMixAudioInputs"
        static let beatSensitivity = "ABSettingsBeatSensitivity"
        static let patternSensitivity = "ABSettingsPatternSensitivity"
        static let emotionalSensitivity = "ABSettingsEmotionalSensitivity"
        static let showBeatIndicator = "ABSettingsShowBeatIndicator"
        static let selectedAudioSource = "ABSettingsSelectedAudioSource"
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
    private static let defaultMicrophoneVolume: Float = 1.0
    private static let defaultSystemAudioVolume: Float = 0.8
    private static let defaultMixAudioInputs: Bool = false
    private static let defaultBeatSensitivity: Float = 0.65
    private static let defaultPatternSensitivity: Float = 0.7
    private static let defaultEmotionalSensitivity: Float = 0.5
    private static let defaultShowBeatIndicator: Bool = true
    private static let defaultSelectedAudioSource: String = "Microphone"
    // MARK: - Properties with UserDefaults Persistence
    
    /// Current visual theme
    @Published public var currentTheme: AudioBloomCore.VisualTheme {
        didSet {
            UserDefaults.standard.set(currentTheme.rawValue, forKey: Keys.currentTheme)
            notifyVisualizationUpdate()
        }
    }
    
    // MARK: - Publishers for Observed Changes
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
    
    /// Microphone volume level (0.0 - 1.0)
    @Published public var microphoneVolume: Float {
        didSet {
            UserDefaults.standard.set(microphoneVolume, forKey: Keys.microphoneVolume)
            notifyAudioConfigUpdate()
        }
    }
    
    /// System audio volume level (0.0 - 1.0)
    @Published public var systemAudioVolume: Float {
        didSet {
            UserDefaults.standard.set(systemAudioVolume, forKey: Keys.systemAudioVolume)
            notifyAudioConfigUpdate()
        }
    }
    
    /// Whether to mix audio inputs
    @Published public var mixAudioInputs: Bool {
        didSet {
            UserDefaults.standard.set(mixAudioInputs, forKey: Keys.mixAudioInputs)
            notifyAudioConfigUpdate()
        }
    }
    
    /// Beat detection sensitivity (0.0 - 1.0)
    @Published public var beatSensitivity: Float {
        didSet {
            UserDefaults.standard.set(beatSensitivity, forKey: Keys.beatSensitivity)
            notifyVisualizationUpdate()
        }
    }
    
    /// Pattern recognition sensitivity (0.0 - 1.0)
    @Published public var patternSensitivity: Float {
        didSet {
            UserDefaults.standard.set(patternSensitivity, forKey: Keys.patternSensitivity)
            notifyVisualizationUpdate()
        }
    }
    
    /// Emotional content analysis sensitivity (0.0 - 1.0)
    @Published public var emotionalSensitivity: Float {
        didSet {
            UserDefaults.standard.set(emotionalSensitivity, forKey: Keys.emotionalSensitivity)
            notifyVisualizationUpdate()
        }
    }
    
    /// Whether to show beat indicator
    @Published public var showBeatIndicator: Bool {
        didSet {
            UserDefaults.standard.set(showBeatIndicator, forKey: Keys.showBeatIndicator)
            notifyVisualizationUpdate()
        }
    }
    
    /// Selected audio source type
    @Published public var selectedAudioSource: String {
        didSet {
            UserDefaults.standard.set(selectedAudioSource, forKey: Keys.selectedAudioSource)
            notifyAudioConfigUpdate()
        }
    }
    
    /// Convenience accessors for compatibility with PresetManager
    public var selectedInputDeviceID: String? {
        get { return lastInputDeviceID }
        set { lastInputDeviceID = newValue }
    }
    
    public var selectedOutputDeviceID: String? {
        get { return lastOutputDeviceID }
        set { lastOutputDeviceID = newValue }
    }
    /// Visualization modes available in the app
    public enum VisualizationMode: String, CaseIterable, Identifiable, Sendable {
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
<<<<<<< HEAD
    }
    
=======
>>>>>>> fix/70-settings-structure
    // MARK: - Publishers for Observed Changes
    
    /// Serial queue for thread-safe publisher operations
    private let publisherQueue = DispatchQueue(label: "com.audiobloom.settings.publishers", qos: .userInitiated)
    
    /// Publisher for visualization parameter updates
    /// - Note: Access to this publisher is synchronized through the publisherQueue
    public let visualizationUpdatePublisher = PassthroughSubject<[String: Any], Never>()
    
    /// Publisher for audio configuration updates
    /// - Note: Access to this publisher is synchronized through the publisherQueue
    public let audioConfigUpdatePublisher = PassthroughSubject<Void, Never>()
    
    /// Publisher for performance configuration updates
    /// - Note: Access to this publisher is synchronized through the publisherQueue
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
        if UserDefaults.standard.object(forKey: Keys.neuralEngineEnabled) == nil {
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
        if UserDefaults.standard.object(forKey: Keys.autoStart) == nil {
            self.autoStart = Self.defaultAutoStart
        }
        
        self.showFPS = UserDefaults.standard.bool(forKey: Keys.showFPS)
        if UserDefaults.standard.object(forKey: Keys.showFPS) == nil {
            self.showFPS = Self.defaultShowFPS
        }
        
        // Load device IDs if available
        self.lastInputDeviceID = UserDefaults.standard.string(forKey: Keys.lastInputDevice)
        self.lastOutputDeviceID = UserDefaults.standard.string(forKey: Keys.lastOutputDevice)
        
        // Load audio-related settings
        self.microphoneVolume = UserDefaults.standard.float(forKey: Keys.microphoneVolume)
        if self.microphoneVolume == 0 { self.microphoneVolume = Self.defaultMicrophoneVolume }
        
        self.systemAudioVolume = UserDefaults.standard.float(forKey: Keys.systemAudioVolume)
        if self.systemAudioVolume == 0 { self.systemAudioVolume = Self.defaultSystemAudioVolume }
        
        self.mixAudioInputs = UserDefaults.standard.bool(forKey: Keys.mixAudioInputs)
        
        // Load neural-related settings
        self.beatSensitivity = UserDefaults.standard.float(forKey: Keys.beatSensitivity)
        if self.beatSensitivity == 0 { self.beatSensitivity = Self.defaultBeatSensitivity }
        
        self.patternSensitivity = UserDefaults.standard.float(forKey: Keys.patternSensitivity)
        if self.patternSensitivity == 0 { self.patternSensitivity = Self.defaultPatternSensitivity }
        
        self.emotionalSensitivity = UserDefaults.standard.float(forKey: Keys.emotionalSensitivity)
        if self.emotionalSensitivity == 0 { self.emotionalSensitivity = Self.defaultEmotionalSensitivity }
        
        // Load visualization-related settings
        self.showBeatIndicator = UserDefaults.standard.bool(forKey: Keys.showBeatIndicator)
        if UserDefaults.standard.object(forKey: Keys.showBeatIndicator) == nil {
            self.showBeatIndicator = Self.defaultShowBeatIndicator
        }
        
        // Load audio source
        if let source = UserDefaults.standard.string(forKey: Keys.selectedAudioSource) {
            self.selectedAudioSource = source
        } else {
            self.selectedAudioSource = Self.defaultSelectedAudioSource
        }
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
        autoStart = Self.defaultAutoStart
        showFPS = Self.defaultShowFPS
        showBeatIndicator = Self.defaultShowBeatIndicator
        
        // Reset audio settings
        microphoneVolume = Self.defaultMicrophoneVolume
        systemAudioVolume = Self.defaultSystemAudioVolume
        mixAudioInputs = Self.defaultMixAudioInputs
        selectedAudioSource = Self.defaultSelectedAudioSource
        
        // Reset neural settings
        beatSensitivity = Self.defaultBeatSensitivity
        patternSensitivity = Self.defaultPatternSensitivity
        emotionalSensitivity = Self.defaultEmotionalSensitivity
        
        // Clear device selections
        lastInputDeviceID = nil
        lastOutputDeviceID = nil
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
        params["beatSensitivity"] = beatSensitivity
        params["patternSensitivity"] = patternSensitivity
        params["emotionalSensitivity"] = emotionalSensitivity
        params["showBeatIndicator"] = showBeatIndicator
        
        return params
    }
    
    /// Get audio configuration
    public func audioConfiguration() -> [String: Any] {
        return [
            "sampleRate": Double(audioQuality),
            "fftSize": AudioBloomCore.Constants.defaultFFTSize,
            "channels": 2,
            "inputDeviceID": lastInputDeviceID as Any,
            "outputDeviceID": lastOutputDeviceID as Any,
            "microphoneVolume": microphoneVolume,
            "systemAudioVolume": systemAudioVolume,
            "mixAudioInputs": mixAudioInputs,
            "selectedAudioSource": selectedAudioSource
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
        
        if let value = dict[Keys.showBeatIndicator] as? Bool {
            showBeatIndicator = value
        }
        
        // Handle audio settings
        if let value = dict[Keys.microphoneVolume] as? Float {
            microphoneVolume = min(max(value, 0), 1)
        }
        
        if let value = dict[Keys.systemAudioVolume] as? Float {
            systemAudioVolume = min(max(value, 0), 1)
        }
        
        if let value = dict[Keys.mixAudioInputs] as? Bool {
            mixAudioInputs = value
        }
        
        // Handle neural settings
        if let value = dict[Keys.beatSensitivity] as? Float {
            beatSensitivity = min(max(value, 0), 1)
        }
        
        if let value = dict[Keys.patternSensitivity] as? Float {
            patternSensitivity = min(max(value, 0), 1)
        }
        
        if let value = dict[Keys.emotionalSensitivity] as? Float {
            emotionalSensitivity = min(max(value, 0), 1)
        }
        
        // Handle audio source
        if let value = dict[Keys.selectedAudioSource] as? String {
            selectedAudioSource = value
        }
        
        // Handle device IDs
        // Handle device IDs
        lastInputDeviceID = dict[Keys.lastInputDevice] as? String
        lastOutputDeviceID = dict[Keys.lastOutputDevice] as? String
        
        notifyAudioConfigUpdate()
        notifyPerformanceUpdate()
    }
    
    /// Exports settings to a dictionary
    public func exportSettings() -> [String: Any] {
        return [
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
            Keys.lastOutputDevice: lastOutputDeviceID as Any,
            Keys.microphoneVolume: microphoneVolume,
            Keys.systemAudioVolume: systemAudioVolume,
            Keys.mixAudioInputs: mixAudioInputs,
            Keys.beatSensitivity: beatSensitivity,
            Keys.patternSensitivity: patternSensitivity,
            Keys.emotionalSensitivity: emotionalSensitivity,
            Keys.showBeatIndicator: showBeatIndicator,
            Keys.selectedAudioSource: selectedAudioSource
        ]
    }
    
    // MARK: - Notification Methods
    
    /// Notifies subscribers of visualization parameter changes
    private func notifyVisualizationUpdate() {
        // Ensure thread-safety by dispatching to dedicated queue
        publisherQueue.async { [weak self] in
            guard let self = self else { return }
            self.visualizationUpdatePublisher.send(self.visualizationParameters())
            self.logger.debug("Visualization update notification sent")
        }
    }
    
    /// Notifies subscribers of audio configuration changes
    private func notifyAudioConfigUpdate() {
        // Ensure thread-safety by dispatching to dedicated queue
        publisherQueue.async { [weak self] in
            guard let self = self else { return }
            self.audioConfigUpdatePublisher.send()
            self.logger.debug("Audio configuration update notification sent")
        }
    }
    
    /// Notifies subscribers of performance configuration changes
    private func notifyPerformanceUpdate() {
        // Ensure thread-safety by dispatching to dedicated queue
        publisherQueue.async { [weak self] in
            guard let self = self else { return }
            self.performanceUpdatePublisher.send()
            self.logger.debug("Performance update notification sent")
        }
    }
