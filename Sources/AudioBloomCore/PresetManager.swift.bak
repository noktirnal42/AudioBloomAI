import Foundation
import Combine
import SwiftUI

/// Represents a saved configuration preset for AudioBloom
public struct Preset: Identifiable, Codable, Equatable {
    /// Unique identifier for the preset
    public let id: UUID
    
    /// Display name of the preset
    public var name: String
    
    /// Optional description of the preset
    public var description: String
    
    /// Date the preset was created
    public let createdDate: Date
    
    /// Date the preset was last modified
    public var lastModifiedDate: Date
    
    /// Visualization settings
    public var visualSettings: VisualizationSettings
    
    /// Audio settings
    public var audioSettings: AudioSettings
    
    /// Neural engine settings
    public var neuralSettings: NeuralSettings
    
    /// Creates a new preset with the given name
    /// - Parameter name: The display name for the preset
    public init(name: String, description: String = "") {
        self.id = UUID()
        self.name = name
        self.description = description
        self.createdDate = Date()
        self.lastModifiedDate = createdDate
        self.visualSettings = VisualizationSettings()
        self.audioSettings = AudioSettings()
        self.neuralSettings = NeuralSettings()
    }
    
    /// Creates a new preset with all parameters
    /// - Parameters:
    ///   - id: Unique identifier (generated if not provided)
    ///   - name: Display name
    ///   - description: Optional description
    ///   - createdDate: Creation date (defaults to now)
    ///   - lastModifiedDate: Last modified date (defaults to now)
    ///   - visualSettings: Visualization settings
    ///   - audioSettings: Audio settings
    ///   - neuralSettings: Neural engine settings
    public init(id: UUID = UUID(),
                name: String,
                description: String = "",
                createdDate: Date = Date(),
                lastModifiedDate: Date? = nil,
                visualSettings: VisualizationSettings,
                audioSettings: AudioSettings,
                neuralSettings: NeuralSettings) {
        self.id = id
        self.name = name
        self.description = description
        self.createdDate = createdDate
        self.lastModifiedDate = lastModifiedDate ?? createdDate
        self.visualSettings = visualSettings
        self.audioSettings = audioSettings
        self.neuralSettings = neuralSettings
    }
    
    /// Creates a preset from the current app settings
    /// - Parameter settings: The current application settings
    /// - Parameter name: Name for the preset
    /// - Parameter description: Optional description
    public static func fromCurrentSettings(settings: AudioBloomSettings, name: String, description: String = "") -> Preset {
        // Create a preset with current settings
        let visualSettings = VisualizationSettings(
            theme: settings.currentTheme,
            sensitivity: settings.audioSensitivity,
            motionIntensity: settings.motionIntensity,
            showFPS: settings.showFPS,
            showBeatIndicator: settings.showBeatIndicator
        )
        
        let audioSettings = AudioSettings(
            inputDevice: settings.selectedInputDeviceID,
            outputDevice: settings.selectedOutputDeviceID,
            audioSource: settings.selectedAudioSource,
            micVolume: settings.microphoneVolume,
            systemAudioVolume: settings.systemAudioVolume,
            mixInputs: settings.mixAudioInputs
        )
        
        let neuralSettings = NeuralSettings(
            enabled: settings.neuralEngineEnabled,
            beatSensitivity: settings.beatSensitivity,
            patternSensitivity: settings.patternSensitivity,
            emotionalSensitivity: settings.emotionalSensitivity
        )
        
        return Preset(
            name: name,
            description: description,
            visualSettings: visualSettings,
            audioSettings: audioSettings,
            neuralSettings: neuralSettings
        )
    }
    
    /// Apply this preset to the application settings
    /// - Parameter settings: The application settings to update
    public func applyToSettings(_ settings: AudioBloomSettings) {
        // Apply visualization settings
        settings.currentTheme = visualSettings.theme
        settings.audioSensitivity = visualSettings.sensitivity
        settings.motionIntensity = visualSettings.motionIntensity
        settings.showFPS = visualSettings.showFPS
        settings.showBeatIndicator = visualSettings.showBeatIndicator
        
        // Apply audio settings
        settings.selectedInputDeviceID = audioSettings.inputDevice
        settings.selectedOutputDeviceID = audioSettings.outputDevice
        settings.selectedAudioSource = audioSettings.audioSource
        settings.microphoneVolume = audioSettings.micVolume
        settings.systemAudioVolume = audioSettings.systemAudioVolume
        settings.mixAudioInputs = audioSettings.mixInputs
        
        // Apply neural settings
        settings.neuralEngineEnabled = neuralSettings.enabled
        settings.beatSensitivity = neuralSettings.beatSensitivity
        settings.patternSensitivity = neuralSettings.patternSensitivity
        settings.emotionalSensitivity = neuralSettings.emotionalSensitivity
    }
}

/// Visualization settings for a preset
public struct VisualizationSettings: Codable, Equatable {
    /// The visual theme
    public var theme: AudioBloomCore.VisualTheme
    
    /// Audio visualization sensitivity (0.0-1.0)
    public var sensitivity: Double
    
    /// Motion intensity (0.0-1.0)
    public var motionIntensity: Double
    
    /// Whether to show FPS counter
    public var showFPS: Bool
    
    /// Whether to show beat indicator
    public var showBeatIndicator: Bool
    
    /// Initialize with default settings
    public init() {
        theme = AudioBloomCore.VisualTheme.classic
        sensitivity = 0.75
        motionIntensity = 0.8
        showFPS = false
        showBeatIndicator = true
    }
    
    /// Initialize with custom settings
    public init(theme: AudioBloomCore.VisualTheme,
                sensitivity: Double,
                motionIntensity: Double,
                showFPS: Bool,
                showBeatIndicator: Bool) {
        self.theme = theme
        self.sensitivity = sensitivity
        self.motionIntensity = motionIntensity
        self.showFPS = showFPS
        self.showBeatIndicator = showBeatIndicator
    }
}

/// Audio settings for a preset
public struct AudioSettings: Codable, Equatable {
    /// Input device identifier
    public var inputDevice: String?
    
    /// Output device identifier
    public var outputDevice: String?
    
    /// Selected audio source type
    public var audioSource: String
    
    /// Microphone volume level (0.0-1.0)
    public var micVolume: Float
    
    /// System audio volume level (0.0-1.0)
    public var systemAudioVolume: Float
    
    /// Whether to mix audio inputs
    public var mixInputs: Bool
    
    /// Initialize with default settings
    public init() {
        inputDevice = nil
        outputDevice = nil
        audioSource = "Microphone"
        micVolume = 1.0
        systemAudioVolume = 0.8
        mixInputs = false
    }
    
    /// Initialize with custom settings
    public init(inputDevice: String?,
                outputDevice: String?,
                audioSource: String,
                micVolume: Float,
                systemAudioVolume: Float,
                mixInputs: Bool) {
        self.inputDevice = inputDevice
        self.outputDevice = outputDevice
        self.audioSource = audioSource
        self.micVolume = micVolume
        self.systemAudioVolume = systemAudioVolume
        self.mixInputs = mixInputs
    }
}

/// Neural engine settings for a preset
public struct NeuralSettings: Codable, Equatable {
    /// Whether neural enhancement is enabled
    public var enabled: Bool
    
    /// Beat detection sensitivity (0.0-1.0)
    public var beatSensitivity: Float
    
    /// Pattern recognition sensitivity (0.0-1.0)
    public var patternSensitivity: Float
    
    /// Emotional content analysis sensitivity (0.0-1.0)
    public var emotionalSensitivity: Float
    
    /// Initialize with default settings
    public init() {
        enabled = true
        beatSensitivity = 0.65
        patternSensitivity = 0.7
        emotionalSensitivity = 0.5
    }
    
    /// Initialize with custom settings
    public init(enabled: Bool,
                beatSensitivity: Float,
                patternSensitivity: Float,
                emotionalSensitivity: Float) {
        self.enabled = enabled
        self.beatSensitivity = beatSensitivity
        self.patternSensitivity = patternSensitivity
        self.emotionalSensitivity = emotionalSensitivity
    }
}

/// Manages preset operations including saving, loading, and applying presets
public class PresetManager: ObservableObject {
    /// List of available presets
    @Published public private(set) var presets: [Preset] = []
    
    /// The currently selected preset
    @Published public private(set) var currentPreset: Preset?
    
    /// Application settings
    private let settings: AudioBloomSettings
    
    /// URL for preset storage
    private let presetsURL: URL
    
    /// A PassthroughSubject to notify when presets change
    public let presetsDidChange = PassthroughSubject<Void, Never>()
    
    /// Initialize with application settings
    /// - Parameter settings: The application settings
    public init(settings: AudioBloomSettings) {
        self.settings = settings
        
        // Get the documents directory for saving presets
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        self.presetsURL = documentsDirectory.appendingPathComponent("AudioBloomPresets")
        
        // Create presets directory if it doesn't exist
        try? FileManager.default.createDirectory(at: presetsURL, withIntermediateDirectories: true)
        
        // Load existing presets
        loadPresets()
        
        // Create default presets if none exist
        if presets.isEmpty {
            createDefaultPresets()
        }
    }
    
    /// Loads all saved presets from disk
    private func loadPresets() {
        do {
            // Get all preset files
            let fileURLs = try FileManager.default.contentsOfDirectory(
                at: presetsURL,
                includingPropertiesForKeys: nil
            ).filter { $0.pathExtension == "abpreset" }
            
            // Load and decode each preset
            var loadedPresets: [Preset] = []
            for fileURL in fileURLs {
                if let preset = loadPresetFromURL(fileURL) {
                    loadedPresets.append(preset)
                }
            }
            
            // Sort by name
            presets = loadedPresets.sorted { $0.name < $1.name }
        } catch {
            print("Error loading presets: \(error)")
        }
    }
    
    /// Loads a single preset from the given URL
    /// - Parameter url: The URL to load from
    /// - Returns: The loaded preset, or nil if loading failed
    private func loadPresetFromURL(_ url: URL) -> Preset? {
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            let preset = try decoder.decode(Preset.self, from: data)
            return preset
        } catch {
            print("Error loading preset from \(url): \(error)")
            return nil
        }
    }
    
    /// Creates default presets if none exist
    private func createDefaultPresets() {
        let defaultPresets = [
            Preset(
                name: "Classic Visualizer",
                description: "Traditional audio visualization with microphone input",
                visualSettings: VisualizationSettings(
                    theme: AudioBloomCore.VisualTheme.classic,
                    sensitivity: 0.75,
                    motionIntensity: 0.8,
                    showFPS: false,
                    showBeatIndicator: true
                ),
                audioSettings: AudioSettings(
                    inputDevice: nil,
                    outputDevice: nil,
                    audioSource: "Microphone",
                    micVolume: 1.0,
                    systemAudioVolume: 0.0,
                    mixInputs: false
                ),
                neuralSettings: NeuralSettings(
                    enabled: false,
                    beatSensitivity: 0.65,
                    patternSensitivity: 0.7,
                    emotionalSensitivity: 0.5
                )
            ),
            Preset(
                name: "Neural System Audio",
                description: "Neural-enhanced visualization of system audio",
                visualSettings: VisualizationSettings(
                    theme: AudioBloomCore.VisualTheme.neon,
                    sensitivity: 0.8,
                    motionIntensity: 0.9,
                    showFPS: false,
                    showBeatIndicator: true
                ),
                audioSettings: AudioSettings(
                    inputDevice: nil,
                    outputDevice: nil,
                    audioSource: "System Audio",
                    micVolume: 0.0,
                    systemAudioVolume: 1.0,
                    mixInputs: false
                ),
                neuralSettings: NeuralSettings(
                    enabled: true,
                    beatSensitivity: 0.7,
                    patternSensitivity: 0.8,
                    emotionalSensitivity: 0.6
                )
            ),
            Preset(
                name: "Cosmic Mix",
                description: "Cosmic theme with mixed audio inputs and maximum neural enhancement",
                visualSettings: VisualizationSettings(
                    theme: AudioBloomCore.VisualTheme.cosmic,
                    sensitivity: 0.85,
                    motionIntensity: 0.95,
                    showFPS: true,
                    showBeatIndicator: true
                ),
                audioSettings: AudioSettings(
                    inputDevice: nil,
                    outputDevice: nil,
                    audioSource: "Mixed (Mic + System)",
                    micVolume: 0.7,
                    systemAudioVolume: 0.7,
                    mixInputs: true
                ),
                neuralSettings: NeuralSettings(
                    enabled: true,
                    beatSensitivity: 0.8,
                    patternSensitivity: 0.9,
                    emotionalSensitivity: 0.7
                )
            )
        ]
        
        // Save default presets
        for preset in defaultPresets {
            savePreset(preset)
        }
        
        // Reload presets
        loadPresets()
    }
    
    /// Saves a preset to disk
    /// - Parameter preset: The preset to save
    /// - Returns: Whether the save was successful
    @discardableResult
    public func savePreset(_ preset: Preset) -> Bool {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(preset)
            
            let fileURL = presetsURL.appendingPathComponent("\(preset.id.uuidString).abpreset")
            try data.write(to: fileURL)
            
            // Reload presets after saving to update the list
            loadPresets()
            
            // Notify observers
            presetsDidChange.send()
            return true
        } catch {
            print("Error saving preset: \(error)")
            return false
        }
    }
    
    /// Deletes a preset
    /// - Parameter preset: The preset to delete
    /// - Returns: Whether the deletion was successful
    @discardableResult
    public func deletePreset(_ preset: Preset) -> Bool {
        let fileURL = presetsURL.appendingPathComponent("\(preset.id.uuidString).abpreset")
        
        do {
            try FileManager.default.removeItem(at: fileURL)
            
            // Remove from current list
            presets.removeAll { $0.id == preset.id }
            
            // Clear current preset if it was deleted
            if currentPreset?.id == preset.id {
                currentPreset = nil
            }
            
            // Notify observers
            presetsDidChange.send()
            return true
        } catch {
            print("Error deleting preset: \(error)")
            return false
        }
    }
    
    /// Updates an existing preset
    /// - Parameters:
    ///   - presetId: The ID of the preset to update
    ///   - updatedPreset: The updated preset data
    /// - Returns: Whether the update was successful
    @discardableResult
    public func updatePreset(presetId: UUID, updatedPreset: Preset) -> Bool {
        // First, make sure the preset exists
        guard presets.contains(where: { $0.id == presetId }) else {
            print("Cannot update preset: preset with ID \(presetId) not found")
            return false
        }
        
        // Create a new preset with the updated data but preserving the ID
        var preset = updatedPreset
        
        // Update the last modified date
        var mutablePreset = preset
        mutablePreset.lastModifiedDate = Date()
        
        // Save the updated preset
        return savePreset(mutablePreset)
    }
    
    /// Creates a new preset from current settings
    /// - Parameters:
    ///   - name: Name for the preset
    ///   - description: Optional description
    /// - Returns: The newly created preset
    @discardableResult
    public func createPresetFromCurrentSettings(name: String, description: String = "") -> Preset {
        let preset = Preset.fromCurrentSettings(settings: settings, name: name, description: description)
        savePreset(preset)
        return preset
    }
    
    /// Applies a preset to the application
    /// - Parameter preset: The preset to apply
    public func applyPreset(_ preset: Preset) {
        // Update current preset
        currentPreset = preset
        
        // Apply settings from preset
        preset.applyToSettings(settings)
        
        // Save the settings
        settings.save()
    }
    
    /// Applies a preset by its ID
    /// - Parameter presetId: The ID of the preset to apply
    /// - Returns: Whether the preset was found and applied
    @discardableResult
    public func applyPresetById(_ presetId: UUID) -> Bool {
        guard let preset = presets.first(where: { $0.id == presetId }) else {
            print("Cannot apply preset: preset with ID \(presetId) not found")
            return false
        }
        
        applyPreset(preset)
        return true
    }
    
    /// Exports a preset to a specific URL for sharing
    /// - Parameters:
    ///   - preset: The preset to export
    ///   - destinationURL: The URL to export to (optional)
    /// - Returns: The URL where the preset was exported, or nil if export failed
    public func exportPreset(_ preset: Preset, to destinationURL: URL? = nil) -> URL? {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(preset)
            
            // If no destination URL is provided, export to the user's Downloads folder
            let exportURL: URL
            if let destinationURL = destinationURL {
                exportURL = destinationURL
            } else {
                let downloadsURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!
                exportURL = downloadsURL.appendingPathComponent("\(preset.name)-\(preset.id.uuidString).abpreset")
            }
            
            try data.write(to: exportURL)
            return exportURL
        } catch {
            print("Error exporting preset: \(error)")
            return nil
        }
    }
    
    /// Imports a preset from a URL
    /// - Parameter url: The URL to import from
    /// - Returns: The imported preset, or nil if import failed
    public func importPreset(from url: URL) -> Preset? {
        guard let preset = loadPresetFromURL(url) else {
            return nil
        }
        
        // Create a new preset with a unique ID to avoid conflicts
        var importedPreset = preset
        importedPreset = Preset(
            id: UUID(), // New ID
            name: preset.name,
            description: preset.description,
            createdDate: Date(), // Current date
            lastModifiedDate: Date(),
            visualSettings: preset.visualSettings,
            audioSettings: preset.audioSettings,
            neuralSettings: preset.neuralSettings
        )
        
        // Save the imported preset
        if savePreset(importedPreset) {
            return importedPreset
        } else {
            return nil
        }
    }
    
    /// Imports multiple presets from a directory
    /// - Parameter directoryURL: The directory containing presets
    /// - Returns: The number of successfully imported presets
    public func importPresetsFromDirectory(_ directoryURL: URL) -> Int {
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(
                at: directoryURL,
                includingPropertiesForKeys: nil
            ).filter { $0.pathExtension == "abpreset" }
            
            var importedCount = 0
            for fileURL in fileURLs {
                if importPreset(from: fileURL) != nil {
                    importedCount += 1
                }
            }
            
            return importedCount
        } catch {
            print("Error importing presets from directory: \(error)")
            return 0
        }
    }
    
    /// Creates a preset export package containing multiple presets
    /// - Parameter presets: The presets to include in the package
    /// - Returns: URL to the created package, or nil if package creation failed
    public func createPresetPackage(_ presets: [Preset]) -> URL? {
        do {
            // Create a temporary directory for the package
            let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            
            // Export each preset to the temp directory
            for preset in presets {
                _ = exportPreset(preset, to: tempDir.appendingPathComponent("\(preset.name)-\(preset.id.uuidString).abpreset"))
            }
            
            // Create a zip file with all presets
            let downloadsURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!
            let packageURL = downloadsURL.appendingPathComponent("AudioBloomPresets-\(Date().timeIntervalSince1970).zip")
            
            // Here we would use a zip library to create the archive
            // For example with ZIPFoundation:
            // try FileManager.default.zipItem(at: tempDir, to: packageURL)
            
            // Simulated implementation - in a real app, you would use a proper zipping mechanism
            // For this example, we'll just copy the files to the package location
            if !FileManager.default.fileExists(atPath: packageURL.path) {
                try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: true)
            }
            
            let files = try FileManager.default.contentsOfDirectory(at: tempDir, includingPropertiesForKeys: nil)
            for file in files {
                let destination = packageURL.appendingPathComponent(file.lastPathComponent)
                try FileManager.default.copyItem(at: file, to: destination)
            }
            
            // Clean up temporary directory
            try FileManager.default.removeItem(at: tempDir)
            
            return packageURL
        } catch {
            print("Error creating preset package: \(error)")
            return nil
        }
    }
    
    /// Get a preset by ID
    /// - Parameter id: The preset ID to look for
    /// - Returns: The preset if found, nil otherwise
    public func getPreset(withId id: UUID) -> Preset? {
        return presets.first { $0.id == id }
    }
    
    /// Get a preset by name
    /// - Parameter name: The preset name to look for
    /// - Returns: The preset if found, nil otherwise
    public func getPreset(withName name: String) -> Preset? {
        return presets.first { $0.name == name }
    }
    
    /// Gets preset categories based on neural settings
    /// - Returns: A dictionary mapping categories to presets
    public func getPresetsByCategory() -> [String: [Preset]] {
        var categories: [String: [Preset]] = [
            "Neural Enhanced": [],
            "Traditional": [],
            "System Audio": [],
            "Microphone": [],
            "Mixed Input": []
        ]
        
        for preset in presets {
            // Categorize by neural enhancement
            if preset.neuralSettings.enabled {
                categories["Neural Enhanced"]?.append(preset)
            } else {
                categories["Traditional"]?.append(preset)
            }
            
            // Categorize by audio source
            switch preset.audioSettings.audioSource {
            case "Microphone":
                categories["Microphone"]?.append(preset)
            case "System Audio":
                categories["System Audio"]?.append(preset)
            case "Mixed (Mic + System)":
                categories["Mixed Input"]?.append(preset)
            default:
                break
            }
        }
        
        // Remove empty categories
        categories = categories.filter { !$0.value.isEmpty }
        
        return categories
    }
}

