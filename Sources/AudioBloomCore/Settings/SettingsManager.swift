// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import Foundation
import os.log

/// Errors that can occur during settings operations
@available(macOS 15.0, *)
public enum SettingsError: Error {
    case keyNotFound
    case invalidType
    case persistenceFailed
    case loadingFailed
    case serializationFailed
    case deserializationFailed
}

/// Thread-safe settings manager using actor isolation
@available(macOS 15.0, *)
public actor SettingsManager {
    // MARK: - Properties
    
    /// In-memory settings storage
    private var settings: [String: Any] = [:]
    
    /// UserDefaults for persistence
    private let userDefaults: UserDefaults
    
    /// App group identifier (if using shared settings)
    private let appGroupIdentifier: String?
    
    /// Logger instance
    private let logger = Logger(subsystem: "com.audiobloomai.core", category: "SettingsManager")
    
    /// Default settings keys
    public enum DefaultKeys: String, CaseIterable {
        case theme = "app_theme"
        case audioQuality = "audio_quality"
        case processingEnabled = "processing_enabled"
        case visualizationMode = "visualization_mode"
        case lastOpenedFile = "last_opened_file"
        case volumeLevel = "volume_level"
        case effectsEnabled = "effects_enabled"
        case analysisDepth = "analysis_depth"
        case aiModelType = "ai_model_type"
        case refreshRate = "refresh_rate"
    }
    
    // MARK: - Initialization
    
    /// Initialize the settings manager
    /// - Parameters:
    ///   - appGroupIdentifier: App group identifier for shared settings (optional)
    ///   - initialSettings: Initial settings values (optional)
    public init(appGroupIdentifier: String? = nil, initialSettings: [String: Any]? = nil) {
        self.appGroupIdentifier = appGroupIdentifier
        
        // Initialize UserDefaults based on app group identifier
        if let groupIdentifier = appGroupIdentifier {
            if let groupDefaults = UserDefaults(suiteName: groupIdentifier) {
                userDefaults = groupDefaults
                logger.debug("Using shared UserDefaults with group: \(groupIdentifier)")
            } else {
                userDefaults = .standard
                logger.warning("Failed to create shared UserDefaults, using standard")
            }
        } else {
            userDefaults = .standard
            logger.debug("Using standard UserDefaults")
        }
        
        // Load saved settings
        loadSettingsFromDefaults()
        
        // Apply initial settings if provided
        if let initialValues = initialSettings {
            for (key, value) in initialValues {
                settings[key] = value
            }
            logger.debug("Applied \(initialValues.count) initial settings")
        }
    }
    
    // MARK: - Core Methods
    
    /// Get all settings
    /// - Returns: Dictionary of all settings
    public func getAllSettings() -> [String: Any] {
        return settings
    }
    
    /// Set a setting value
    /// - Parameters:
    ///   - key: Setting key
    ///   - value: Setting value
    /// - Throws: SettingsError
    public func setSetting(key: String, value: Any) throws {
        // Ensure value is a supported type
        guard isValidSettingType(value) else {
            logger.error("Invalid setting type for key: \(key)")
            throw SettingsError.invalidType
        }
        
        // Store value in memory
        settings[key] = value
        logger.debug("Set setting \(key) to \(String(describing: value))")
        
        // Persist to UserDefaults
        try persistToDefaults(key: key, value: value)
    }
    
    /// Get a setting value
    /// - Parameter key: Setting key
    /// - Returns: Setting value
    /// - Throws: SettingsError
    public func getSetting(key: String) throws -> Any {
        guard let value = settings[key] else {
            logger.error("Setting not found for key: \(key)")
            throw SettingsError.keyNotFound
        }
        
        logger.debug("Retrieved setting for key: \(key)")
        return value
    }
    
    /// Get a setting with a specific type
    /// - Parameters:
    ///   - key: Setting key
    ///   - type: Expected value type
    /// - Returns: Typed setting value
    /// - Throws: SettingsError
    public func getTypedSetting<T>(key: String, type: T.Type) throws -> T {
        let value = try getSetting(key: key)
        
        guard let typedValue = value as? T else {
            logger.error("Type mismatch for key: \(key), expected: \(type), got: \(type(of: value))")
            throw SettingsError.invalidType
        }
        
        return typedValue
    }
    
    /// Remove a setting
    /// - Parameter key: Setting key
    /// - Throws: SettingsError
    public func removeSetting(key: String) throws {
        settings.removeValue(forKey: key)
        userDefaults.removeObject(forKey: key)
        try synchronizeDefaults()
        logger.debug("Removed setting for key: \(key)")
    }
    
    /// Reset all settings to default values
    /// - Throws: SettingsError
    public func resetToDefaults() throws {
        settings = [:]
        
        // Clear UserDefaults for all known keys
        for key in DefaultKeys.allCases {
            userDefaults.removeObject(forKey: key.rawValue)
        }
        
        try synchronizeDefaults()
        logger.debug("Reset all settings to defaults")
    }
    
    // MARK: - Helper Methods
    
    /// Load settings from UserDefaults
    private func loadSettingsFromDefaults() {
        // Load all settings from UserDefaults
        for key in DefaultKeys.allCases {
            if let value = userDefaults.object(forKey: key.rawValue) {
                settings[key.rawValue] = value
            }
        }
        
        logger.debug("Loaded \(settings.count) settings from UserDefaults")
    }
    
    /// Persist a setting to UserDefaults
    /// - Parameters:
    ///   - key: Setting key
    ///   - value: Setting value
    /// - Throws: SettingsError
    private func persistToDefaults(key: String, value: Any) throws {
        // Store value in UserDefaults
        userDefaults.set(value, forKey: key)
        
        // Synchronize to ensure persistence
        try synchronizeDefaults()
    }
    
    /// Synchronize UserDefaults to ensure persistence
    /// - Throws: SettingsError
    private func synchronizeDefaults() throws {
        if #available(macOS 15.0, *) {
            // In macOS 15+, synchronize is automatic
            return
        } else {
            // For compatibility with older systems
            if !userDefaults.synchronize() {
                logger.error("Failed to synchronize UserDefaults")
                throw SettingsError.persistenceFailed
            }
        }
    }
    
    /// Check if a value's type is valid for settings
    /// - Parameter value: Value to check
    /// - Returns: Whether the type is valid
    private func isValidSettingType(_ value: Any) -> Bool {
        return value is String ||
               value is Int ||
               value is Float ||
               value is Double ||
               value is Bool ||
               value is Date ||
               value is [String] ||
               value is [Int] ||
               value is [Float] ||
               value is [Double] ||
               value is [String: String] ||
               value is [String: Int] ||
               value is [String: Double] ||
               value is [String: Bool]
    }
    
    // MARK: - Import/Export
    
    /// Export settings to a dictionary
    /// - Returns: Settings dictionary
    /// - Throws: SettingsError
    public func exportSettings() -> [String: Any] {
        return settings
    }
    
    /// Import settings from a dictionary
    /// - Parameter dictionary: Settings dictionary
    /// - Throws: SettingsError
    public func importSettings(from dictionary: [String: Any]) throws {
        for (key, value) in dictionary {
            try setSetting(key: key, value: value)
        }
        
        logger.debug("Imported \(dictionary.count) settings")
    }
    
    /// Export settings to JSON data
    /// - Returns: JSON data
    /// - Throws: SettingsError
    public func exportToJSON() throws -> Data {
        let exportable = settings.filter { isValidSettingType($0.value) }
        
        do {
            let data = try JSONSerialization.data(withJSONObject: exportable, options: .prettyPrinted)
            logger.debug("Exported \(exportable.count) settings to JSON")
            return data
        } catch {
            logger.error("Failed to serialize settings to JSON: \(error.localizedDescription)")
            throw SettingsError.serializationFailed
        }
    }
    
    /// Import settings from JSON data
    /// - Parameter data: JSON data
    /// - Throws: SettingsError
    public func importFromJSON(data: Data) throws {
        do {
            guard let dictionary = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                logger.error("Invalid JSON format for settings")
                throw SettingsError.deserializationFailed
            }
            
            try importSettings(from: dictionary)
        } catch {
            logger.error("Failed to deserialize settings from JSON: \(error.localizedDescription)")
            throw SettingsError.deserializationFailed
        }
    }
    
    // MARK: - Type-Specific Convenience Methods
    
    /// Get a string setting
    /// - Parameter key: Setting key
    /// - Returns: String value
    /// - Throws: SettingsError
    public func getString(key: String) throws -> String {
        return try getTypedSetting(key: key, type: String.self)
    }
    
    /// Get an integer setting
    /// - Parameter key: Setting key
    /// - Returns: Integer value
    /// - Throws: SettingsError
    public func getInt(key: String) throws -> Int {
        return try getTypedSetting(key: key, type: Int.self)
    }
    
    /// Get a double setting
    /// - Parameter key: Setting key
    /// - Returns: Double value
    /// - Throws: SettingsError
    public func getDouble(key: String) throws -> Double {
        return try getTypedSetting(key: key, type: Double.self)
    }
    
    /// Get a boolean setting
    /// - Parameter key: Setting key
    /// - Returns: Boolean value
    /// - Throws: SettingsError
    public func getBool(key: String) throws -> Bool {
        return try getTypedSetting(key: key, type: Bool.self)
    }
    
    /// Get a date setting
    /// - Parameter key: Setting key
    /// - Returns: Date value
    /// - Throws: SettingsError
    public func getDate(key: String) throws -> Date {
        return try getTypedSetting(key: key, type: Date.self)
    }
    
    /// Get a string array setting
    /// - Parameter key: Setting key
    /// - Returns: String array value
    /// - Throws: SettingsError
    public func getStringArray(key: String) throws -> [String] {
        return try getTypedSetting(key: key, type: [String].self)
    }
}

