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
}

/// Thread-safe settings manager using actor isolation
@available(macOS 15.0, *)
public actor SettingsManager {
    // MARK: - Properties
    
    /// In

