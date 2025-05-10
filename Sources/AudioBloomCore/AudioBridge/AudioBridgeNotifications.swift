// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import Foundation

/// Notification extensions for AudioBridge
@available(macOS 15.0, *)
public extension Notification.Name {
    /// Notification posted when audio bridge state changes
    static let audioBridgeStateChanged = Notification.Name("audioBridgeStateChanged")
    
    /// Notification posted when bridge encounters an error
    static let audioBridgeError = Notification.Name("audioBridgeError")
    
    /// Notification posted when performance metrics are updated
    static let audioBridgePerformanceUpdate = Notification.Name("audioBridgePerformanceUpdate")
}

