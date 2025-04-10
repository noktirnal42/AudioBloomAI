
//
// AudioBridgeNotifications.swift
// Notification definitions for AudioBridge
//

import Foundation

extension Notification.Name {
    /// Notification posted when audio bridge state changes
    public static let audioBridgeStateChanged = Notification.Name("audioBridgeStateChanged")
    
    /// Notification posted when bridge encounters an error
    public static let audioBridgeError = Notification.Name("audioBridgeError")
    
    /// Notification posted when performance metrics are updated
    public static let audioBridgePerformanceUpdate = Notification.Name("audioBridgePerformanceUpdate")
}

