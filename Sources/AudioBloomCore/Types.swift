// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import Foundation

@available(macOS 15.0, *)
public enum VisualizationMode: String, CaseIterable, Identifiable, Sendable {
    case beatTracking = "Beat Tracking"
    case frequencyTracking = "Frequency Tracking"
    case tempoTracking = "Tempo Tracking"
    
    public var id: Self { self } // Required for Identifiable
    
    // Helper for UI display
    public var displayName: String {
        return self.rawValue
    }
    
    // Default visualization mode
    public static var defaultMode: VisualizationMode {
        return .frequencyTracking
    }
}

