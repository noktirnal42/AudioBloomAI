// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency
//
// PerformanceTracker.swift
// Performance tracking for AudioBridge
//

import Foundation
import QuartzCore

/// Performance tracking for audio processing
@available(macOS 15.0, *)
class PerformanceTracker: Sendable {
    /// Number of frames processed
    private var frameCount: Int = 0
    
    /// Total processing time (seconds)
    private var totalProcessingTime: Double = 0
    
    /// Recent processing times (seconds)
    private var recentProcessingTimes: [Double] = []
    
    /// Maximum number of recent times to track
    private let maxRecentTimes = 30
    
    /// Significant events detected
    private var significantEventCount: Int = 0
    
    /// Errors encountered
    private var errorCount: Int = 0
    
    /// Tracking start time
    private var trackingStartTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
    
    /// Indicates the start of a processing operation
    func beginProcessing() {
        // Implementation in real code would track start time
    }
    
    /// Indicates the end of a processing operation
    func endProcessing() {
        // Implementation in real code would calculate elapsed time
    }
    
    /// Resets all tracking counters
    func reset() {
        frameCount = 0
        totalProcessingTime = 0
        recentProcessingTimes = []
        significantEventCount = 0
        errorCount = 0
        trackingStartTime = CFAbsoluteTimeGetCurrent()
    }
    
    /// Records a significant event
    func recordSignificantEvent() {
        significantEventCount += 1
    }
    
    /// Records an error
    func recordError() {
        errorCount += 1
    }
    
    /// Records processing time
    /// - Parameter time: Processing time in seconds
    func recordProcessingTime(_ time: Double) {
        // Add to running totals
        frameCount += 1
        totalProcessingTime += time
        
        // Add to recent times circular buffer
        if recentProcessingTimes.count >= maxRecentTimes {
            recentProcessingTimes.removeFirst()
        }
        recentProcessingTimes.append(time)
    }
    
    /// Gets the current performance metrics
    /// - Returns: Performance metrics
    func getCurrentMetrics() -> AudioBridge.PerformanceMetrics {
        let currentTime = CFAbsoluteTimeGetCurrent()
        let elapsedMinutes = (currentTime - trackingStartTime) / 60.0
        
        var metrics = AudioBridge.PerformanceMetrics()
        
        // Calculate frames per second
        if elapsedMinutes > 0 {
            metrics.framesPerSecond = Double(frameCount) / (elapsedMinutes * 60.0)
            metrics.eventsPerMinute = Double(significantEventCount) / elapsedMinutes
            metrics.errorRate = Double(errorCount) / elapsedMinutes
        }
        
        // Calculate average processing time
        if !recentProcessingTimes.isEmpty {
            let recentTotal = recentProcessingTimes.reduce(0, +)
            metrics.averageProcessingTime = (recentTotal / Double(recentProcessingTimes.count)) * 1000 // Convert to ms
        } else if frameCount > 0 {
            metrics.averageProcessingTime = (totalProcessingTime / Double(frameCount)) * 1000 // Convert to ms
        }
        
        // Calculate conversion efficiency
        metrics.conversionEfficiency = max(0, min(1, 1.0 - (metrics.averageProcessingTime / 16.0))) // Target is sub-16ms
        
        return metrics
    }
}

