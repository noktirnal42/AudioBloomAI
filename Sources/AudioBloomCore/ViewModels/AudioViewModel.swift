// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import Foundation
import SwiftUI
import Combine
import os.log

/// View model for audio processing UI, isolated to the main actor
@available(macOS 15.0, *)
@MainActor
public class AudioViewModel: ObservableObject {
    // MARK: - Published Properties
    
    /// Indicates whether audio is currently being processed
    @Published public var isProcessing: Bool = false
    
    /// Progress of the current audio processing operation (0.0 to 1.0)
    @Published public var processingProgress: Double = 0.0
    
    /// Current error message if processing failed
    @Published public var errorMessage: String?
    
    /// Current audio level (-60dB to 0dB)
    @Published public var audioLevel: Float = -60.0
    
    // MARK: - Private Properties
    
    /// Audio processor for handling audio operations
    private let audioProcessor: AudioProcessor
    
    /// Settings manager for app configuration
    private let settingsManager: SettingsManager
    
    /// Performance monitor for tracking metrics
    private let performanceMonitor: PerformanceMonitor
    
    /// Logger instance
    private let logger = Logger(subsystem: "com.audiobloomai.core", category: "ViewModel")
    
    /// Cancellable set for subscriptions
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    
    /// Initialize with dependencies
    /// - Parameters:
    ///   - audioProcessor: Audio processor instance
    ///   - settingsManager: Settings manager instance
    ///   - performanceMonitor: Performance monitor instance
    public init(
        audioProcessor: AudioProcessor,
        settingsManager: SettingsManager,
        performanceMonitor: PerformanceMonitor
    ) {
        self.audioProcessor = audioProcessor
        self.settingsManager = settingsManager
        self.performanceMonitor = performanceMonitor
    }
    
    // MARK: - Public Methods
    
    /// Process audio from the specified URL
    /// - Parameter url: URL to audio file
    public func processAudio(url: URL) async {
        // Reset state
        isProcessing = true
        processingProgress = 0.0
        errorMessage = nil
        
        // Start performance measurement
        let measurementId = performanceMonitor.beginMeasuring(operation: "AudioProcessing")
        
        do {
            // Process audio using actor-isolated processor
            try await audioProcessor.processAudio(url: url) { [weak self] progress in
                // Update progress on main actor (guaranteed by @MainActor annotation)
                self?.processingProgress = progress
            }
            
            logger.info("Audio processing completed successfully")
        } catch {
            // Handle errors on main actor
            errorMessage = error.localizedDescription
            logger.error("Audio processing failed: \(error.localizedDescription)")
        }
        
        // End performance measurement
        performanceMonitor.endMeasuring(id: measurementId)
        
        // Update state
        isProcessing = false
        processingProgress = 1.0
    }
    
    /// Update audio settings
    /// - Parameters:
    ///   - key: Setting key
    ///   - value: Setting value
    public func updateSetting(key: String, value: Any) async {
        do {
            // Update setting using actor-isolated settings manager
            try await settingsManager.setSetting(key: key, value: value)
            logger.debug("Updated setting: \(key)")
        } catch {
            errorMessage = "Failed to update setting: \(error.localizedDescription)"
            logger.error("Failed to update setting: \(error.localizedDescription)")
        }
    }
    
    /// Load user settings from storage
    public func loadSettings() async {
        do {
            // Load settings from actor-isolated settings manager
            let settings = try await settingsManager.getAllSettings()
            logger.debug("Loaded \(settings.count) settings")
        } catch {
            errorMessage = "Failed to load settings: \(error.localizedDescription)"
            logger.error("Failed to load settings: \(error.localizedDescription)")
        }
    }
    
    /// Start real-time audio monitoring
    public func startAudioMonitoring() async {
        do {
            // Start audio monitoring using actor-isolated processor
            try await audioProcessor.startMonitoring { [weak self] level in
                // Update audio level on main actor (guaranteed by @MainActor annotation)
                self?.audioLevel = level
            }
            logger.debug("Started audio monitoring")
        } catch {
            errorMessage = "Failed to start audio monitoring: \(error.localizedDescription)"
            logger.error("Failed to start audio monitoring: \(error.localizedDescription)")
        }
    }
    
    /// Stop real-time audio monitoring
    public func stopAudioMonitoring() async {
        do {
            // Stop audio monitoring using actor-isolated processor
            try await audioProcessor.stopMonitoring()
            logger.debug("Stopped audio monitoring")
        } catch {
            errorMessage = "Failed to stop audio monitoring: \(error.localizedDescription)"
            logger.error("Failed to stop audio monitoring: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Memory Management
    
    deinit {
        cancellables.forEach { $0.cancel() }
        logger.debug("AudioViewModel deallocated")
    }
}

