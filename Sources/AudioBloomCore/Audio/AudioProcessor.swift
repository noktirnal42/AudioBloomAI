// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import Foundation
import AVFoundation
import Accelerate
import os.log

/// Errors that can occur during audio processing
@available(macOS 15.0, *)
public enum AudioProcessorError: Error {
    case engineStartFailed
    case invalidAudioFile
    case processingFailed(String)
    case monitoringAlreadyActive
    case monitoringNotActive
    case invalidFormat
}

/// Audio processor that handles audio loading, processing, and monitoring
/// Using actor isolation for thread safety
@available(macOS 15.0, *)
public actor AudioProcessor {
    // MARK: - Types
    
    /// Progress handler type for reporting processing progress
    public typealias ProgressHandler = (Double) -> Void
    
    /// Audio level handler type for reporting audio levels
    public typealias LevelHandler = (Float) -> Void
    
    // MARK: - Properties
    
    /// Audio engine instance
    private let engine = AVAudioEngine()
    
    /// Audio player node for playback
    private var playerNode: AVAudioPlayerNode?
    
    /// Mixer node for processing
    private var mixerNode: AVAudioMixerNode?
    
    /// Flag indicating if monitoring is active
    private var isMonitoring = false
    
    /// Current audio format
    private var currentFormat: AVAudioFormat?
    
    /// Logger instance
    private let logger = Logger(subsystem: "com.audiobloomai.core", category: "AudioProcessor")
    
    // MARK: - Initialization
    
    /// Initialize audio processor
    public init() {
        setupAudioEngine()
    }
    
    /// Set up audio engine components
    private func setupAudioEngine() {
        // Create and attach player node
        let player = AVAudioPlayerNode()
        engine.attach(player)
        self.playerNode = player
        
        // Create and attach mixer node
        let mixer = AVAudioMixerNode()
        engine.attach(mixer)
        self.mixerNode = mixer
        
        // Connect player to mixer
        engine.connect(player, to: mixer, format: nil)
        
        // Connect mixer to main mixer
        engine.connect(mixer, to: engine.mainMixerNode, format: nil)
        
        logger.debug("Audio engine setup complete")
    }
    
    // MARK: - Audio Processing
    
    /// Process audio from the specified URL
    /// - Parameters:
    ///   - url: URL to audio file
    ///   - progressHandler: Handler for progress updates
    /// - Throws: AudioProcessorError
    public func processAudio(url: URL, progressHandler: @escaping ProgressHandler) async throws {
        guard let playerNode = playerNode, let mixerNode = mixerNode else {
            throw AudioProcessorError.processingFailed("Audio engine not properly initialized")
        }
        
        // Load audio file
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
            currentFormat = file.processingFormat
            logger.debug("Loaded audio file: \(url.lastPathComponent), format: \(file.processingFormat)")
        } catch {
            logger.error("Failed to load audio file: \(error.localizedDescription)")
            throw AudioProcessorError.invalidAudioFile
        }
        
        // Install tap on mixer node to process audio
        let bufferSize = AVAudioFrameCount(4096)
        mixerNode.installTap(onBus: 0, bufferSize: bufferSize, format: file.processingFormat) { [weak self] buffer, time in
            guard let self = self else { return }
            
            // Process audio buffer
            self.processBuffer(buffer)
            
            // Calculate progress
            if let nodeTime = playerNode.lastRenderTime, 
               let playerTime = playerNode.playerTime(forNodeTime: nodeTime) {
                let progress = Double(playerTime.sampleTime) / Double(file.length)
                Task { @MainActor in 
                    progressHandler(min(max(progress, 0.0), 1.0))
                }
            }
        }
        
        // Start engine if not running
        if !engine.isRunning {
            do {
                try engine.start()
                logger.debug("Audio engine started")
            } catch {
                logger.error("Failed to start audio engine: \(error.localizedDescription)")
                throw AudioProcessorError.engineStartFailed
            }
        }
        
        // Schedule file for playback
        playerNode.scheduleFile(file, at: nil)
        playerNode.play()
        
        // Wait for playback to complete
        await withCheckedContinuation { continuation in
            playerNode.scheduleBuffer(AVAudioPCMBuffer()) {
                // Remove tap when done
                self.mixerNode?.removeTap(onBus: 0)
                logger.debug("Audio processing completed")
                continuation.resume()
            }
        }
    }
    
    /// Process an audio buffer (apply effects, analyze, etc.)
    /// - Parameter buffer: Audio buffer to process
    private func processBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        
        // Process each channel
        for channel in 0..<channelCount {
            let channelPtr = channelData[channel]
            
            // Example: Apply gain (can be replaced with more complex processing)
            var gain: Float = 0.8
            vDSP_vsmul(channelPtr, 1, &gain, channelPtr, 1, vDSP_Length(frameCount))
        }
    }
    
    // MARK: - Audio Monitoring
    
    /// Start audio level monitoring
    /// - Parameter levelHandler: Handler for audio level updates
    /// - Throws: AudioProcessorError
    public func startMonitoring(levelHandler: @escaping LevelHandler) throws {
        // Check if already monitoring
        guard !isMonitoring else {
            throw AudioProcessorError.monitoringAlreadyActive
        }
        
        // Create format for monitoring
        let format = AVAudioFormat(
            standardFormatWithSampleRate: 44100,
            channels: 1
        )
        
        guard let format = format else {
            throw AudioProcessorError.invalidFormat
        }
        
        // Install tap on main mixer for level monitoring
        engine.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, time in
            guard let self = self, self.isMonitoring else { return }
            
            // Calculate audio level
            let level = self.calculateLevel(buffer: buffer)
            
            // Send level update on main actor
            Task { @MainActor in 
                levelHandler(level)
            }
        }
        
        // Start engine if not running
        if !engine.isRunning {
            do {
                try engine.start()
                logger.debug("Audio engine started for monitoring")
            } catch {
                logger.error("Failed to start audio engine for monitoring: \(error.localizedDescription)")
                throw AudioProcessorError.engineStartFailed
            }
        }
        
        isMonitoring = true
        logger.debug("Audio monitoring started")
    }
    
    /// Stop audio level monitoring
    /// - Throws: AudioProcessorError
    public func stopMonitoring() throws {
        guard isMonitoring else {
            throw AudioProcessorError.monitoringNotActive
        }
        
        // Remove tap
        engine.mainMixerNode.removeTap(onBus: 0)
        
        isMonitoring = false
        logger.debug("Audio monitoring stopped")
    }
    
    /// Calculate audio level from buffer
    /// - Parameter buffer: Audio buffer
    /// - Returns: Audio level in dB (-60 to 0)
    private func calculateLevel(buffer: AVAudioPCMBuffer) -> Float {
        guard let channelData = buffer.floatChannelData else {
            return -60.0
        }
        
        let channelPtr = channelData[0]
        let frameCount = vDSP_Length(buffer.frameLength)
        
        // Calculate RMS (root mean square)
        var rms: Float = 0.0
        vDSP_measqv(channelPtr, 1, &rms, frameCount)
        rms = sqrt(rms)
        
        // Convert to dB with lower bound
        let db = max(20.0 * log10(rms), -60.0)
        
        return db
    }
    
    // MARK: - Clean up
    
    deinit {
        // Stop and clean up engine
        if engine.isRunning {
            engine.stop()
        }
        
        if isMonitoring {
            engine.mainMixerNode.removeTap(onBus: 0)
        }
        
        logger.debug("AudioProcessor deallocated")
    }
}

