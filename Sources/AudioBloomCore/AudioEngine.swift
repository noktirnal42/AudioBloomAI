// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import Foundation
import AVFoundation
import Accelerate

@available(macOS 15.0, *)
public class AudioEngine {
    // MARK: - Properties
    
    private let audioEngine = AVAudioEngine()
    private var playerNode: AVAudioPlayerNode?
    private var audioBuffer: AVAudioPCMBuffer?
    private var isSetup = false
    
    // Configuration
    private let sampleRate: Double
    private let channelCount: Int
    private let frameCount: AVAudioFrameCount
    
    // MARK: - Initialization
    
    public init(sampleRate: Double = 48000.0, channelCount: Int = 2, frameCount: AVAudioFrameCount = 1024) {
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.frameCount = frameCount
    }
    
    // MARK: - Public Methods
    
    /// Set up the audio engine with the proper configuration
    public func setup() async throws {
        // Prevent setup from being called multiple times
        guard !isSetup else { return }
        
        // Configure audio session
        let audioSession = AVAudioSession.sharedInstance()
        try await audioSession.setCategory(.playAndRecord, mode: .default)
        try await audioSession.setActive(true)
        
        // Create and setup player node
        playerNode = AVAudioPlayerNode()
        guard let playerNode = playerNode else {
            throw AudioEngineError.failedToCreatePlayerNode
        }
        
        // Attach and connect the player node
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: createAudioFormat())
        
        // Install a tap to process audio
        audioEngine.mainMixerNode.installTap(onBus: 0, bufferSize: frameCount, format: nil) { [weak self] buffer, time in
            self?.processAudioBuffer(buffer)
        }
        
        // Start the audio engine
        try audioEngine.start()
        isSetup = true
    }
    
    /// Play a sound from a resource file
    public func playSound(resourceName: String, resourceExtension: String) async throws {
        guard let url = Bundle.module.url(forResource: resourceName, withExtension: resourceExtension) else {
            throw AudioEngineError.resourceNotFound
        }
        
        guard isSetup, let playerNode = playerNode else {
            throw AudioEngineError.engineNotSetup
        }
        
        do {
            let file = try AVAudioFile(forReading: url)
            playerNode.scheduleFile(file, at: nil) { [weak self] in
                // Completion handler - clean up resources if needed
                Task { @MainActor in
                    // Any UI updates or cleanup would go here
                }
            }
            
            playerNode.play()
        } catch {
            throw AudioEngineError.failedToPlaySound(error)
        }
    }
    
    /// Stop the audio engine and clean up resources
    public func stop() {
        playerNode?.stop()
        audioEngine.stop()
        audioEngine.mainMixerNode.removeTap(onBus: 0)
        isSetup = false
    }
    
    // MARK: - Private Methods
    
    /// Create the proper audio format
    private func createAudioFormat() -> AVAudioFormat {
        return AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: AVAudioChannelCount(channelCount)
        )!
    }
    
    /// Process audio buffer using Accelerate framework for optimization
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameLength = Int(buffer.frameLength)
        
        // Example: Apply gain using vDSP for high-performance processing
        let gain: Float = 0.8
        
        // Process each channel
        for channel in 0..<Int(buffer.format.channelCount) {
            let channelPtr = channelData[channel]
            
            // Use vDSP to efficiently apply gain to the samples
            vDSP_vsmul(channelPtr, 1, &gain, channelPtr, 1, vDSP_Length(frameLength))
            
            // Additional processing could be done here using other vDSP functions
            // For example, apply a low-pass filter:
            // var filterCoefficients: [Float] = [0.2, 0.2, 0.2, 0.2, 0.2] // Simple moving average filter
            // vDSP_conv(channelPtr, 1, filterCoefficients, 1, channelPtr, 1, vDSP_Length(frameLength), vDSP_Length(filterCoefficients.count))
        }
    }
    
    /// Analyze audio features using vDSP
    public func analyzeAudioFeatures(buffer: AVAudioPCMBuffer) -> [String: Double] {
        guard let channelData = buffer.floatChannelData else {
            return [:]
        }
        
        let frameLength = Int(buffer.frameLength)
        let channelPtr = channelData[0] // Use first channel for analysis
        
        // Calculate RMS (Root Mean Square) - a measure of signal power
        var rms: Float = 0.0
        vDSP_measqv(channelPtr, 1, &rms, vDSP_Length(frameLength))
        rms = sqrt(rms)
        
        // Calculate peak amplitude
        var peak: Float = 0.0
        vDSP_maxmgv(channelPtr, 1, &peak, vDSP_Length(frameLength))
        
        // Calculate average value
        var average: Float = 0.0
        vDSP_meanv(channelPtr, 1, &average, vDSP_Length(frameLength))
        
        // Return the features
        return [
            "rms": Double(rms),
            "peak": Double(peak),
            "average": Double(average)
        ]
    }
}

// MARK: - Error Types

public enum AudioEngineError: Error {
    case engineNotSetup
    case failedToCreatePlayerNode
    case resourceNotFound
    case failedToPlaySound(Error)
}

