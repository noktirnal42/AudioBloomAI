import Foundation
import AVFoundation
import Combine
import AudioBloomCore

/// Audio Engine for capturing and processing audio data
public class AudioEngine: ObservableObject, AudioDataProvider {
    /// Published audio levels for the left and right channels
    @Published public private(set) var levels: (left: Float, right: Float) = (0, 0)
    
    /// Published frequency data from FFT analysis
    @Published public private(set) var frequencyData: [Float] = []
    
    /// Audio data publisher for subscribers
    private let audioDataPublisher = AudioDataPublisher()
    
    /// The AVAudioEngine instance for audio processing
    private let avAudioEngine = AVAudioEngine()
    
    /// FFT helper for frequency analysis
    private var fftHelper: FFTHelper?
    
    /// Audio tap node for extracting audio data
    private var audioTap: AVAudioNode?
    
    /// Processing queue for audio analysis
    private let processingQueue = DispatchQueue(label: "com.audiobloom.audioprocessing", qos: .userInteractive)
    
    /// Timer for polling audio data
    private var audioPollingTimer: Timer?
    
    /// Indicates if the audio engine is currently running
    private var isRunning = false
    
    /// Initializes a new AudioEngine
    public init() {
        // Initialize with default FFT size
        self.fftHelper = FFTHelper(fftSize: AudioBloomCore.Constants.defaultFFTSize)
        
        // Setup notification observers for audio session interruptions
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioSessionInterruption),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )
    }
    
    /// Sets up the audio session
    public func setupAudioSession() async throws {
        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            // Configure audio session for playback and recording
            try audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothA2DP])
            try audioSession.setActive(true)
            
            // Set up audio engine
            configureAudioEngine()
        } catch {
            throw AudioBloomCore.Error.audioSessionSetupFailed
        }
    }
    
    /// Starts audio capture
    public func startCapture() throws {
        guard !isRunning else { return }
        
        do {
            try avAudioEngine.start()
            startAudioPolling()
            isRunning = true
        } catch {
            throw AudioBloomCore.Error.audioEngineStartFailed
        }
    }
    
    /// Stops audio capture
    public func stopCapture() {
        guard isRunning else { return }
        
        stopAudioPolling()
        avAudioEngine.stop()
        isRunning = false
    }
    
    /// Configures the audio engine components
    private func configureAudioEngine() {
        // Reset the engine
        avAudioEngine.stop()
        avAudioEngine.reset()
        
        // Set up the input node
        let inputNode = avAudioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        
        // Create a mixer node to tap into
        let mixerNode = AVAudioMixerNode()
        avAudioEngine.attach(mixerNode)
        
        // Connect input to mixer
        avAudioEngine.connect(inputNode, to: mixerNode, format: inputFormat)
        
        // Install tap on mixer node to receive audio buffers
        let tapFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: AudioBloomCore.Constants.defaultSampleRate,
            channels: 2,
            interleaved: false
        )
        
        mixerNode.installTap(onBus: 0, bufferSize: UInt32(AudioBloomCore.Constants.defaultFFTSize), format: tapFormat) { [weak self] buffer, time in
            self?.processingQueue.async {
                self?.processAudioBuffer(buffer)
            }
        }
        
        self.audioTap = mixerNode
        
        // Prepare engine
        avAudioEngine.prepare()
    }
    
    /// Processes audio buffer data
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let fftHelper = fftHelper,
              let channelData = buffer.floatChannelData else { return }
        
        // Process left and right channels if available
        let leftChannelData = Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
        
        var rightChannelData: [Float] = []
        if buffer.format.channelCount > 1 {
            rightChannelData = Array(UnsafeBufferPointer(start: channelData[1], count: Int(buffer.frameLength)))
        } else {
            rightChannelData = leftChannelData
        }
        
        // Calculate RMS level for each channel
        let leftLevel = calculateRMSLevel(data: leftChannelData)
        let rightLevel = calculateRMSLevel(data: rightChannelData)
        
        // Perform FFT to get frequency data
        let fftData = fftHelper.performFFT(data: leftChannelData)
        
        // Update published values on main thread
        DispatchQueue.main.async { [weak self] in
            self?.levels = (left: leftLevel, right: rightLevel)
            self?.frequencyData = fftData
            
            // Publish data for subscribers
            self?.audioDataPublisher.publish(frequencyData: fftData, levels: (leftLevel, rightLevel))
        }
    }
    
    /// Calculates RMS (Root Mean Square) level from audio sample data
    private func calculateRMSLevel(data: [Float]) -> Float {
        let sumSquares = data.reduce(0) { $0 + $1 * $1 }
        return sqrt(sumSquares / Float(data.count))
    }
    
    /// Starts the audio polling timer
    private func startAudioPolling() {
        audioPollingTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / Double(AudioBloomCore.Constants.defaultFrameRate), repeats: true) { [weak self] _ in
            // This timer ensures we're regularly checking for audio data
            // even if the system isn't providing new buffers
        }
    }
    
    /// Stops the audio polling timer
    private func stopAudioPolling() {
        audioPollingTimer?.invalidate()
        audioPollingTimer = nil
    }
    
    /// Handles audio session interruptions
    @objc private func handleAudioSessionInterruption(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }
        
        switch type {
        case .began:
            stopCapture()
        case .ended:
            if let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt,
               AVAudioSession.InterruptionOptions(rawValue: optionsValue).contains(.shouldResume) {
                try? startCapture()
            }
        @unknown default:
            break
        }
    }
}

/// Helper class for performing FFT (Fast Fourier Transform) on audio data
private class FFTHelper {
    /// Size of the FFT
    private let fftSize: Int
    
    /// Initializes with the specified FFT size
    init(fftSize: Int) {
        self.fftSize = fftSize
    }
    
    /// Performs FFT on the provided audio data
    func performFFT(data: [Float]) -> [Float] {
        // In a real implementation, this would use the Accelerate framework
        // to perform an actual FFT. For now, we'll return a placeholder.
        
        // Placeholder: Return a simulated spectrum with values between 0 and 1
        var result = [Float](repeating: 0, count: fftSize / 2)
        
        // Create a simple simulated spectrum for demonstration
        for i in 0..<result.count {
            let normalizedIndex = Float(i) / Float(result.count)
            // Generate some random variation to simulate spectrum
            let baseValue = sin(normalizedIndex * 10) * 0.5 + 0.5
            let randomComponent = Float.random(in: -0.2...0.2)
            
            // Create a simulated frequency spectrum that looks somewhat realistic
            // Low frequencies often have higher energy
            let frequencyFalloff = pow(1.0 - normalizedIndex, 2.0) * 0.5
            
            // Combine components with some randomness to simulate audio reactivity
            result[i] = min(1.0, max(0.0, baseValue + randomComponent + frequencyFalloff))
        }
        
        return result
    }
}
