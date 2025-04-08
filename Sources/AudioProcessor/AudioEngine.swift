import Foundation
import AVFoundation
import Combine
import AudioBloomCore
import Accelerate

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
        
        // Setup notification observers for audio engine interruptions
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioEngineInterruption),
            name: NSNotification.Name.AVAudioEngineConfigurationChange,
            object: avAudioEngine
        )
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
        stopCapture()
    }
    
    /// Sets up the audio session for macOS
    public func setupAudioSession() async throws {
        // Configure audio engine for macOS
        configureAudioEngine()
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
        
        // Set up the input node - macOS typically has multiple audio devices
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
        // Use Accelerate framework for efficient RMS calculation
        var rms: Float = 0.0
        var squares = [Float](repeating: 0.0, count: data.count)
        
        // Square all samples
        vDSP_vsq(data, 1, &squares, 1, vDSP_Length(data.count))
        
        // Get mean
        vDSP_meanv(squares, 1, &rms, vDSP_Length(data.count))
        
        // Take square root
        rms = sqrt(rms)
        
        return rms
    }
    
    /// Starts the audio polling timer
    private func startAudioPolling() {
        audioPollingTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / Double(AudioBloomCore.Constants.defaultFrameRate), repeats: true) { [weak self] _ in
            // This timer ensures we're regularly checking for audio data
            // even if the system isn't providing new buffers
            
            // This is particularly useful for detecting silence
            guard let self = self else { return }
            
            // If we haven't received audio data for a while, publish zeros
            if self.isRunning && self.avAudioEngine.isRunning {
                // Only update UI if we don't have recent data
                if self.frequencyData.isEmpty {
                    DispatchQueue.main.async {
                        self.levels = (left: 0, right: 0)
                        self.frequencyData = [Float](repeating: 0, count: AudioBloomCore.Constants.defaultFFTSize / 2)
                        
                        // Publish silence data
                        self.audioDataPublisher.publish(
                            frequencyData: self.frequencyData,
                            levels: self.levels
                        )
                    }
                }
            }
        }
    }
    
    /// Stops the audio polling timer
    private func stopAudioPolling() {
        audioPollingTimer?.invalidate()
        audioPollingTimer = nil
    }
    
    /// Handles audio engine configuration changes
    @objc private func handleAudioEngineInterruption(_ notification: Notification) {
        // This is called when audio configuration changes (like unplugging a device)
        if !avAudioEngine.isRunning && isRunning {
            // Try to reconfigure and restart
            do {
                configureAudioEngine()
                try startCapture()
            } catch {
                print("Failed to restart audio engine after interruption: \(error)")
            }
        }
    }
}

/// Helper class for performing FFT (Fast Fourier Transform) on audio data
private class FFTHelper {
    /// Size of the FFT
    private let fftSize: Int
    
    /// FFT setup for real signal
    private var fftSetup: vDSP_DFT_Setup?
    
    /// Window buffer for windowing the signal before FFT
    private var window: [Float]
    
    /// Temporary buffers for FFT computation
    private var realInput: [Float]
    private var imagInput: [Float]
    private var realOutput: [Float]
    private var imagOutput: [Float]
    private var magnitude: [Float]
    
    /// Real and imaginary pointers for DSPSplitComplex
    private var realPtr: UnsafeMutablePointer<Float>?
    private var imagPtr: UnsafeMutablePointer<Float>?
    
    /// Initializes with the specified FFT size
    init(fftSize: Int) {
        self.fftSize = fftSize
        
        // Create FFT setup
        self.fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.FORWARD
        )
        
        // Create Hann window for better frequency resolution
        self.window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        
        // Initialize working buffers
        self.realInput = [Float](repeating: 0, count: fftSize)
        self.imagInput = [Float](repeating: 0, count: fftSize)
        self.realOutput = [Float](repeating: 0, count: fftSize)
        self.imagOutput = [Float](repeating: 0, count: fftSize)
        self.magnitude = [Float](repeating: 0, count: fftSize/2)
        
        // Allocate memory for real and imaginary parts that will persist
        self.realPtr = UnsafeMutablePointer<Float>.allocate(capacity: fftSize)
        self.imagPtr = UnsafeMutablePointer<Float>.allocate(capacity: fftSize)
        
        // Initialize to zero
        self.realPtr?.initialize(repeating: 0, count: fftSize)
        self.imagPtr?.initialize(repeating: 0, count: fftSize)
    }
    
    deinit {
        // Clean up FFT setup
        if let fftSetup = fftSetup {
            vDSP_DFT_DestroySetup(fftSetup)
        }
        
        // Clean up allocated memory
        if let realPtr = realPtr {
            realPtr.deinitialize(count: fftSize)
            realPtr.deallocate()
        }
        
        if let imagPtr = imagPtr {
            imagPtr.deinitialize(count: fftSize)
            imagPtr.deallocate()
        }
    }
    
    /// Performs FFT on the provided audio data
    func performFFT(data: [Float]) -> [Float] {
        guard let fftSetup = fftSetup,
              let realPtr = realPtr,
              let imagPtr = imagPtr else {
            return [Float](repeating: 0, count: fftSize / 2)
        }
        
        // Prepare input data - pad or trim to fit FFT size
        let count = min(data.count, fftSize)
        for i in 0..<count {
            realInput[i] = data[i] * window[i]  // Apply window function
        }
        for i in count..<fftSize {
            realInput[i] = 0  // Zero-padding if needed
        }
        
        // Clear imaginary input (we're analyzing real signals)
        for i in 0..<fftSize {
            imagInput[i] = 0
        }
        
        // Create a properly-initialized DSPSplitComplex for output
        var splitComplex = DSPSplitComplex(
            realp: realPtr,
            imagp: imagPtr
        )
        
        // Perform the FFT
        vDSP_DFT_Execute(
            fftSetup,
            realInput, imagInput,
            &realOutput, &imagOutput
        )
        
        // Copy output to our persistent buffers
        for i in 0..<fftSize {
            realPtr[i] = realOutput[i]
            imagPtr[i] = imagOutput[i]
        }
        
        // Compute magnitude using the proper approach for complex numbers
        for i in 0..<fftSize/2 {
            let real = realPtr[i]
            let imag = imagPtr[i]
            magnitude[i] = sqrt(real * real + imag * imag)
        }
        
        // Scale the magnitudes (normalize)
        var scale = Float(1.0 / Float(fftSize))
        vDSP_vsmul(magnitude, 1, &scale, &magnitude, 1, vDSP_Length(fftSize/2))
        
        // Apply logarithmic scaling for better visualization
        var scaledMagnitude = [Float](repeating: 0, count: fftSize/2)
        for i in 0..<fftSize/2 {
            // Convert to dB with some scaling and clamping
            let logValue = 10.0 * log10f(magnitude[i] + 1e-9)
            // Normalize to 0.0-1.0 range
            let normalizedValue = (logValue + 90.0) / 90.0
            scaledMagnitude[i] = min(max(normalizedValue, 0.0), 1.0)
        }
        
        return scaledMagnitude
    }
}
