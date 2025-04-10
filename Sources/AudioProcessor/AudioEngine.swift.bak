import Foundation
import AVFoundation
import Combine
import AudioBloomCore
import Accelerate
#if os(macOS)
import CoreAudio
import CoreMedia
import AppKit
import os.log
#endif

/// Audio Engine for capturing and processing audio data
public class AudioEngine: ObservableObject, AudioDataProvider {
    /// Published audio levels for the left and right channels
    @Published public private(set) var levels: (left: Float, right: Float) = (0, 0)
    
    /// Published frequency data from FFT analysis
    @Published public private(set) var frequencyData: [Float] = []
    
    /// Available audio input devices
    @Published public private(set) var availableInputDevices: [AudioDevice] = []
    
    /// Available audio output devices
    @Published public private(set) var availableOutputDevices: [AudioDevice] = []
    
    /// Currently selected audio input device
    @Published public private(set) var selectedInputDevice: AudioDevice?
    
    /// Currently selected audio output device
    @Published public private(set) var selectedOutputDevice: AudioDevice?
    
    /// Currently active audio source
    @Published public private(set) var activeAudioSource: AudioSourceType = .microphone
    
    /// Audio data publisher for subscribers
    /// Audio data publisher for subscribers
    private let audioDataPublisher = AudioDataPublisher()
    
    /// The AVAudioEngine instance for audio processing
    private let avAudioEngine = AVAudioEngine()
    
    /// The AVAudioEngine instance for system audio (if available)
    private let systemAudioEngine = AVAudioEngine()
    
    /// FFT helper for frequency analysis
    private var fftHelper: FFTHelper?
    
    /// Logger for audio processing
    private let logger = Logger(subsystem: "com.audiobloom.audioprocessor", category: "AudioEngine")
    /// Types of audio sources
    public enum AudioSourceType: String, CaseIterable, Identifiable {
        case microphone = "Microphone"
        case systemAudio = "System Audio"
        case mixed = "Mixed (Mic + System)"
        
        public var id: String { self.rawValue }
    }
    
    /// Audio device configuration
    public struct AudioConfiguration {
        /// Whether to enable system audio capture
        public var enableSystemAudioCapture: Bool = true
        
        /// Whether to use custom audio device selection
        public var useCustomAudioDevice: Bool = false
        
        /// Volume level for microphone input (0.0-1.0)
        public var microphoneVolume: Float = 1.0
        
        /// Volume level for system audio input (0.0-1.0)
        public var systemAudioVolume: Float = 1.0
        
        /// Whether to mix inputs or use only selected source
        public var mixInputs: Bool = false
    }
    
    /// Audio device model
    public struct AudioDevice: Identifiable, Hashable {
        /// Unique identifier for the device
        public let id: String
        
        /// Human-readable name of the device
        public let name: String
        
        /// Device manufacturer
        public let manufacturer: String
        
        /// Whether this is an input device
        public let isInput: Bool
        
        /// Sample rate of the device
        public let sampleRate: Double
        
        /// Number of audio channels
        public let channelCount: Int
        
        public var description: String {
            return "\(name) (\(manufacturer))"
        }
        
        public func hash(into hasher: inout Hasher) {
            hasher.combine(id)
        }
        
        public static func == (lhs: AudioDevice, rhs: AudioDevice) -> Bool {
            return lhs.id == rhs.id
        }
    }
    
    /// Audio tap node for extracting audio data
    private var audioTap: AVAudioNode?
    
    /// Current audio configuration
    private var audioConfig = AudioConfiguration()
    
    /// System audio capture node (implementation depends on platform)
    private var systemAudioNode: AVAudioNode?
    
    /// Audio mixer for routing between sources
    private var mainMixer: AVAudioMixerNode?
    
    /// Audio device observer
    private var deviceObserver: Any?
    
    /// Processing queue for audio analysis
    private let processingQueue = DispatchQueue(label: "com.audiobloom.audioprocessing", qos: .userInteractive)
    
    
    /// Timer for polling audio data
    private var audioPollingTimer: Timer?
    
    /// Flag indicating if the audio engine is running
    private var isRunning = false
    public init() {
        // Initialize with default FFT size
        self.fftHelper = FFTHelper(fftSize: AudioBloomCore.Constants.defaultFFTSize)
        
        // Setup notification observers for audio engine interruptions
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioEngineInterruption(_:)),
            name: AVAudioEngine.configurationChangeNotification,
            object: avAudioEngine
        )
        
        // Set up device observer for monitoring audio device changes
        #if os(macOS)
        // On macOS, we use Core Audio to monitor device changes
        let deviceChangeSelector = kAudioHardwarePropertyDevices
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: deviceChangeSelector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMaster
        )
        
        // Setup callback for device changes
        let deviceListenerProc: AudioObjectPropertyListenerProc = { _, _, _, _ -> OSStatus in
            // Post notification for device changes
            NotificationCenter.default.post(name: Notification.Name("AudioDeviceListChanged"), object: nil)
            return noErr
        }
        
        let status = AudioObjectAddPropertyListener(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            deviceListenerProc,
            nil
        )
        
        if status == noErr {
            // Add observer for our custom notification
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(refreshAudioDevices),
                name: Notification.Name("AudioDeviceListChanged"),
                object: nil
            )
        } else {
            print("Failed to add audio device listener: \(status)")
        }
        #endif
        
        // Initialize available devices
        refreshAudioDevices()
        
        // Configure the audio engine
        configureAudioEngine()
    }
    
    // MARK: - Device Selection
    
    /// Sets the active audio device
    /// - Parameters:
    ///   - device: The audio device to activate
    ///   - reconfigureAudio: Whether to reconfigure the audio engine immediately
    public func setActiveDevice(_ device: AudioDevice, reconfigureAudio: Bool = true) {
        if device.isInput {
            // If we're changing input device
            guard selectedInputDevice?.id != device.id else { return }
            
            // Stop audio engine if it's running
            let wasRunning = isRunning
            if wasRunning {
                stopCapture()
            }
            
            // Update selected device
            selectedInputDevice = device
            
            // Reconfigure and restart if needed
            if reconfigureAudio {
                configureAudioEngine()
                if wasRunning {
                    try? startCapture()
                }
            }
        } else {
            // Output device handling
            selectedOutputDevice = device
            
            #if os(macOS)
            // Set output device on macOS
            setAudioDevice(id: device.id, isInput: false)
            #endif
        }
    }
    
    /// Sets the active audio source type
    /// - Parameter sourceType: The audio source type to activate
    public func setAudioSource(_ sourceType: AudioSourceType) {
        guard activeAudioSource != sourceType else { return }
        
        // Stop audio if it's running
        let wasRunning = isRunning
        if wasRunning {
            stopCapture()
        }
        
        // Update audio source
        activeAudioSource = sourceType
        
        // Update audio configuration
        switch sourceType {
        case .microphone:
            audioConfig.mixInputs = false
            audioConfig.microphoneVolume = 1.0
            audioConfig.systemAudioVolume = 0.0
        case .systemAudio:
            audioConfig.mixInputs = false
            audioConfig.microphoneVolume = 0.0
            audioConfig.systemAudioVolume = 1.0
        case .mixed:
            audioConfig.mixInputs = true
            audioConfig.microphoneVolume = 0.7
            audioConfig.systemAudioVolume = 0.7
        }
        
        // Reconfigure audio engine
        configureAudioEngine()
        
        // Restart if needed
        if wasRunning {
            try? startCapture()
        }
    }
    
    /// Adjusts volume levels for audio sources
    /// - Parameters:
    ///   - micVolume: Microphone volume (0.0-1.0)
    ///   - systemVolume: System audio volume (0.0-1.0)
    public func adjustVolumes(micVolume: Float, systemVolume: Float) {
        audioConfig.microphoneVolume = max(0, min(1, micVolume))
        audioConfig.systemAudioVolume = max(0, min(1, systemVolume))
        
        // Update mixer volumes if we have an active mixer
        if let mainMixer = mainMixer {
            if audioConfig.mixInputs && systemAudioNode != nil {
                // Update the gain on the input connections
                // This is a simple approach; in a real-world implementation
                // you would use a more sophisticated mixing approach
                mainMixer.volume = audioConfig.microphoneVolume
                systemAudioNode?.volume = audioConfig.systemAudioVolume
            }
        }
    }
    
    #if os(macOS)
    /// Sets the audio device for the system
    /// - Parameters:
    ///   - id: Device ID
    ///   - isInput: Whether this is an input device
    private func setAudioDevice(id: String, isInput: Bool) {
        // On macOS, changing audio devices requires working with Core Audio
        // This is a simplified implementation
        
        guard let deviceID = AudioDeviceID(id) else { return }
        
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: isInput ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMaster
        )
        
        // Set the default device
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let result = AudioObjectSetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            size,
            &deviceID
        )
        
        if result != 0 {
            print("Failed to set audio device: \(id), error: \(result)")
        }
    }
    #endif
    
    // MARK: - System Audio Capture
    /// Sets up system audio capture on macOS
    /// 
    /// This method configures the capture of system audio using macOS audio APIs.
    /// It uses AVCaptureSession to capture system audio output and routes it to the
    /// AVAudioEngine for processing. This implementation requires user permission
    /// to access screen recording (which includes system audio capture capability).
    ///
    /// - Throws: AudioBloomCore.Error.systemAudioCaptureSetupFailed if setup fails
    private func setupSystemAudioCapture() throws {
        #if os(macOS)
        logger.info("Setting up system audio capture")
        
        // Check if screen capturing is permitted (required for system audio)
        guard CGPreflightScreenCaptureAccess() else {
            // Request screen capture permission if not already granted
            if CGRequestScreenCaptureAccess() {
                logger.info("Screen capture access granted")
            } else {
                logger.error("Screen capture access denied - cannot capture system audio")
                throw AudioBloomCore.Error.systemAudioCaptureSetupFailed
            }
        }
        
        // Create a mixer node for system audio
        let systemMixer = AVAudioMixerNode()
        systemAudioEngine.attach(systemMixer)
        
        let mainMixer = self.mainMixer ?? AVAudioMixerNode()
        if self.mainMixer == nil {
            avAudioEngine.attach(mainMixer)
            self.mainMixer = mainMixer
        }
        
        // Set up audio capture using AVCaptureSession
        let captureSession = AVCaptureSession()
        captureSession.beginConfiguration()
        
        // Find audio device for system audio
        guard let systemAudioDevice = AVCaptureDevice.default(for: .audio) else {
            logger.error("No default audio capture device found")
            throw AudioBloomCore.Error.systemAudioCaptureSetupFailed
        }
        
        // Create capture input
        do {
            let audioInput = try AVCaptureDeviceInput(device: systemAudioDevice)
            
            if captureSession.canAddInput(audioInput) {
                captureSession.addInput(audioInput)
                logger.debug("Added system audio input to capture session")
            } else {
                logger.error("Could not add audio input to capture session")
                throw AudioBloomCore.Error.systemAudioCaptureSetupFailed
            }
            
            // Create audio data output
            let audioOutput = AVCaptureAudioDataOutput()
            let processingQueue = DispatchQueue(label: "com.audiobloom.systemaudio", qos: .userInteractive)
            
            audioOutput.setSampleBufferDelegate(self, queue: processingQueue)
            
            if captureSession.canAddOutput(audioOutput) {
                captureSession.addOutput(audioOutput)
                logger.debug("Added system audio output to capture session")
            } else {
                logger.error("Could not add audio output to capture session")
                throw AudioBloomCore.Error.systemAudioCaptureSetupFailed
            }
            
            // Configure format
            let formatDescription = audioOutput.connections.first?.audioChannels.first?.formatDescription
            
            captureSession.commitConfiguration()
            captureSession.startRunning()
            
            // Create a tap to handle system audio
            let systemAudioTap = AVAudioSourceNode { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
                guard let self = self else { return noErr }
                
                let ablPointer = UnsafeMutableAudioBufferListPointer(audioBufferList)
                
                // Access the last received system audio data
                self.systemAudioLock.lock()
                defer { self.systemAudioLock.unlock() }
                
                for frame in 0..<Int(frameCount) {
                    let frameIdx = frame % self.systemAudioBuffer.count
                    let value = self.systemAudioBuffer[frameIdx] * self.audioConfig.systemAudioVolume
                    
                    // Fill all channels with the value
                    for buffer in ablPointer {
                        let bufferPointer = UnsafeMutableBufferPointer<Float>(
                            start: buffer.mData?.assumingMemoryBound(to: Float.self),
                            count: Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
                        )
                        bufferPointer[frame] = value
                    }
                }
                
                return noErr
            }
            
            avAudioEngine.attach(systemAudioTap)
            
            // Set the system audio node
            systemAudioNode = systemAudioTap
            
            // Connect the system audio to the main mixer if we're mixing inputs
            if audioConfig.mixInputs {
                let format = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32,
                    sampleRate: AudioBloomCore.Constants.defaultSampleRate,
                    channels: 2,
                    interleaved: false
                )
                
                if let format = format {
                    avAudioEngine.connect(systemAudioTap, to: mainMixer, format: format)
                    logger.debug("Connected system audio to main mixer")
                }
            }
            
            logger.info("System audio capture setup complete")
            
        } catch {
            logger.error("Error setting up system audio capture: \(error.localizedDescription)")
            throw AudioBloomCore.Error.systemAudioCaptureSetupFailed
        }
        #endif
    }
    
    // Buffer to store system audio data
    private var systemAudioBuffer = [Float](repeating: 0, count: 8192)
    private let systemAudioLock = NSLock()
    }
    
    /// Returns an audio data publisher for subscribers
    public func getAudioDataPublisher() -> AudioDataPublisher {
        return audioDataPublisher
    }
    /// Sets up the audio session for capturing audio
    public func setupAudioSession() async throws {
        #if os(iOS) || os(tvOS) || os(watchOS)
        // iOS devices need to configure AVAudioSession
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setActive(true)
        } catch {
            throw AudioBloomCore.Error.audioSessionSetupFailed
        }
        #endif
        
        // Configure audio engine
        configureAudioEngine()
        
        // Check if system audio capture is available and set up if needed
        if audioConfig.enableSystemAudioCapture {
            do {
                try setupSystemAudioCapture()
            } catch {
                print("Warning: System audio capture not available: \(error)")
                // Continue without system audio as this is not a critical failure
            }
        }
    }
    
    /// Starts audio capture
    /// Starts audio capture
    /// 
    /// This method starts the AVAudioEngine and begins the audio capture process.
    /// It sets up all necessary components for audio processing and starts the
    /// audio polling timer.
    ///
    /// - Throws: AudioBloomCore.Error.audioEngineStartFailed if the audio engine fails to start
    public func startCapture() throws {
        guard !isRunning else { 
            logger.notice("Audio capture already running, ignoring start request")
            return 
        }
        
        logger.info("Starting audio capture")
        do {
            try avAudioEngine.start()
            if audioConfig.enableSystemAudioCapture && activeAudioSource != .microphone {
                try systemAudioEngine.start()
                logger.debug("System audio engine started")
            }
            startAudioPolling()
            isRunning = true
            logger.info("Audio capture started successfully")
        } catch {
            logger.error("Failed to start audio engine: \(error.localizedDescription)")
            throw AudioBloomCore.Error.audioEngineStartFailed
        }
    }
    /// Stops audio capture
    /// 
    /// This method stops the AVAudioEngine and ends the audio capture process.
    /// It cleans up all running audio components and stops the audio polling timer.
    public func stopCapture() {
        guard isRunning else {
            logger.notice("Audio capture not running, ignoring stop request")
            return
        }
        
        logger.info("Stopping audio capture")
        stopAudioPolling()
        avAudioEngine.stop()
        
        if audioConfig.enableSystemAudioCapture {
            systemAudioEngine.stop()
            logger.debug("System audio engine stopped")
        }
        
        isRunning = false
        logger.info("Audio capture stopped successfully")
    }
    
    /// Configures the audio engine components
    private func configureAudioEngine() {
        // Reset the engine
        avAudioEngine.stop()
        avAudioEngine.reset()
        
        // Create a main mixer node
        let mainMixer = AVAudioMixerNode()
        avAudioEngine.attach(mainMixer)
        self.mainMixer = mainMixer
        
        // Configure the input based on selected device
        configureMicrophoneInput(mixerNode: mainMixer)
        
        // Install tap on the main mixer node to receive audio buffers
        let tapFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: AudioBloomCore.Constants.defaultSampleRate,
            channels: 2,
            interleaved: false
        )
        
        mainMixer.installTap(onBus: 0, bufferSize: UInt32(AudioBloomCore.Constants.defaultFFTSize), format: tapFormat) { [weak self] buffer, time in
            self?.processingQueue.async {
                self?.processAudioBuffer(buffer)
            }
        }
        
        self.audioTap = mainMixer
        
        // Prepare engine
        avAudioEngine.prepare()
    }
    
    /// Configures the microphone input
    private func configureMicrophoneInput(mixerNode: AVAudioMixerNode) {
        let inputNode = avAudioEngine.inputNode
        
        // If we have a selected input device, try to use it
        if let selectedDevice = selectedInputDevice, audioConfig.useCustomAudioDevice {
            #if os(macOS)
            // On macOS, we need to set the device ID before using the input node
            setAudioDevice(id: selectedDevice.id, isInput: true)
            #endif
        }
        
        // Now connect the input to the mixer
        let inputFormat = inputNode.outputFormat(forBus: 0)
        avAudioEngine.connect(inputNode, to: mixerNode, format: inputFormat)
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
            // Refresh available audio devices
            refreshAudioDevices()
            
            // Try to reconfigure and restart
            do {
                configureAudioEngine()
                try startCapture()
            } catch {
                print("Failed to restart audio engine after interruption: \(error)")
            }
        }
    }
    
    // MARK: - Device Management
    
    /// Refreshes the list of available audio devices
    @objc public func refreshAudioDevices() {
        #if os(macOS)
        // Get all audio devices on macOS
        var inputDevices: [AudioDevice] = []
        var outputDevices: [AudioDevice] = []
        
        // Use Core Audio to enumerate audio devices
        var propertySize: UInt32 = 0
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMaster
        )
        
        // Get the size of the device list
        var result = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &propertySize
        )
        
        if result == 0 {
            let deviceCount = Int(propertySize) / MemoryLayout<AudioDeviceID>.size
            var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
            
            // Get the device IDs
            result = AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject),
                &propertyAddress,
                0,
                nil,
                &propertySize,
                &deviceIDs
            )
            
            if result == 0 {
                // Process each device
                for deviceID in deviceIDs {
                    // Get device name
                    var nameProperty = AudioObjectPropertyAddress(
                        mSelector: kAudioObjectPropertyName,
                        mScope: kAudioObjectPropertyScopeGlobal,
                        mElement: kAudioObjectPropertyElementMaster
                    )
                    
                    var deviceName: CFString = "" as CFString
                    var namePropertySize = UInt32(MemoryLayout<CFString>.size)
                    
                    result = AudioObjectGetPropertyData(deviceID, &nameProperty, 0, nil, &namePropertySize, &deviceName)
                    
                    if result == 0 {
                        let name = deviceName as String
                        
                        // Check if it's an output device
                        var outputScopeProperty = AudioObjectPropertyAddress(
                            mSelector: kAudioDevicePropertyStreamConfiguration,
                            mScope: kAudioDevicePropertyScopeOutput,
                            mElement: kAudioObjectPropertyElementMaster
                        )
                        
                        var outputPropertySize: UInt32 = 0
                        result = AudioObjectGetPropertyDataSize(deviceID, &outputScopeProperty, 0, nil, &outputPropertySize)
                        
                        if result == 0 && outputPropertySize > 0 {
                            // This is an output device
                            let device = AudioDevice(
                                id: String(deviceID),
                                name: name,
                                manufacturer: "Apple",
                                isInput: false,
                                sampleRate: Double(AudioBloomCore.Constants.defaultSampleRate),
                                channelCount: 2
                            )
                            outputDevices.append(device)
                        }
                        
                        // Check if it's an input device
                        var inputScopeProperty = AudioObjectPropertyAddress(
                            mSelector: kAudioDevicePropertyStreamConfiguration,
                            mScope: kAudioDevicePropertyScopeInput,
                            mElement: kAudioObjectPropertyElementMaster
                        )
                        
                        var inputPropertySize: UInt32 = 0
                        result = AudioObjectGetPropertyDataSize(deviceID, &inputScopeProperty, 0, nil, &inputPropertySize)
                        
                        if result == 0 && inputPropertySize > 0 {
                            // This is an input device
                            let device = AudioDevice(
                                id: String(deviceID),
                                name: name,
                                manufacturer: "Apple",
                                isInput: true,
                                sampleRate: Double(AudioBloomCore.Constants.defaultSampleRate),
                                channelCount: 2
                            )
                            inputDevices.append(device)
                        }
                    }
                }
            }
        }
        #else
        // For iOS, the device selection is much simpler
        let inputDevices = [AudioDevice(
            id: "default",
            name: "Default Input",
            manufacturer: "Apple",
            isInput: true,
            sampleRate: Double(AudioBloomCore.Constants.defaultSampleRate),
            channelCount: 2
        )]
        
        let outputDevices = [AudioDevice(
            id: "default",
            name: "Default Output",
            manufacturer: "Apple",
            isInput: false,
            sampleRate: Double(AudioBloomCore.Constants.defaultSampleRate),
            channelCount: 2
        )]
        #endif
        
        // Update our published properties on the main thread
        DispatchQueue.main.async {
            self.availableInputDevices = inputDevices
            self.availableOutputDevices = outputDevices
            
            // Set default devices if none selected
            if self.selectedInputDevice == nil && !inputDevices.isEmpty {
                self.selectedInputDevice = inputDevices.first
            }
            
            if self.selectedOutputDevice == nil && !outputDevices.isEmpty {
                self.selectedOutputDevice = outputDevices.first
            }
        }
    }
    
    /// Cleanup resources when the object is deallocated
    deinit {
        // Stop capture if running
        if isRunning {
            stopCapture()
        }
        
        // Remove audio tap if installed
        if let audioTap = audioTap as? AVAudioMixerNode {
            audioTap.removeTap(onBus: 0)
        }
        
        // Remove notification observers
        NotificationCenter.default.removeObserver(self)
        
        #if os(macOS)
        // Remove Core Audio property listener
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMaster
        )
        
        // The listener removal might fail, but that's okay during cleanup
        _ = AudioObjectRemovePropertyListener(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            nil,
            nil
        )
        #endif
    }

    
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
