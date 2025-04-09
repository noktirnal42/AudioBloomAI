import Cocoa
import MetalKit
import AVFoundation
import Accelerate

class ViewController: NSViewController {
    
    // MARK: - Properties
    
    // Metal rendering
    private var metalView: MTKView!
    private var audioVisualizer: AudioVisualizer!
    private var device: MTLDevice!
    
    // Audio engine components
    private var audioEngine: AVAudioEngine!
    private var audioPlayerNode: AVAudioPlayerNode!
    private var audioFile: AVAudioFile?
    private var fftMagnitudes: [Float] = []
    private var audioSamples: [Float] = []
    
    // UI components
    private var progressBar: NSProgressIndicator!
    private var timeLabel: NSTextField!
    private var visualizationTypeSegment: NSSegmentedControl!
    private var volumeSlider: NSSlider!
    private var amplitudeSlider: NSSlider!
    
    // Audio state
    private var isPlaying = false
    private var isAudioFileLoaded = false
    private var isRecordingFromMic = false
    private var currentAudioTime: TimeInterval = 0
    private var currentVisualizationType: VisualizationType = .waveform
    
    // Audio analysis
    private let fftSize = 1024
    private var fftSetup: vDSP_DFT_Setup?
    private var audioMagnitude: Float = 1.0
    
    // Timer for UI updates
    private var uiUpdateTimer: Timer?
    // MARK: - View Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupUI()
        setupMetalRendering()
        setupAudioEngine()
        setupFFT()
        
        // Set default amplitude value
        setAmplitude(1.0)
        
        // Start the UI update timer
        startUIUpdateTimer()
    }
    
    override func viewDidAppear() {
        super.viewDidAppear()
        
        // Configure the window through its controller
        if let windowController = view.window?.windowController as? WindowController {
            configureWindowControllerActions(windowController)
        }
        
        // Request microphone permission if not already granted
        requestMicrophonePermission()
    }
    
    override func viewWillDisappear() {
        super.viewWillDisappear()
        stopAudio()
    }
    
    deinit {
        // Clean up audio resources
        if let fftSetup = fftSetup {
            vDSP_DFT_DestroySetup(fftSetup)
        }
        
        // Clean up visualization resources
        audioVisualizer?.cleanup()
        
        // Stop audio engine
        audioEngine?.stop()
        
        // Stop UI update timer
        uiUpdateTimer?.invalidate()
        uiUpdateTimer = nil
        
        // Remove audio taps
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.outputNode.removeTap(onBus: 0)
    }
    
    private func startUIUpdateTimer() {
        uiUpdateTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.updateUIForAudioPlayback()
        }
    }
    
    private func configureWindowControllerActions(_ windowController: WindowController) {
        // Connect toolbar actions with view controller methods
        windowController.openFileAction = { [weak self] in
            self?.openAudioFile()
        }
        
        windowController.playPauseAction = { [weak self] in
            self?.togglePlayPause()
        }
        
        windowController.recordAction = { [weak self] in
            self?.startMicrophoneInput()
        }
        
        windowController.stopAction = { [weak self] in
            self?.stopAudio()
        }
        
        windowController.effectsAction = { [weak self] in
            self?.showEffectsPanel()
        }
    }
    
    // MARK: - UI Setup
    
    private func setupUI() {
        // Configure the main view
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.black.cgColor
        
        // Create Metal view for visualization
        metalView = MTKView(frame: NSRect(x: 0, y: 50, width: view.bounds.width, height: view.bounds.height - 100))
        metalView.translatesAutoresizingMaskIntoConstraints = false
        metalView.framebufferOnly = false
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.1, alpha: 1.0)
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.sampleCount = 1
        view.addSubview(metalView)
        
        // Add progress bar for audio timeline
        progressBar = NSProgressIndicator(frame: NSRect(x: 20, y: 20, width: view.bounds.width - 160, height: 20))
        progressBar.translatesAutoresizingMaskIntoConstraints = false
        progressBar.minValue = 0
        progressBar.maxValue = 100
        progressBar.doubleValue = 0
        progressBar.isIndeterminate = false
        progressBar.style = .bar
        view.addSubview(progressBar)
        
        // Add time label
        timeLabel = NSTextField(labelWithString: "00:00")
        timeLabel.translatesAutoresizingMaskIntoConstraints = false
        timeLabel.alignment = .right
        timeLabel.textColor = .white
        timeLabel.backgroundColor = .clear
        timeLabel.isBezeled = false
        timeLabel.isEditable = false
        view.addSubview(timeLabel)
        
        // Add visualization type selector
        visualizationTypeSegment = NSSegmentedControl(labels: ["Waveform", "Spectrum", "Circular"], trackingMode: .selectOne, target: self, action: #selector(visualizationTypeChanged(_:)))
        visualizationTypeSegment.translatesAutoresizingMaskIntoConstraints = false
        visualizationTypeSegment.selectedSegment = 0
        view.addSubview(visualizationTypeSegment)
        
        // Add volume slider
        volumeSlider = NSSlider(value: 0.8, minValue: 0, maxValue: 1, target: self, action: #selector(volumeChanged(_:)))
        volumeSlider.translatesAutoresizingMaskIntoConstraints = false
        volumeSlider.isContinuous = true
        volumeSlider.controlSize = .regular
        view.addSubview(volumeSlider)
        
        // Add amplitude slider for visualization intensity
        amplitudeSlider = NSSlider(value: 1.0, minValue: 0.1, maxValue: 5.0, target: self, action: #selector(amplitudeChanged(_:)))
        amplitudeSlider.translatesAutoresizingMaskIntoConstraints = false
        amplitudeSlider.isContinuous = true
        amplitudeSlider.controlSize = .regular
        view.addSubview(amplitudeSlider)
        
        // Add slider labels
        let volumeLabel = NSTextField(labelWithString: "Volume")
        volumeLabel.translatesAutoresizingMaskIntoConstraints = false
        volumeLabel.textColor = .white
        volumeLabel.backgroundColor = .clear
    
    // MARK: - Metal Setup
    
    private func setupMetalRendering() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        metalView.device = device
        
        // Configure the Metal view
        metalView.framebufferOnly = false
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.1, alpha: 1.0)
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.preferredFramesPerSecond = 60
        
        // Create the command queue
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Could not create Metal command queue")
        }
        self.commandQueue = commandQueue
        
        // Create the render pipeline
        createRenderPipeline()
    }
    
    private func createRenderPipeline() {
        // Normally we would load shader functions from a Metal library
        // For this example, we'll use a very basic pipeline
        
        // This is a placeholder - in a real app, you would create a .metal file with proper shaders
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        
        // For development purposes, we'll create a basic pipeline without shader functions
        // In a real app, you would do:
        // let library = device.makeDefaultLibrary()
        // let vertexFunction = library?.makeFunction(name: "vertexShader")
        // let fragmentFunction = library?.makeFunction(name: "fragmentShader")
        // pipelineDescriptor.vertexFunction = vertexFunction
        // pipelineDescriptor.fragmentFunction = fragmentFunction
        
        do {
            // For now, we'll just log that this would be implemented with real shaders
            print("In a real implementation, proper Metal shaders would be used for visualization")
            
            // Placeholder for the render pipeline state that would be created
            // pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create render pipeline state: \(error)")
        }
    }
    
    // MARK: - FFT Setup
    
    private func setupFFT() {
        // Create an FFT setup for the specified size
        fftSetup = vDSP_DFT_zrop_CreateSetup(nil, UInt(fftSize), vDSP_DFT_Direction.FORWARD)
        
        // Initialize arrays for FFT data
        fftMagnitudes = [Float](repeating: 0, count: fftSize / 2)
        audioSamples = [Float](repeating: 0, count: fftSize)
    }
    
    // MARK: - Audio Setup
    
    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        audioPlayerNode = AVAudioPlayerNode()
        
        // Add nodes to the engine
        audioEngine.attach(audioPlayerNode)
        audioEngine.connect(audioPlayerNode, to: audioEngine.mainMixerNode, format: nil)
        
        // Set up a tap on the engine's output to get audio samples for visualization
        let output = audioEngine.outputNode
        let format = output.inputFormat(forBus: 0)
        
        output.installTap(onBus: 0, bufferSize: AVAudioFrameCount(fftSize), format: format) { [weak self] buffer, time in
            guard let self = self else { return }
            
            // Process audio buffer to get samples for visualization
            self.processAudioBuffer(buffer)
        }
        
        // Request microphone permission for recording
        requestMicrophonePermission()
        
        // Set up a timer to update UI
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.updateUIForAudioPlayback()
        }
    }
    
    private func requestMicrophonePermission() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            // Already authorized
            break
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                if !granted {
                    self?.showMicrophonePermissionAlert()
                }
            }
        case .denied, .restricted:
            showMicrophonePermissionAlert()
        @unknown default:
            break
        }
    }
    
    private func showMicrophonePermissionAlert() {
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.messageText = "Microphone Access Required"
            alert.informativeText = "AudioBloomAI needs access to your microphone for audio visualization. Please enable it in System Preferences."
            alert.alertStyle = .warning
            alert.addButton(withTitle: "Open Settings")
            alert.addButton(withTitle: "Cancel")
            
            if alert.runModal() == .alertFirstButtonReturn {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                    NSWorkspace.shared.open(url)
                }
            }
        }
    }
    
    // MARK: - Audio Processing
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        
        // Get audio samples from the first channel
        let channelCount = Int(buffer.format.channelCount)
        if channelCount > 0 {
            let firstChannelData = channelData[0]
            let frameLength = Int(buffer.frameLength)
            
            // Copy the samples to our audio samples array
            for i in 0..<min(fftSize, frameLength) {
                audioSamples[i] = firstChannelData[i]
            }
            
            // If we have fewer samples than fftSize, pad with zeros
            if frameLength < fftSize {
                for i in frameLength..<fftSize {
                    audioSamples[i] = 0
                }
            }
            
            // Perform FFT on the samples
            performFFT()
        }
    }
    
    private func performFFT() {
        guard let fftSetup = fftSetup else { return }
        
        // Prepare real and imaginary parts for the FFT
        var realIn = [Float](repeating: 0, count: fftSize)
        var imagIn = [Float](repeating: 0, count: fftSize)
        var realOut = [Float](repeating: 0, count: fftSize)
        var imagOut = [Float](repeating: 0, count: fftSize)
        
        // Apply a Hann window to the input samples to reduce spectral leakage
        var window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        vDSP_vmul(audioSamples, 1, window, 1, &realIn, 1, vDSP_Length(fftSize))
        
        // Perform the FFT
        vDSP_DFT_Execute(fftSetup, realIn, imagIn, &realOut, &imagOut)
        
        // Compute magnitudes for visualization
        var magnitudes = [Float](repeating: 0, count: fftSize / 2)
        vDSP_zvmags(&realOut, 1, &imagOut, 1, &magnitudes, 1, vDSP_Length(fftSize / 2))
        
        // Convert to dB scale (log10) with some scaling for better visualization
        var scaledMagnitudes = [Float](repeating: 0, count: fftSize / 2)
        var one: Float = 1
        vDSP_vdbcon(magnitudes, 1, &one, &scaledMagnitudes, 1, vDSP_Length(fftSize / 2), 1)
        
        // Normalize between 0 and 1 for visualization
        var min: Float = 0
        var max: Float = 0
        vDSP_minv(scaledMagnitudes, 1, &min, vDSP_Length(fftSize / 2))
        vDSP_maxv(scaledMagnitudes, 1, &max, vDSP_Length(fftSize / 2))
        
        if max > min

import Cocoa
import MetalKit
import AVFoundation

class ViewController: NSViewController {
    
    // MARK: - Properties
    
    // Metal rendering
    private var metalView: MTKView!
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var pipelineState: MTLRenderPipelineState!
    
    // Audio engine components
    private var audioEngine: AVAudioEngine!
    private var audioPlayerNode: AVAudioPlayerNode!
    private var audioFile: AVAudioFile?
    private var audioNodeTap: AVAudioNodeTap?
    private var fftData: [Float] = []
    private var audioData: [Float] = []
    
    // UI components
    private var progressBar: NSProgressIndicator!
    private var timeLabel: NSTextField!
    private var visualizationTypeSegment: NSSegmentedControl!
    
    // Audio state
    private var isPlaying = false
    private var isAudioFileLoaded = false
    private var isRecordingFromMic = false
    private var currentAudioTime: TimeInterval = 0
    private var visualizationType = 0 // 0: waveform, 1: spectrum, 2: circle
    
    // MARK: - View Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupUI()
        setupMetalRendering()
        setupAudioEngine()
    }
    
    override func viewWillAppear() {
        super.viewWillAppear()
        self.view.window?.delegate = self
    }
    
    override func viewWillDisappear() {
        super.viewWillDisappear()
        stopAudio()
    }
    
    // MARK: - UI Setup
    
    private func setupUI() {
        // Configure the main view
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.black.cgColor
        
        // Create Metal view for visualization
        metalView = MTKView(frame: NSRect(x: 0, y: 50, width: view.bounds.width, height: view.bounds.height - 100))
        metalView.translatesAutoresizingMaskIntoConstraints = false
        metalView.delegate = self
        metalView.framebufferOnly = false
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        view.addSubview(metalView)
        
        // Add progress bar for audio timeline
        progressBar = NSProgressIndicator(frame: NSRect(x: 20, y: 20, width: view.bounds.width - 100, height: 20))
        progressBar.translatesAutoresizingMaskIntoConstraints = false
        progressBar.minValue = 0
        progressBar.maxValue = 100
        progressBar.doubleValue = 0
        progressBar.isIndeterminate = false
        view.addSubview(progressBar)
        
        // Add time label
        timeLabel = NSTextField(frame: NSRect(x: view.bounds.width - 70, y: 20, width: 60, height: 20))
        timeLabel.translatesAutoresizingMaskIntoConstraints = false
        timeLabel.stringValue = "00:00"
        timeLabel.alignment = .right
        timeLabel.isBezeled = false
        timeLabel.drawsBackground = false
        timeLabel.isEditable = false
        timeLabel.textColor = .white
        view.addSubview(timeLabel)
        
        // Add visualization type selector
        visualizationTypeSegment = NSSegmentedControl(frame: NSRect(x: view.bounds.width/2 - 100, y: view.bounds.height - 40, width: 200, height: 30))
        visualizationTypeSegment.translatesAutoresizingMaskIntoConstraints = false
        visualizationTypeSegment.segmentCount = 3
        visualizationTypeSegment.setLabel("Waveform", forSegment: 0)
        visualizationTypeSegment.setLabel("Spectrum", forSegment: 1)
        visualizationTypeSegment.setLabel("Circle", forSegment: 2)
        visualizationTypeSegment.selectedSegment = 0
        visualizationTypeSegment.target = self
        visualizationTypeSegment.action = #selector(visualizationTypeChanged(_:))
        view.addSubview(visualizationTypeSegment)
        
        // Set up constraints
        NSLayoutConstraint.activate([
            // Metal view constraints
            metalView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            metalView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            metalView.topAnchor.constraint(equalTo: view.topAnchor, constant: 50),
            metalView.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -50),
            
            // Progress bar constraints
            progressBar.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            progressBar.trailingAnchor.constraint(equalTo: timeLabel.leadingAnchor, constant: -10),
            progressBar.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -20),
            progressBar.heightAnchor.constraint(equalToConstant: 20),
            
            // Time label constraints
            timeLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            timeLabel.centerYAnchor.constraint(equalTo: progressBar.centerYAnchor),
            timeLabel.widthAnchor.constraint(equalToConstant: 60),
            
            // Visualization type segment constraints
            visualizationTypeSegment.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            visualizationTypeSegment.topAnchor.constraint(equalTo: view.topAnchor, constant: 10),
            visualizationTypeSegment.widthAnchor.constraint(equalToConstant: 200),
            visualizationTypeSegment.heightAnchor.constraint(equalToConstant: 30)
        ])
    }
    
    // MARK: - Metal Setup
    
    private func setupMetalRendering() {
        // Get the default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        
        // Configure the Metal view
        metalView.device = device
        metalView.framebufferOnly = false
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.preferredFramesPerSecond = 60
        
        // Create the command queue
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Could not create Metal command queue")
        }
        self.commandQueue = commandQueue
        
        // Create the render pipeline
        do {
            let library = device.makeDefaultLibrary()
            let vertexFunction = library?.makeFunction(name: "vertexShader")
            let fragmentFunction = library?.makeFunction(name: "fragmentShader")
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
            
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("Failed to create render pipeline state: \(error)")
        }
    }
    
    // MARK: - Audio Setup
    
    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        audioPlayerNode = AVAudioPlayerNode()
        
        // Add nodes to the engine
        audioEngine.attach(audioPlayerNode)
        
        // Request microphone permission
        requestMicrophonePermission()
        
        // Set up a timer to update UI
        Timer.scheduledTimer(timeInterval: 0.1, target: self, selector: #selector(updateAudioDisplay), userInfo: nil, repeats: true)
    }
    
    private func requestMicrophonePermission() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            setupAudioInputTap()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                if granted {
                    DispatchQueue.main.async {
                        self.setupAudioInputTap()
                    }
                }
            }
        case .denied, .restricted:
            let alert = NSAlert()
            alert.messageText = "Microphone Access Required"
            alert.informativeText = "AudioBloomAI needs access to your microphone for audio visualization. Please enable it in System Preferences."
            alert.alertStyle = .warning
            alert.addButton(withTitle: "Open Settings")
            alert.addButton(withTitle: "Cancel")
            
            if alert.runModal() == .alertFirstButtonReturn {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                    NSWorkspace.shared.open(url)
                }
            }
        @unknown default:
            break
        }
    }
    
    private func setupAudioInputTap() {
        // Get the input node (microphone)
        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        
        // Install a tap on the input node to get audio data
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, time in
            guard let self = self, self.isRecordingFromMic else { return }
            
            // Process the audio buffer to get visualization data
            let channelData = buffer.floatChannelData?[0]
            let frameLength = Int(buffer.frameLength)
            
            // Store audio data for visualization
            self.audioData = Array(UnsafeBufferPointer(start: channelData, count: frameLength))
            
            // Perform FFT for spectrum visualization
            self.performFFT(on: self.audioData)
        }
        
        // Start the audio engine if not already running
        do {
            try audioEngine.start()
        } catch {
            print("Error starting audio engine: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Audio Processing
    
    private func performFFT(on audioData: [Float]) {
        // This is a placeholder for FFT processing
        // In a real app, you would use vDSP or Accelerate framework to perform FFT
        // For now, we'll just create some dummy data
        let fftSize = 512
        var result = [Float](repeating: 0, count: fftSize)
        
        // Generate mock FFT data based on the audio samples
        for i in 0..<min(fftSize, audioData.count) {
            // Create a simple amplitude-based visualization
            result[i] = abs(audioData[i]) * 5.0
        }
        
        self.fftData = result
    }
    
    // MARK: - Audio Controls
    
    func playAudioFile(_ url: URL) {
        stopAudio()
        
        do {
            // Create an audio file
            let audioFile = try AVAudioFile(forReading: url)
            self.audioFile = audioFile
            
            // Set up the audio format
            let audioFormat = audioFile.processingFormat
            
            // Connect the player to the mixer
            audioEngine.connect(audioPlayerNode, to: audioEngine.mainMixerNode, format: audioFormat)
            
            // Schedule the file for playback
            audioPlayerNode.scheduleFile(audioFile, at: nil) { [weak self] in
                self?.isPlaying = false
            }
            
            // Start the audio engine if not already running
            if !audioEngine.isRunning {
                try audioEngine.start()
            }
            
            // Play the audio
            audioPlayerNode.play()
            isPlaying = true
            isAudioFileLoaded = true
            isRecordingFromMic = false
            
        } catch {
            print("Error loading audio file: \(error.localizedDescription)")
            let alert = NSAlert()
            alert.messageText = "Error Loading Audio"
            alert.informativeText = "Could not load the selected audio file: \(error.localizedDescription)"
            alert.alertStyle = .warning
            alert.runModal()
        }
    }
    
    func startMicrophoneInput() {
        stopAudio()
        
        isRecordingFromMic = true
        isPlaying = true
        isAudioFileLoaded = false
        
        // Make sure audio engine is running
        if !audioEngine.isRunning {
            do {
                try audioEngine.start()
            } catch {
                print("Error starting audio engine: \(error.localizedDescription)")
            }
        }
    }
    
    func togglePlayPause() {
        if isAudioFileLoaded {
            if isPlaying {
                audioPlayerNode.pause()
            } else {
                audioPlayerNode.play()
            }
            isPlaying = !isPlaying
        } else if isRecordingFromMic {
            if isPlaying {
                audioEngine.pause()
            } else {
                try? audioEngine.start()
            }
            isPlaying = !isPlaying
        }
    }
    
    func stopAudio() {
        // Stop any current audio
        audioPlayerNode.stop()
        
        if isRecordingFromMic {
            // Remove tap from input node
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        
        isPlaying = false
        currentAudioTime = 0
        progressBar.doubleValue = 0
        timeLabel.stringValue = "00:00"
    }
    
    // MARK: - UI Updates
    
    @objc private func updateAudioDisplay() {
        if isPlaying && isAudioFileLoaded {
            // Update time display for audio file playback
            if let nodeTime = audioPlayerNode.lastRenderTime,
               let playerTime = audioPlayerNode.playerTime(forNodeTime: nodeTime) {
                
                // Calculate current time in seconds
                currentAudioTime = Double(playerTime.sampleTime) / Double(playerTime.sampleRate)
                
                // Update progress bar if we have a valid audio file
                if let audioFile = audioFile {
                    let progress = (currentAudioTime / Double(audioFile.length) * 100)
                    progressBar.doubleValue = min(100, progress)
                }
                
                // Format time display
                let minutes = Int(currentAudioTime) / 60
                let seconds = Int(currentAudioTime) % 60
                timeLabel.stringValue = String(format: "%02d:%02d", minutes, seconds)
            }
        } else if isRecordingFromMic && isPlaying {
            // Update time display for microphone input
            currentAudioTime += 0.1
            let minutes = Int(current

