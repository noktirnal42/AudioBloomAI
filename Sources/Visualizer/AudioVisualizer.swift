    private func cleanup() {
        // Cancel timers
        fpsUpdateTimer?.invalidate()
        fpsUpdateTimer = nil
        
        // Cancel subscriptions
        cancellables.forEach { $0.cancel() }
        cancellables.removeAll()
        
        // Pause visualization
        isActive = false
        
        // Clear cached data
        lastAudioData = []
        lastLevels = (0, 0)
    }
}

// MARK: - VisualizationParameterReceiver Extension

extension AudioVisualizer: VisualizationParameterReceiver {
    /// Updates parameters for the visualization
    public func updateParameters(_ parameters: [String: Any]) {
        // Update sensitivity if provided
        if let sensitivity = parameters["sensitivity"] as? Float {
            self.sensitivity = sensitivity
        }
        
        // Update motion intensity if provided
        if let motionIntensity = parameters["motionIntensity"] as? Float {
            self.motionIntensity = motionIntensity
        }
        
        // Update theme if provided
        if let themeName = parameters["theme"] as? String,
           let theme = VisualTheme(rawValue: themeName) {
            self.currentTheme = theme
        }
        
        // Pass the parameters to the renderer
        updateVisualizationParameters()
    }
}

// MARK: - SwiftUI Previews

#if DEBUG
struct AudioVisualizerPreviews: PreviewProvider {
    static var previews: some View {
        VStack {
            Text("Audio Visualizer")
                .font(.headline)
            
            let visualizer = AudioVisualizer()
            visualizer.createMetalView()
                .frame(width: 400, height: 300)
                .onAppear {
                    visualizer.initialize()
                    visualizer.start()
                    
                    // Simulate some audio data
                    let mockData = AudioData(
                        frequencyData: Array(repeating: 0, count: 1024).enumerated().map { i, _ in
                            Float(sin(Double(i) / 100.0) * 0.5 + 0.5)
                        },
                        timeData: [],
                        leftLevel: 0.7,
                        rightLevel: 0.8,
                        timestamp: Date()
                    )
                    
                    visualizer.processAudioData(mockData)
                }
        }
        .padding()
        .background(Color.black)
    }
}
#endif

import Foundation
import Metal
import MetalKit
import SwiftUI
import Combine
import AudioBloomCore

#if canImport(AudioProcessor)
import AudioProcessor
#endif

/// Main public class for the audio visualizer system
public class AudioVisualizer {
    // The shared visualizer instance
    private static var shared: AudioVisualizer?
    
    // The underlying renderer for Metal-based visualization
    private var renderer: MetalRenderer?
    
    // Subscriptions for audio data
    private var audioSubscriptions = Set<AnyCancellable>()
    
    /// Initialize the audio visualizer system
    public init() {
        do {
            renderer = try MetalRenderer()
        } catch {
            print("Failed to initialize Metal renderer: \(error.localizedDescription)")
        }
        
        // Configure default theme and visualization mode
        setTheme(.classic)
        setVisualizationMode(.spectrum)
    }
    
    /// Get or create the shared visualizer instance
    public static func shared() -> AudioVisualizer {
        if shared == nil {
            shared = AudioVisualizer()
        }
        return shared!
    }
    
    /// Create a SwiftUI view for the visualizer
    /// - Returns: A SwiftUI view for the audio visualizer
    public func createView() -> some View {
        AudioVisualizerView(visualizer: self)
    }
    
    // MARK: - Public Configuration Methods
    
    /// Set the visualization mode
    /// - Parameter mode: The visualization mode
    public func setVisualizationMode(_ mode: VisualizationMode) {
        renderer?.setVisualizationMode(mode)
    }
    
    /// Set the visualization theme
    /// - Parameter theme: The visualization theme
    public func setTheme(_ theme: Theme) {
        renderer?.setTheme(theme)
    }
    
    /// Set custom colors for visualization
    /// - Parameters:
    ///   - primary: Primary color
    ///   - secondary: Secondary color
    ///   - background: Background color
    ///   - accent: Accent color
    public func setCustomColors(primary: SIMD4<Float>, secondary: SIMD4<Float>,
                              background: SIMD4<Float>, accent: SIMD4<Float>) {
        renderer?.setCustomColors(primary: primary, secondary: secondary,
                                background: background, accent: accent)
    }
    
    /// Set visualization parameters
    /// - Parameters:
    ///   - sensitivity: Audio sensitivity (0.0-1.0)
    ///   - motionIntensity: Motion intensity (0.0-1.0)
    ///   - colorIntensity: Color intensity (0.0-1.0)
    public func setVisualizationParameters(sensitivity: Float = 0.8,
                                         motionIntensity: Float = 0.7,
                                         colorIntensity: Float = 0.8) {
        renderer?.setVisualizationParameters(sensitivity: sensitivity,
                                           motionIntensity: motionIntensity,
                                           colorIntensity: colorIntensity)
    }
    
    // MARK: - Audio Processing Integration
    
    #if canImport(AudioProcessor)
    /// Connect to the audio processor for real-time updates
    /// - Parameter audioProcessor: The audio processor instance
    public func connectToAudioProcessor(_ audioProcessor: AudioEngine) {
        // Subscribe to frequency data updates
        audioProcessor.frequencyDataPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] frequencyData in
                self?.renderer?.updateAudioData(frequencyData)
            }
            .store(in: &audioSubscriptions)
        
        // Subscribe to audio level updates
        audioProcessor.audioLevelsPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] levels in
                self?.renderer?.setAudioLevels(
                    bass: levels.bass,
                    mid: levels.mid,
                    treble: levels.treble,
                    left: levels.left,
                    right: levels.right
                )
            }
            .store(in: &audioSubscriptions)
        
        // Subscribe to beat detection
        audioProcessor.beatDetectionPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] beatDetected in
                if let energy = audioProcessor.currentEnergy,
                   let pleasantness = audioProcessor.currentPleasantness,
                   let complexity = audioProcessor.currentComplexity {
                    self?.renderer?.setNeuralParameters(
                        energy: energy,
                        pleasantness: pleasantness,
                        complexity: complexity,
                        beatDetected: beatDetected ? 1.0 : 0.0
                    )
                }
            }
            .store(in: &audioSubscriptions)
    }
    #endif
    
    // MARK: - Internal Methods for View Integration
    
    /// Internal method to render to a Metal drawable
    /// - Parameters:
    ///   - drawable: The Metal drawable to render to
    ///   - renderPassDescriptor: The render pass descriptor
    internal func render(to drawable: CAMetalDrawable, with renderPassDescriptor: MTLRenderPassDescriptor) {
        renderer?.render(to: drawable, with: renderPassDescriptor)
    }
    
    /// Internal method to update the viewport size
    /// - Parameter size: The new viewport size
    internal func updateViewportSize(_ size: CGSize) {
        renderer?.setViewportSize(size)
    }
    
    /// Manually update audio data (for testing or custom input)
    /// - Parameter data: Array of audio frequency data (0.0-1.0)
    public func updateAudioData(_ data: [Float]) {
        renderer?.updateAudioData(data)
    }
    
    /// Manually set audio levels (for testing or custom input)
    /// - Parameters:
    ///   - bass: Bass level (0.0-1.0)
    ///   - mid: Mid level (0.0-1.0)
    ///   - treble: Treble level (0.0-1.0)
    ///   - left: Left channel level (0.0-1.0)
    ///   - right: Right channel level (0.0-1.0)
    public func setAudioLevels(bass: Float, mid: Float, treble: Float, left: Float, right: Float) {
        renderer?.setAudioLevels(bass: bass, mid: mid, treble: treble, left: left, right: right)
    }
}
    /// Audio engine for capturing and processing audio
    private let audioEngine: AudioDataProvider
    
    /// The visualization renderer
    private let visualizer: VisualizationRenderer & VisualizationParameterReceiver
    
    /// Optional neural engine for enhanced audio analysis
    private var neuralEngine: MLProcessing?
    
    /// Audio data subscription
    private var audioDataSubscription: AnyCancellable?
    
    /// Neural data subscription
    private var neuralDataSubscription: AnyCancellable?
    
    /// Application settings
    private let settings: AudioBloomSettings
    
    /// Whether audio visualization is active
    @Published public private(set) var isActive: Bool = false
    
    /// Current visualization theme
    @Published public var currentTheme: AudioBloomCore.VisualTheme {
        didSet {
            updateVisualizationParameters()
        }
    }
    
    /// Audio sensitivity (0.0 - 1.0)
    @Published public var audioSensitivity: Double {
        didSet {
            updateVisualizationParameters()
        }
    }
    
    /// Motion intensity (0.0 - 1.0)
    @Published public var motionIntensity: Double {
        didSet {
            updateVisualizationParameters()
        }
    }
    
    /// Neural engine enabled flag
    @Published public var neuralEngineEnabled: Bool {
        didSet {
            updateVisualizationParameters()
        }
    }
    
    /// Beat detection events
    @Published public private(set) var beatDetected: Bool = false
    
    /// Frames per second (for performance monitoring)
    @Published public private(set) var framesPerSecond: Double = 0
    
    /// Frame calculation helper properties
    private var frameCount: Int = 0
    private var lastFrameTime: TimeInterval = 0
    private var frameTimer: Timer?
    
    /// Initializes a new AudioVisualizer
    /// - Parameters:
    ///   - audioEngine: The audio data provider
    ///   - visualizer: The visualization renderer
    ///   - settings: Application settings
    public init(audioEngine: AudioDataProvider, 
                visualizer: VisualizationRenderer & VisualizationParameterReceiver,
                neuralEngine: MLProcessing? = nil,
                settings: AudioBloomSettings) {
        self.audioEngine = audioEngine
        self.visualizer = visualizer
        self.neuralEngine = neuralEngine
        self.settings = settings
        
        // Initialize visualization parameters
        self.currentTheme = settings.currentTheme
        self.audioSensitivity = settings.audioSensitivity
        self.motionIntensity = settings.motionIntensity
        self.neuralEngineEnabled = settings.neuralEngineEnabled
        
        // Set up audio data subscription
        setupAudioSubscription()
        
        // Set up neural engine if available
        if let neuralEngine = neuralEngine, neuralEngineEnabled {
            setupNeuralSubscription(neuralEngine: neuralEngine)
        }
    }
    
    /// Starts audio visualization
    public func start() async throws {
        guard !isActive else { return }
        
        // Prepare visualization renderer if not already ready
        if !visualizer.isReady {
            visualizer.prepareRenderer()
        }
        
        // Set up neural engine if enabled
        if neuralEngineEnabled, let neuralEngine = neuralEngine {
            neuralEngine.prepareMLModel()
        }
        
        // Set up audio session
        try await audioEngine.setupAudioSession()
        
        // Start audio capture
        try audioEngine.startCapture()
        
        // Set active status
        isActive = true
        
        // Start frame rate timer
        startFrameRateTimer()
    }
    
    /// Stops audio visualization
    public func stop() {
        guard isActive else { return }
        
        // Stop audio capture
        audioEngine.stopCapture()
        
        // Set inactive status
        isActive = false
        
        // Stop frame rate timer
        stopFrameRateTimer()
    }
    
    /// Sets up audio data subscription
    private func setupAudioSubscription() {
        // Subscribe to audio data changes from the audio engine
        audioDataSubscription = (audioEngine as? AudioEngine)?
            .getAudioDataPublisher()
            .publisher
            .sink { [weak self] (frequencyData, levels) in
                guard let self = self else { return }
                
                // Process audio data for visualization
                self.processAudioData(frequencyData, levels: levels)
                
                // If neural engine is enabled, pass data to it
                if self.neuralEngineEnabled, let neuralEngine = self.neuralEngine {
                    Task {
                        await neuralEngine.processAudioData(frequencyData)
                    }
                }
                
                // Update the visualization
                self.updateVisualization(frequencyData, levels: levels)
            }
    }
    
    /// Sets up neural engine subscription
    private func setupNeuralSubscription(neuralEngine: MLProcessing) {
        // Subscribe to neural engine outputs
        neuralDataSubscription = neuralEngine.objectWillChange
            .sink { [weak self] _ in
                guard let self = self, self.neuralEngineEnabled else { return }
                
                // Extract additional neural features that affect visualization
                if let neuralEngine = self.neuralEngine as? NeuralEngine {
                    // Update beat detection status
                    DispatchQueue.main.async {
                        self.beatDetected = neuralEngine.beatDetected
                    }
                    
                    // Use neural engine output to enhance visualization parameters
                    self.enhanceVisualizationWithNeuralData(neuralEngine.outputData)
                }
            }
    }
    
    /// Processes audio data for visualization
    private func processAudioData(_ frequencyData: [Float], levels: (left: Float, right: Float)) {
        // Preprocessing of audio data could go here if needed
        // For example, noise reduction, equalization, etc.
    }
    
    /// Updates the visualization with new audio data
    private func updateVisualization(_ frequencyData: [Float], levels: (left: Float, right: Float)) {
        // Update the visualization with the latest audio data
        visualizer.update(audioData: frequencyData, levels: levels)
        
        // Trigger a render
        visualizer.render()
        
        // Update frame counter
        updateFrameCount()
    }
    
    /// Enhances visualization with neural data
    private func enhanceVisualizationWithNeuralData(_ neuralData: [Float]) {
        // No need to do anything if neural data is empty
        guard !neuralData.isEmpty else { return }
        
        // Create enhanced parameters based on neural data
        var enhancedParams: [String: Any] = [:]
        
        // Example: Use neural data to affect visualization intensity
        if neuralData.count > 16 {
            // Extract key neural features
            let energyLevel = neuralData[8]  // Energy feature from neural output
            let pleasantness = neuralData[9] // Pleasantness feature
            let complexity = neuralData[18]  // Complexity feature
            
            // Enhance visualization sensitivity based on neural features
            let enhancedSensitivity = Float(audioSensitivity) * (1.0 + energyLevel * 0.5)
            enhancedParams["sensitivity"] = enhancedSensitivity
            
            // Enhance motion intensity based on complexity
            let enhancedMotion = Float(motionIntensity) * (1.0 + complexity * 0.3)
            enhancedParams["motionIntensity"] = enhancedMotion
            
            // Add neural-specific parameters
            enhancedParams["neuralEnergy"] = energyLevel
            enhancedParams["neuralPleasantness"] = pleasantness
            enhancedParams["neuralComplexity"] = complexity
            
            // Update visualization with enhanced parameters
            visualizer.updateParameters(enhancedParams)
        }
    }
    
    /// Updates visualization parameters based on current settings
    private func updateVisualizationParameters() {
        // Get color parameters from the theme
        var parameters = currentTheme.colorParameters
        
        // Add additional parameters
        parameters["sensitivity"] = Float(audioSensitivity)
        parameters["motionIntensity"] = Float(motionIntensity)
        parameters["neuralEngineEnabled"] = neuralEngineEnabled
        parameters["theme"] = currentTheme.rawValue
        
        // Update the visualizer with the parameters
        visualizer.updateParameters(parameters)
        
        // Update application settings
        settings.currentTheme = currentTheme
        settings.audioSensitivity = audioSensitivity
        settings.motionIntensity = motionIntensity
        settings.neuralEngineEnabled = neuralEngineEnabled
    }
    
    /// Updates frame count for FPS calculation
    private func updateFrameCount() {
        frameCount += 1
        
        let currentTime = CACurrentMediaTime()
        let elapsed = currentTime - lastFrameTime
        
        // Calculate FPS approximately every second
        if elapsed > 1.0 {
            DispatchQueue.main.async {
                self.framesPerSecond = Double(self.frameCount) / elapsed
                self.frameCount = 0
                self.lastFrameTime = currentTime
            }
        }
    }
    
    /// Starts the frame rate timer
    private func startFrameRateTimer() {
        frameCount = 0
        lastFrameTime = CACurrentMediaTime()
        
        frameTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            // This timer ensures FPS is updated even if few frames are being rendered
            self?.updateFrameCount()
        }
    }
    
    /// Stops the frame rate timer
    private func stopFrameRateTimer() {
        frameTimer?.invalidate()
        frameTimer = nil
    }
    
    /// Changes the active visualization theme
    /// - Parameter theme: The new theme to use
    public func setTheme(_ theme: AudioBloomCore.VisualTheme) {
        currentTheme = theme
    }
    
    /// Cycles to the next available visualization theme
    public func cycleToNextTheme() {
        let allThemes = AudioBloomCore.VisualTheme.allCases
        if let currentIndex = allThemes.firstIndex(where: { $0.id == currentTheme.id }),
           let nextTheme = allThemes[safe: currentIndex + 1] {
            currentTheme = nextTheme
        } else {
            // Wrap around to first theme
            currentTheme = allThemes.first ?? .classic
        }
    }
}

/// Extension to provide safe array access
extension Array {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}

/// SwiftUI View for audio visualization
public struct AudioVisualizerView: View {
    /// The audio visualizer controller
    @ObservedObject public var visualizer: AudioVisualizer
    
    /// The metal renderer
    private let renderer: MetalRenderer
    
    /// Initializes a new AudioVisualizerView
    /// - Parameter visualizer: The audio visualizer controller
    public init(visualizer: AudioVisualizer, renderer: MetalRenderer) {
        self.visualizer = visualizer
        self.renderer = renderer
    }
    
    public var body: some View {
        ZStack {
            // Main visualization view
            MetalView(renderer: renderer)
                .ignoresSafeArea()
            
            // Optional overlays for UI controls, beat indicators, etc.
            VStack {
                Spacer()
                
                // Performance monitor
                HStack {
                    Text("FPS: \(Int(visualizer.framesPerSecond))")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                        .padding(8)
                        .background(Color.black.opacity(0.5))
                        .cornerRadius(8)
                    
                    Spacer()
                    
                    // Beat indicator
                    if visualizer.beatDetected {
                        Circle()
                            .fill(Color.white)
                            .frame(width: 12, height: 12)
                            .padding(8)
                    }
                }
                .padding()
            }
        }
        .onTapGesture {
            // Cycle to next theme on tap
            visualizer.cycleToNextTheme()
        }
    }
}

