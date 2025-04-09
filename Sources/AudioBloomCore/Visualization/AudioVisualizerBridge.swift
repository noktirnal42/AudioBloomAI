import Foundation
import Combine
import MetalKit
import AVFoundation
import SwiftUI
import Logging
import CoreVideo

/// Enumeration of available visualization modes
public enum VisualizationMode: String, CaseIterable, Identifiable, Sendable {
    case spectrum = "Spectrum"
    case waveform = "Waveform"
    case particles = "Particles"
    case neural = "Neural"
    
    public var id: String { rawValue }
    
    /// Get a user-friendly description of the mode
    public var description: String {
        switch self {
        case .spectrum:
            return "Frequency Spectrum"
        case .waveform:
            return "Waveform Oscilloscope"
        case .particles:
            return "Particle System"
        case .neural:
            return "Neural Pattern"
        }
    }
}

/// A bridge that connects audio processing to visualization systems
public final class AudioVisualizerBridge: ObservableObject, @unchecked Sendable {
    
    // MARK: - Published Properties
    
    /// Current visualization mode
    @Published public private(set) var currentMode: VisualizationMode = .spectrum
    
    /// Previous visualization mode (for transitions)
    @Published public private(set) var previousMode: VisualizationMode = .spectrum
    
    /// Transition progress (0-1)
    @Published public private(set) var transitionProgress: Double = 0.0
    
    /// Whether visualization is active
    @Published public private(set) var isActive: Bool = false
    
    /// Spectrum data for visualization (normalized 0-1)
    @Published public private(set) var spectrumData: [Float] = []
    
    /// Waveform data for visualization (normalized -1 to 1)
    @Published public private(set) var waveformData: [Float] = []
    
    /// Audio levels (left and right channels, 0-1)
    @Published public private(set) var audioLevels: (left: Float, right: Float) = (0, 0)
    
    /// Beat detected flag (resets automatically)
    @Published public private(set) var beatDetected: Bool = false
    
    // MARK: - Configuration
    
    /// Configuration for the visualizer bridge
    public struct Configuration: Equatable, Sendable {
        /// FFT smoothing factor (0-1)
        public var fftSmoothingFactor: Float = 0.7
        
        /// Level smoothing factor (0-1)
        public var levelSmoothingFactor: Float = 0.5
        
        /// Transition duration in seconds
        public var transitionDuration: Double = 0.5
        
        /// Maximum FPS for visualization rendering
        public var maxFps: Double = 60
        
        /// Whether to use GPU-accelerated audio processing
        public var useGPUProcessing: Bool = true
        
        /// Custom theme colors
        public var primaryColor: Color = .blue
        public var secondaryColor: Color = .purple
        public var backgroundColor: Color = .black
        public var accentColor: Color = .white
        
        /// Visualization-specific settings
        public var spectrumSettings: SpectrumSettings = SpectrumSettings()
        public var waveformSettings: WaveformSettings = WaveformSettings()
        public var particleSettings: ParticleSettings = ParticleSettings()
        public var neuralSettings: NeuralSettings = NeuralSettings()
        
        /// Settings for spectrum visualization
        public struct SpectrumSettings: Equatable, Sendable {
            /// Bar spacing (0-1)
            public var barSpacing: Float = 0.2
            
            /// Bar count (16-256)
            public var barCount: Int = 64
            
            /// Use logarithmic scale for frequency distribution
            public var useLogScale: Bool = true
            
            /// Show peak indicators
            public var showPeaks: Bool = true
            
            /// Frequency range (Hz)
            public var minFrequency: Float = 20
            public var maxFrequency: Float = 20000
            
            /// Default settings
            public init() {}
        }
        
        /// Settings for waveform visualization
        public struct WaveformSettings: Equatable, Sendable {
            /// Line thickness (1-10)
            public var lineThickness: Float = 2.0
            
            /// Line smoothing (0-1)
            public var smoothing: Float = 0.3
            
            /// Show multiple lines
            public var showMultipleLines: Bool = true
            
            /// Time scale (seconds of audio visible)
            public var timeScale: Float = 0.05
            
            /// Default settings
            public init() {}
        }
        
        /// Settings for particle visualization
        public struct ParticleSettings: Equatable, Sendable {
            /// Particle count (100-10000)
            public var particleCount: Int = 2000
            
            /// Particle size (0.5-5.0)
            public var particleSize: Float = 1.5
            
            /// Audio reactivity (0-1)
            public var audioReactivity: Float = 0.8
            
            /// Motion speed (0-2)
            public var motionSpeed: Float = 1.0
            
            /// Default settings
            public init() {}
        }
        
        /// Settings for neural visualization
        public struct NeuralSettings: Equatable, Sendable {
            /// Complexity (0-1)
            public var complexity: Float = 0.7
            
            /// Speed (0-2)
            public var speed: Float = 1.0
            
            /// Color intensity (0-1)
            public var colorIntensity: Float = 0.8
            
            /// Beat reactivity (0-1)
            public var beatReactivity: Float = 0.9
            
            /// Default settings
            public init() {}
        }
        
        /// Default configuration
        public init() {}
    }
    
    /// Neural processing response for visualization
    public struct NeuralProcessingResponse: Equatable, Sendable {
        /// Energy level (0-1)
        public var energy: Float = 0.0
        
        /// Pleasantness (0-1)
        public var pleasantness: Float = 0.5
        
        /// Complexity (0-1)
        public var complexity: Float = 0.5
        
        /// Pattern type (0-3)
        public var patternType: Int = 0
        
        /// Beat confidence (0-1)
        public var beatConfidence: Float = 0.0
        
        /// Default initialization
        public init() {}
        
        /// Full initialization
        public init(energy: Float, pleasantness: Float, complexity: Float, patternType: Int = 0, beatConfidence: Float = 0.0) {
            self.energy = energy
            self.pleasantness = pleasantness
            self.complexity = complexity
            self.patternType = patternType
            self.beatConfidence = beatConfidence
        }
    }
    
    // MARK: - Private Properties
    
    /// Current configuration
    private var configuration: Configuration
    
    /// Queue for audio processing
    private let processingQueue = DispatchQueue(label: "com.audiobloom.visualizer", qos: .userInteractive)
    
    /// Lock for thread safety
    private let lock = NSLock()
    
    /// Subscribers to audio data
    private var audioSubscription: AnyCancellable?
    
    /// Subscribers to neural processing
    private var neuralSubscription: AnyCancellable?
    
    /// Timer for transitions
    private var transitionTimer: Timer?
    
    /// Time when transition started
    private var transitionStartTime: CFTimeInterval = 0
    
    /// Raw audio data buffer
    private var audioDataBuffer: [Float] = []
    
    /// Logger
    private let logger = Logger(label: "com.audiobloom.visualizerbridge")
    
    /// Spectrum data history (for smoothing)
    private var spectrumHistory: [[Float]] = []
    
    /// Waveform data history
    private var waveformHistory: [[Float]] = []
    
    /// Display link for frame timing
    private var displayLink: CVDisplayLink?
    
    /// Performance monitoring service
    private weak var performanceMonitor: PerformanceMonitor?
    
    // MARK: - Initialization
    
    /// Initialize with a custom configuration
    /// - Parameter configuration: The configuration to use
    public init(configuration: Configuration = Configuration(), performanceMonitor: PerformanceMonitor? = nil) {
        self.configuration = configuration
        self.performanceMonitor = performanceMonitor
        
        // Initialize with empty data
        spectrumData = [Float](repeating: 0, count: 128)
        waveformData = [Float](repeating: 0, count: 512)
        
        // Setup display link
        setupDisplayLink()
        
        logger.info("AudioVisualizerBridge initialized")
    }
    
    deinit {
        stopDisplayLink()
        audioSubscription?.cancel()
        neuralSubscription?.cancel()
        transitionTimer?.invalidate()
        logger.debug("AudioVisualizerBridge deinitialized")
    }
    
    // MARK: - Public Methods
    
    /// Update the configuration
    /// - Parameter configuration: New configuration
    public func updateConfiguration(_ configuration: Configuration) {
        lock.lock()
        self.configuration = configuration
        lock.unlock()
        
        logger.debug("Configuration updated")
    }
    
    /// Switch to a different visualization mode with transition
    /// - Parameter mode: The mode to switch to
    public func switchToMode(_ mode: VisualizationMode) {
        guard mode != currentMode else { return }
        
        lock.lock()
        previousMode = currentMode
        currentMode = mode
        transitionStartTime = CFAbsoluteTimeGetCurrent()
        transitionProgress = 0.0
        lock.unlock()
        
        // Cancel any existing transition timer
        transitionTimer?.invalidate()
        
        // Start transition timer
        transitionTimer = Timer.scheduledTimer(withTimeInterval: 1/60, repeats: true) { [weak self] timer in
            self?.updateTransitionProgress()
        }
        
        logger.debug("Switching visualization mode: \(previousMode.rawValue) -> \(mode.rawValue)")
    }
    
    /// Activate visualization
    public func activate() {
        isActive = true
        startDisplayLink()
        logger.debug("Visualization activated")
    }
    
    /// Deactivate visualization
    public func deactivate() {
        isActive = false
        stopDisplayLink()
        logger.debug("Visualization deactivated")
    }
    
    /// Process audio data for visualization
    /// - Parameter audioData: Raw audio data
    public func processAudioData(_ audioData: [Float], levels: (left: Float, right: Float)) {
        guard isActive else { return }
        
        // Start performance measurement
        performanceMonitor?.beginMeasuring("AudioVisualization")
        
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.lock.lock()
            
            // Store audio data
            self.audioDataBuffer = audioData
            
            // Apply smoothing to levels
            let smoothingFactor = self.configuration.levelSmoothingFactor
            let smoothedLeft = levels.left * smoothingFactor + (self.audioLevels.left * (1 - smoothingFactor))
            let smoothedRight = levels.right * smoothingFactor + (self.audioLevels.right * (1 - smoothingFactor))
            
            // Update levels
            self.audioLevels = (smoothedLeft, smoothedRight)
            
            // Process for each visualization mode
            self.processSpectrumData(audioData)
            self.processWaveformData(audioData)
            
            self.lock.unlock()
            
            // End performance measurement
            self.performanceMonitor?.endMeasuring("AudioVisualization")
        }
    }
    
    /// Process neural analysis results
    /// - Parameter response: Neural processing response
    public func processNeuralResponse(_ response: NeuralProcessingResponse) {
        guard isActive else { return }
        
        let wasDetected = beatDetected
        
        // Update beat detection based on neural confidence
        if response.beatConfidence > 0.7 && !wasDetected {
            DispatchQueue.main.async {
                self.beatDetected = true
                
                // Auto-reset the beat detection after a short delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    self.beatDetected = false
                }
            }
        }
        
        // Can also update other visualization parameters based on neural response
        // Implementation would depend on how different visualizations use the neural data
    }
    
    /// Subscribe to audio data from a publisher
    /// - Parameter publisher: The publisher to subscribe to
    public func subscribeToAudioData<P: Publisher>(_ publisher: P) where P.Output == ([Float], (Float, Float)), P.Failure == Never {
        audioSubscription = publisher
            .receive(on: processingQueue)
            .sink { [weak self] audioData, levels in
                self?.processAudioData(audioData, levels: levels)
            }
        
        logger.debug("Subscribed to audio data")
    }
    
    /// Subscribe to neural processing results
    /// - Parameter publisher: The publisher of neural responses
    public func subscribeToNeuralProcessing<P: Publisher>(_ publisher: P) where P.Output == NeuralProcessingResponse, P.Failure == Never {
        neuralSubscription = publisher
            .receive(on: processingQueue)
            .sink { [weak self] response in
                self?.processNeuralResponse(response)
            }
        
        logger.debug("Subscribed to neural processing")
    }
    
    // MARK: - Private Methods
    
    /// Setup the display link for frame synchronization
    private func setupDisplayLink() {
        // Create a display link capable of being used with all active displays
        var newDisplayLink: CVDisplayLink?
        
        // Set up display link callback
        let displayLinkOutputCallback: CVDisplayLinkOutputCallback = { 
            (displayLink: CVDisplayLink, 
             inNow: UnsafePointer<CVTimeStamp>, 
             inOutputTime: UnsafePointer<CVTimeStamp>, 
             flagsIn: CVOptionFlags, 
             flagsOut: UnsafeMutablePointer<CVOptionFlags>, 
             displayLinkContext: UnsafeMutableRawPointer?) -> CVReturn in
            
            // Get the object reference from context
            let bridge = Unmanaged<AudioVisualizerBridge>.fromOpaque(displayLinkContext!).takeUnretainedValue()
            bridge.displayLinkDidFire()
            
            return kCVReturnSuccess
        }
        
        // Create display link
        let error = CVDisplayLinkCreateWithActiveCGDisplays(&newDisplayLink)
        
        if error == kCVReturnSuccess, let newDisplayLink = newDisplayLink {
            // Set the context to point to self
            let pointerToSelf = Unmanaged.passUnretained(self).toOpaque()
            CVDisplayLinkSetOutputCallback(newDisplayLink, displayLinkOutputCallback, pointerToSelf)
            
            self.displayLink = newDisplayLink
        }
    }
    
    /// Start the display link
    private func startDisplayLink() {
        if let displayLink = displayLink, !CVDisplayLinkIsRunning(displayLink) {
            CVDisplayLinkStart(displayLink)
        }
    }
    
    /// Stop the display link
    private func stopDisplayLink() {
        if let displayLink = displayLink, CVDisplayLinkIsRunning(displayLink) {
            CVDisplayLinkStop(displayLink)
        }
    }
    
    /// Called when the display link fires
    @objc private func displayLinkDidFire() {
        // Check if we need to update the UI for transition progress
        updateTransitionProgress()
        
        // Update data on main thread
        updateVisualizationData()
    }
    
    /// Update visualization data on the main thread
    private func updateVisualizationData() {
        // Make sure we update the UI on the main thread
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            self.lock.lock()
            // Copy data for thread safety
            let spectrumCopy = self.spectrumData
            let waveformCopy = self.waveformData
            let levelsCopy = self.audioLevels
            self.lock.unlock()
            
            // Update published properties
            self.spectrumData = spectrumCopy
            self.waveformData = waveformCopy
            self.audioLevels = levelsCopy
        }
    }
    
    /// Update transition progress
    private func updateTransitionProgress() {
        guard transitionProgress < 1.0 else {
            // Transition complete
            transitionTimer?.invalidate()
            transitionTimer = nil
            return
        }
        
        let currentTime = CFAbsoluteTimeGetCurrent()
        let elapsedTime = currentTime - transitionStartTime
        let progress = min(1.0, elapsedTime / configuration.transitionDuration)
        
        // Update the progress on the main thread
        DispatchQueue.main.async { [weak self] in
            self?.transitionProgress = progress
        }
        
        // If transition is complete, clean up
        if progress >= 1.0 {
            transitionTimer?.invalidate()
            transitionTimer = nil
        }
    }
    
    /// Process audio data for spectrum visualization
    private func processSpectrumData(_ audioData: [Float]) {
        // Begin performance monitoring
        performanceMonitor?.beginMeasuring("ProcessSpectrum")
        
        // Extract or compute frequency spectrum data
        let fftSize = min(2048, audioData.count)
        var spectrum = processFFT(audioData, fftSize: fftSize)
        
        // Apply settings from configuration
        let settings = configuration.spectrumSettings
        
        // Resize to match configuration's desired bar count
        spectrum = resizeSpectrumData(spectrum, targetSize: settings.barCount)
        
        // Apply logarithmic scaling if configured
        if settings.useLogScale {
            applyLogScaling(&spectrum, minFreq: settings.minFrequency, maxFreq: settings.maxFrequency)
        }
        
        // Apply smoothing
        applySpectrumSmoothing(&spectrum)
        
        // Update spectrum data (thread-safe)
        lock.lock()
        spectrumData = spectrum
        lock.unlock()
        
        // End performance monitoring
        performanceMonitor?.endMeasuring("ProcessSpectrum")
    }
    
    /// Process FFT on audio data
    private func processFFT(_ audioData: [Float], fftSize: Int) -> [Float] {
        // This is a simplified placeholder implementation.
        // In a real implementation, you would use vDSP for efficient FFT processing.
        
        // For now, return a simulated spectrum based on the audio data
        var spectrum = [Float](repeating: 0, count: min(128, fftSize / 2))
        
        // Generate a fake spectrum based on audio amplitude for testing
        if !audioData.isEmpty {
            let amplitude = audioData.reduce(0, { max($0, abs($1)) })
            
            for i in 0..<spectrum.count {
                // Create a curve that emphasizes mid-frequencies
                let normalizedIndex = Float(i) / Float(spectrum.count)
                let value = amplitude * sin(normalizedIndex * .pi) * (1.0 - 0.7 * abs(normalizedIndex - 0.5))
                spectrum[i] = value
            }
        }
        
        return spectrum
    }
    
    /// Resize spectrum data to target size
    private func resizeSpectrumData(_ data: [Float], targetSize: Int) -> [Float] {
        guard data.count != targetSize, !data.isEmpty, targetSize > 0 else {
            return data
        }
        
        var result = [Float](repeating: 0, count: targetSize)
        
        // Simple linear interpolation for resizing
        for i in 0..<targetSize {
            let index = Float(i) * Float(data.count) / Float(targetSize)
            let lowerIndex = Int(index)
            let upperIndex = min(lowerIndex + 1, data.count - 1)
            let fraction = index - Float(lowerIndex)
            
            result[i] = data[lowerIndex] * (1 - fraction) + data[upperIndex] * fraction
        }
        
        return result
    }
    
    /// Apply logarithmic scaling to spectrum data
    private func applyLogScaling(_ spectrum: inout [Float], minFreq: Float, maxFreq: Float) {
        guard !spectrum.isEmpty else { return }
        
        // Create a temporary buffer for the result
        var result = [Float](repeating: 0, count: spectrum.count)
        
        // Calculate logarithmic frequency bands
        let minLog = log10(max(20.0, Float(minFreq)))
        let maxLog = log10(min(20000.0, Float(maxFreq)))
        let logRange = maxLog - minLog
        
        for i in 0..<result.count {
            // Map i to logarithmic frequency
            let normalizedIndex = Float(i) / Float(result.count - 1)
            let logFreq = pow(10, minLog + normalizedIndex * logRange)
            
            // Map log frequency to linear spectrum index
            let linearIndex = (logFreq - minFreq) / (maxFreq - minFreq) * Float(spectrum.count - 1)
            
            // Interpolate to get the value
            let lowerIndex = max(0, min(spectrum.count - 1, Int(floor(linearIndex))))
            let upperIndex = max(0, min(spectrum.count - 1, Int(ceil(linearIndex))))
            let fraction = linearIndex - Float(lowerIndex)
            
            result[i] = spectrum[lowerIndex] * (1 - fraction) + spectrum[upperIndex] * fraction
        }
        
        spectrum = result
    }
    
    /// Apply smoothing to spectrum data
    private func applySpectrumSmoothing(_ spectrum: inout [Float]) {
        // Add the current spectrum to history for smoothing
        spectrumHistory.append(spectrum)
        
        // Keep a limited history
        let maxHistory = 10
        if spectrumHistory.count > maxHistory {
            spectrumHistory.removeFirst(spectrumHistory.count - maxHistory)
        }
        
        // Apply smoothing factor from configuration
        let smoothingFactor = configuration.fftSmoothingFactor
        
        // If we have history, apply temporal smoothing
        if spectrumHistory.count > 1 {
            let lastSpectrum = spectrumHistory[spectrumHistory.count - 2]
            
            // Apply smoothing only if sizes match
            if lastSpectrum.count == spectrum.count {
                for i in 0..<spectrum.count {
                    spectrum[i] = lastSpectrum[i] * (1 - smoothingFactor) + spectrum[i] * smoothingFactor
                }
            }
        }
        
        // Also apply spatial (frequency) smoothing
        let spatialSmoothing: Float = 0.2
        if spectrum.count > 2 {
            var smoothed = spectrum
            
            for i in 1..<(spectrum.count - 1) {
                smoothed[i] = spectrum[i-1] * spatialSmoothing * 0.5 +
                              spectrum[i] * (1 - spatialSmoothing) +
                              spectrum[i+1] * spatialSmoothing * 0.5
            }
            
            spectrum = smoothed
        }
    }
    
    /// Process audio data for waveform visualization
    private func processWaveformData(_ audioData: [Float]) {
        // Begin performance monitoring
        performanceMonitor?.beginMeasuring("ProcessWaveform")
        
        // Extract or compute waveform data
        var waveform = processWaveform(audioData)
        
        // Apply settings from configuration
        let settings = configuration.waveformSettings
        
        // Apply smoothing if configured
        if settings.smoothing > 0 {
            applyWaveformSmoothing(&waveform, amount: settings.smoothing)
        }
        
        // Store in waveform history if multiple lines are enabled
        if settings.showMultipleLines {
            waveformHistory.append(waveform)
            
            // Keep a limited history
            let maxHistory = 5
            if waveformHistory.count > maxHistory {
                waveformHistory.removeFirst(waveformHistory.count - maxHistory)
            }
        }
        
        // Update waveform data (thread-safe)
        lock.lock()
        waveformData = waveform
        lock.unlock()
        
        // End performance monitoring
        performanceMonitor?.endMeasuring("ProcessWaveform")
    }
    
    /// Process audio data into waveform display data
    private func processWaveform(_ audioData: [Float]) -> [Float] {
        // Ensure we have an appropriate sample count for visualization
        let targetSampleCount = 512
        var waveform = [Float](repeating: 0, count: targetSampleCount)
        
        if audioData.isEmpty {
            return waveform
        }
        
        // Resample the audio data to fit our target sample count
        let step = Float(audioData.count) / Float(targetSampleCount)
        
        for i in 0..<targetSampleCount {
            let index = min(Int(Float(i) * step), audioData.count - 1)
            waveform[i] = audioData[index]
        }
        
        // Normalize to range -1.0 to 1.0
        normalizeWaveform(&waveform)
        
        return waveform
    }
    
    /// Apply smoothing to waveform data
    private func applyWaveformSmoothing(_ waveform: inout [Float], amount: Float) {
        guard waveform.count > 2 else { return }
        
        let smoothing = min(max(amount, 0), 1)
        var smoothed = waveform
        
        // Apply a simple moving average filter
        for i in 1..<(waveform.count - 1) {
            smoothed[i] = waveform[i-1] * smoothing * 0.5 +
                         waveform[i] * (1 - smoothing) +
                         waveform[i+1] * smoothing * 0.5
        }
        
        // Handle endpoints
        smoothed[0] = waveform[0] * (1 - smoothing/2) + waveform[1] * smoothing/2
        smoothed[waveform.count - 1] = waveform[waveform.count - 1] * (1 - smoothing/2) + 
                                     waveform[waveform.count - 2] * smoothing/2
        
        waveform = smoothed
    }
    
    /// Normalize waveform data to range -1.0 to 1.0
    private func normalizeWaveform(_ waveform: inout [Float]) {
        guard !waveform.isEmpty else { return }
        
        // Find the maximum absolute value
        var maxValue: Float = 0
        for value in waveform {
            maxValue = max(maxValue, abs(value))
        }
        
        // Normalize only if we have a non-zero maximum
        if maxValue > 0.001 {
            for i in 0..<waveform.count {
                waveform[i] = waveform[i] / maxValue
            }
        }
    }
    
    /// Get visualization data suitable for the current mode
    public func getVisualizationData() -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        
        switch currentMode {
        case .spectrum:
            return spectrumData
        case .waveform:
            return waveformData
        case .particles, .neural:
            // For particles and neural patterns, we provide frequency data
            // as they typically need frequency information for reactivity
            return spectrumData
