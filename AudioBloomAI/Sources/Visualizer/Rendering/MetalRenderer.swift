    /// Set audio levels for visualization
    /// - Parameters:
    ///   - bass: Bass frequency level (0.0-1.0)
    ///   - mid: Mid frequency level (0.0-1.0)
    ///   - treble: Treble frequency level (0.0-1.0)
    ///   - left: Left channel volume level (0.0-1.0)
    ///   - right: Right channel volume level (0.0-1.0)
    public func setAudioLevels(bass: Float, mid: Float, treble: Float, left: Float, right: Float) {
        // Apply safe bounds checking (0.0-1.0)
        bassLevel = max(0.0, min(1.0, bass))
        midLevel = max(0.0, min(1.0, mid))
        trebleLevel = max(0.0, min(1.0, treble))
        leftLevel = max(0.0, min(1.0, left))
        rightLevel = max(0.0, min(1.0, right))
    }
    
    /// Reset audio data to default state (silence)
    public func resetAudioData() {
        for i in 0..<audioData.count {
            audioData[i] = 0.0
        }
        
        bassLevel = 0.0
        midLevel = 0.0
        trebleLevel = 0.0
        leftLevel = 0.0
        rightLevel = 0.0
    }
    
    // MARK: - Visualization Parameters
    
    /// Set the audio sensitivity parameter
    /// - Parameter value: Sensitivity value (0.0-1.0)
    public func setSensitivity(_ value: Float) {
        sensitivity = max(0.0, min(1.0, value))
    }
    
    /// Set the motion intensity parameter
    /// - Parameter value: Motion intensity value (0.0-1.0)
    public func setMotionIntensity(_ value: Float) {
        motionIntensity = max(0.0, min(1.0, value))
    }
    
    /// Set the color intensity parameter
    /// - Parameter value: Color intensity value (0.0-1.0)
    public func setColorIntensity(_ value: Float) {
        colorIntensity = max(0.0, min(1.0, value))
    }
    
    /// Set the spectrum smoothing parameter
    /// - Parameter value: Smoothing value (0.0-1.0)
    public func setSpectrumSmoothing(_ value: Float) {
        spectrumSmoothing = max(0.0, min(1.0, value))
    }
    
    /// Set the particle count for particle visualization
    /// - Parameter count: Number of particles to display (1-100)
    public func setParticleCount(_ count: Int) {
        particleCount = Float(max(1, min(100, count)))
    }
    
    // MARK: - Theme Management
    
    /// Available visualization themes
    public enum VisualizationTheme: Int {
        case classic = 0
        case neon = 1
        case monochrome = 2
        case cosmic = 3
    }
    
    /// Set the visualization theme
    /// - Parameter theme: Theme to use
    public func setTheme(_ theme: VisualizationTheme) {
        themeIndex = Float(theme.rawValue)
    }
    
    /// Set custom theme colors
    /// - Parameters:
    ///   - primary: Primary theme color
    ///   - secondary: Secondary theme color
    ///   - background: Background color
    ///   - accent: Accent color for highlights
    public func setThemeColors(
        primary: SIMD4<Float>? = nil,
        secondary: SIMD4<Float>? = nil,
        background: SIMD4<Float>? = nil,
        accent: SIMD4<Float>? = nil
    ) {
        if let primary = primary {
            primaryColor = primary
        }
        
        if let secondary = secondary {
            secondaryColor = secondary
        }
        
        if let background = background {
            backgroundColor = background
        }
        
        if let accent = accent {
            accentColor = accent
        }
    }
    
    /// Reset theme colors to defaults
    public func resetThemeColors() {
        primaryColor = Constants.defaultPrimaryColor
        secondaryColor = Constants.defaultSecondaryColor
        backgroundColor = Constants.defaultBackgroundColor
        accentColor = Constants.defaultAccentColor
    }
    
    // MARK: - Visualization Mode Management
    
    /// Available visualization modes
    public enum VisualizationMode: Int {
        case spectrum = 0
        case waveform = 1
        case particles = 2
        case neural = 3
    }
    
    /// Set the visualization mode with optional transition
    /// - Parameters:
    ///   - mode: Visualization mode to switch to
    ///   - withTransition: Whether to animate the transition
    ///   - duration: Duration of the transition animation (if enabled)
    public func setVisualizationMode(_ mode: VisualizationMode, withTransition: Bool = true, duration: TimeInterval = 0.75) {
        // If the new mode is the same as current, do nothing
        if Float(mode.rawValue) == currentVisualizationMode {
            return
        }
        
        // Store the previous mode for transition
        previousVisualizationMode = currentVisualizationMode
        
        // Update to the new mode
        currentVisualizationMode = Float(mode.rawValue)
        
        // Set up transition if requested
        if withTransition {
            transitionStartTime = CACurrentMediaTime()
            transitionDuration = duration
            transitionProgress = 0.0
            isInTransition = true
        } else {
            // Skip transition
            transitionProgress = 1.0
            isInTransition = false
        }
    }
    
    // MARK: - Neural Visualization Parameters
    
    /// Set neural visualization parameters
    /// - Parameters:
    ///   - energy: Overall energy level (0.0-1.0)
    ///   - pleasantness: Pleasantness factor (0.0-1.0)
    ///   - complexity: Complexity factor (0.0-1.0)
    public func setNeuralParameters(energy: Float, pleasantness: Float, complexity: Float) {
        neuralEnergy = max(0.0, min(1.0, energy))
        neuralPleasantness = max(0.0, min(1.0, pleasantness))
        neuralComplexity = max(0.0, min(1.0, complexity))
    }
    
    /// Signal that a beat was detected in the audio
    /// - Parameter intensity: Beat intensity (0.0-1.0)
    public func signalBeatDetected(intensity: Float = 1.0) {
        // Set beat detected flag (will automatically decay in shader)
        beatDetected = max(0.0, min(1.0, intensity))
    }
    
    // MARK: - Resource Management
    
    /// Reset the renderer state
    public func reset() {
        // Reset audio data
        resetAudioData()
        
        // Reset timing
        startTime = CACurrentMediaTime()
        lastUpdateTime = 0
        
        // Reset transitions
        isInTransition = false
        transitionProgress = 1.0
    }
    
    /// Clean up resources when the renderer is no longer needed
    public func cleanup() {
        // Wait for any in-flight operations to complete
        for _ in 0..<Constants.maxInflightBuffers {
            _ = semaphore.wait(timeout: .distantFuture)
        }
        
        // Release resources
        device = nil
        commandQueue = nil
        renderPipelineState = nil
        vertexBuffer = nil
        uniformBuffer = nil
    }
    
    /// Structure to match the Metal shader uniform buffer
    private struct AudioUniforms {
        // Audio data
        var audioData: [Float] = [Float](repeating: 0, count: Constants.maxAudioFrames)
        var bassLevel: Float = 0
        var midLevel: Float = 0
        var trebleLevel: Float = 0
        var leftLevel: Float = 0
        var rightLevel: Float = 0
        
        // Theme colors
        var primaryColor: SIMD4<Float> = Constants.defaultPrimaryColor
        var secondaryColor: SIMD4<Float> = Constants.defaultSecondaryColor
        var backgroundColor: SIMD4<Float> = Constants.defaultBackgroundColor
        var accentColor: SIMD4<Float> = Constants.defaultAccentColor
        
        // Animation parameters
        var time: Float = 0
        var sensitivity: Float = 0.8
        var motionIntensity: Float = 0.7
        var themeIndex: Float = 0
        
        // Visualization settings
        var visualizationMode: Float = 0
        var transitionProgress: Float = 1.0
        var previousMode: Float = 0
        var colorIntensity: Float = 0.8
        var spectrumSmoothing: Float = 0.3
        var particleCount: Float = 50
        
        // Neural visualization parameters
        var neuralEnergy: Float = 0.5
        var neuralPleasantness: Float = 0.5
        var neuralComplexity: Float = 0.5
        var beatDetected: Float = 0
    }
}
