import XCTest
import Metal
import AVFoundation
import Combine
@testable import AudioBloomCore
@testable import AudioBloomUI
@testable import Visualizer

/// Integration tests for AudioBloom focusing on component interaction and real-world scenarios
final class AudioBloomIntegrationTests: XCTestCase {
    // Test components
    var audioEngine: AudioEngine!
    var visualizer: AudioVisualizer!
    var presetManager: PresetManager!
    var performanceMonitor: PerformanceMonitor!
    var settings: AudioBloomSettings!
    
    // Metal resources
    var metalDevice: MTLDevice!
    
    // Test resources
    var testAudioURL: URL!
    var cancellables = Set<AnyCancellable>()
    
    // Flags
    var isM3Hardware: Bool = false
    var aneCoreML: Bool = false // Apple Neural Engine availability
override func setUp() {
        super.setUp()
        
        // Check if auto-optimization kicked in
        if performanceMonitor.autoOptimizationEnabled {
            let recommendations = performanceMonitor.getOptimizationRecommendations()
            XCTAssertFalse(recommendations.isEmpty, "Should generate optimization recommendations under load")
            
            // Verify quality level was appropriately adjusted
            let currentQuality = performanceMonitor.currentQualityLevel
            XCTAssertNotEqual(currentQuality, .ultra, "Quality should be reduced from ultra under heavy load")
        }
        
        // Verify render times are still acceptable
        let averageRenderTime = performanceMonitor.getAverageTime(for: "LoadTestRender")
        
        // M3 Pro hardware should handle even the most intensive workloads well
        if isM3Hardware {
            XCTAssertLessThan(averageRenderTime, 10.0, "Render time should stay under 10ms on M3 Pro even under load")
        } else {
            XCTAssertLessThan(averageRenderTime, 33.3, "Render time should stay under 33.3ms (30fps) on all hardware")
        }
        
        // Check ANE capability
        if #available(macOS 12.0, *) {
        performanceMonitor.startMonitoring()
        performanceMonitor.setQualityLevel(.ultra) // Start at maximum quality
        
        // Load audio file
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Create a stress scenario by rapidly switching themes and processing simultaneously
        var currentFrame = 0
        let totalFrames = 300 // Run for longer to test stability
        
        // Track optimization events
        var qualityAdjustments = 0
        var neuralOptimizations = 0
        
        // Set up observer for quality level changes 
        performanceMonitor.qualityLevelDidChange
            .sink { _ in
                qualityAdjustments += 1
            }
            .store(in: &cancellables)
        
        // Set up observer for neural engine optimizations
        NotificationCenter.default.addObserver(
            forName: Notification.Name("NeuralEngineOptimized"),
            object: nil,
            queue: .main
        ) { _ in
            neuralOptimizations += 1
        }
        
        // Create test visualization pipeline
        audioEngine.fftPublisher
            .sink { [weak self] fftData in
                guard let self = self else { return }
                
                // Get audio levels
                let audioLevels = AudioLevels(
                    bass: self.audioEngine.bassLevel,
                    mid: self.audioEngine.midLevel,
                    treble: self.audioEngine.trebleLevel,
                    left: self.audioEngine.leftLevel,
                    right: self.audioEngine.rightLevel
                )
                
                // Process with neural engine (while adding variable load)
                self.performanceMonitor.beginMeasuring("M3NeuralProcessing")
                self.audioEngine.processAudioWithNeuralEngine(fftData: fftData, levels: audioLevels) { neuralResults in
                    let processingTime = self.performanceMonitor.endMeasuring("M3NeuralProcessing")
                    
                    // Add extra computational load every few frames
                    if currentFrame % 30 == 0 {
                        _ = self.performExtraWorkload()
                    }
                    
                    // Update visualizer with neural data if available
                    if let results = neuralResults {
                        self.visualizer.updateNeuralData(results)
                    }
                }
                
                // Change theme periodically to stress test
                if currentFrame % 50 == 0 {
                    let themes = VisualTheme.allCases
                    let themeIndex = (currentFrame / 50) % themes.count
                    self.visualizer.updateTheme(themes[themeIndex])
                }
                
                // Update visualizer with audio data
                self.visualizer.updateAudioData(fftData: fftData, levels: audioLevels)
                
                // Render frame with full optimization
                self.performanceMonitor.beginMeasuring("M3OptimizedRender")
                let texture = self.visualizer.renderFrame()
                let renderTime = self.performanceMonitor.endMeasuring("M3OptimizedRender")
                
                // Verify frame was rendered
                XCTAssertNotNil(texture, "Frame should render in full system test")
                
                // Check if M3 hardware is properly optimizing rendering
                if currentFrame > 100 && self.isM3Hardware {
                    XCTAssertLessThan(renderTime, 12.0, "M3 hardware should render frames under 12ms even with full workload")
                }
                
                // Monitor memory usage and trigger artificial memory warning to test adaptability
                if currentFrame == 150 {
                    self.performanceMonitor.simulateMemoryWarning()
                }
                
                // Count frames
                currentFrame += 1
                
                // After sufficient frames, fulfill expectation
                if currentFrame >= totalFrames {
                    fullSystemExpectation.fulfill()
                }
            }
            .store(in: &cancellables)
        
        // Start audio
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for test completion with a generous timeout since we're doing a lot
        wait(for: [fullSystemExpectation], timeout: 90.0)
        
        // Verify optimization was effective
        XCTAssertGreaterThan(qualityAdjustments, 0, "Quality should have been dynamically adjusted")
        
        // Get final performance metrics
        let finalFPS = performanceMonitor.fps
        let finalQuality = performanceMonitor.currentQualityLevel
        let cpuUsage = performanceMonitor.cpuUsage
        let memoryUsage = performanceMonitor.memoryUsage
        
        // Check memory impact - should recover after memory warning
        XCTAssertLessThan(memoryUsage, 500.0, "Memory usage should be kept reasonable")
        
        // On M3 Pro, performance should be excellent
        XCTAssertGreaterThan(finalFPS, 45, "Final FPS should be above 45 on M3 Pro")
        
        // Verify consistent performance with ANE utilization
        let neuralTimes = performanceMonitor.getAllTimes(for: "M3NeuralProcessing")
        if !neuralTimes.isEmpty {
            // Calculate standard deviation of neural processing times to verify consistency
            let mean = neuralTimes.reduce(0, +) / Double(neuralTimes.count)
            let variance = neuralTimes.map { pow($0 - mean, 2) }.reduce(0, +) / Double(neuralTimes.count)
            let stdDev = sqrt(variance)
            
            // M3 Pro with ANE should have very consistent neural processing times
            XCTAssertLessThan(stdDev, 3.0, "Neural processing times should be very consistent on M3 Pro with ANE")
            XCTAssertLessThan(mean, 8.0, "Average neural processing time should be under 8ms on M3 Pro")
        }
        
        // Verify the full performance report
        let fullReport = performanceMonitor.generatePerformanceReport()
        XCTAssertTrue(fullReport.contains("AudioBloom Performance Report"), "Full performance report should be generated")
        XCTAssertTrue(fullReport.contains("M3 Optimizations"), "Report should include M3 optimization data")
        
        // Check Neural Engine utilization metrics
        XCTAssertTrue(performanceMonitor.isUsingANE, "Should be using Apple Neural Engine on M3 Pro")
        XCTAssertGreaterThan(performanceMonitor.aneUtilization, 0.2, "ANE utilization should be significant")
    }
    
    /// Tests end-to-end integration with real-time preset switching and neural processing
    ///
    /// This comprehensive integration test validates the complete AudioBloom workflow,
    /// focusing on real-world usage patterns including preset switching during active
    /// visualization. It ensures all components work together seamlessly through
    /// multiple configuration changes.
    func testEndToEndIntegration() {
        // Start performance monitoring
        performanceMonitor.startMonitoring()
the most intensive visualization theme with neural processing enabled.
    /// It validates that the performance monitoring system correctly identifies optimization
    /// opportunities and makes appropriate quality adjustments to maintain acceptable
    /// frame rates even under stress.
    func testPerformanceUnderLoad() {
        // Set up a scenario with high CPU/GPU load
        settings.currentTheme = .cosmic // Most intensive theme
erify that GPU shader compilation and resource management
    /// is optimized for M3 Pro hardware. It ensures smooth transitions between
    /// visual themes without significant performance degradation.
    func testThemeSwitchingPerformance() {
        // Skip detailed performance test on non-M3 hardware
        guard isM3Hardware else {
ween the PresetManager, settings object, and
    /// the visualization system.
    func testPresetApplicationToVisualization() {
        // Create a test preset with specific settings
        let testPreset = Preset(
nalysis to visualization.
    /// It tests that audio data properly flows through the system and results 
    /// in correctly rendered visualization frames.
    func testAudioToVisualizationPipeline() {
        // Create expectation for rendering completion
        let expectation = XCTestExpectation(description: "Audio visualization pipeline")
focus on real-time audio analysis, beat detection, and
    /// emotional analysis performance on M3 Pro hardware.
    func testNeuralEngineIntegration() {
        // Skip if not running on M3 hardware
        guard isM3Hardware else {
            print("Skipping neural engine integration test on non-M3 hardware")
            return
e handling, and ANE utilization.
    func testFullSystemIntegrationWithNeuralEngine() {
        // Skip if not running on M3 hardware
        guard isM3Hardware else {
            print("Skipping full system integration test on non-M3 hardware")
            return
        }
        
        // Enable neural engine processing
        settings.neuralEngineEnabled = true
        settings.beatSensitivity = 0.8
        settings.patternSensitivity = 0.7
        settings.emotionalSensitivity = 0.9
        
        // Start performance monitoring
        performanceMonitor.startMonitoring()
        
        // Configure audio for test
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Set up visualization
        settings.currentTheme = .cosmic
        visualizer.updateTheme(.cosmic)
        
        // Create expectations
        let neuralProcessingExpectation = XCTestExpectation(description: "Neural processing completed")
        let renderingExpectation = XCTestExpectation(description: "Visualization rendering completed")
        
        // Track neural processing results
        var beatDetections = 0
        var patternDetections = 0
        var emotionalUpdates = 0
        
        // Add observer for neural events
        let beatObserver = NotificationCenter.default.addObserver(
            forName: Notification.Name("BeatDetected"),
            object: nil,
            queue: .main
        ) { _ in
            beatDetections += 1
        }
        
        let patternObserver = NotificationCenter.default.addObserver(
            forName: Notification.Name("PatternDetected"),
            object: nil,
            queue: .main
        ) { _ in
            patternDetections += 1
        }
        
        let emotionObserver = NotificationCenter.default.addObserver(
            forName: Notification.Name("EmotionalAnalysisComplete"),
            object: nil,
            queue: .main
        ) { _ in
            emotionalUpdates += 1
            
            // After receiving some emotional data, consider neural processing sufficient
            if emotionalUpdates >= 3 {
                neuralProcessingExpectation.fulfill()
            }
        }
        
        // Connect audio to visualization with neural processing
        var framesRendered = 0
        
        audioEngine.fftPublisher
            .sink { [weak self] fftData in
                guard let self = self else { return }
                
                // Get audio levels
                let audioLevels = AudioLevels(
                    bass: self.audioEngine.bassLevel,
                    mid: self.audioEngine.midLevel,
                    treble: self.audioEngine.trebleLevel,
                    left: self.audioEngine.leftLevel,
                    right: self.audioEngine.rightLevel
                )
                
                // Process with neural engine
                self.performanceMonitor.beginMeasuring("NeuralProcessing")
                self.audioEngine.processAudioWithNeuralEngine(fftData: fftData, levels: audioLevels) { neuralResults in
                    let processingTime = self.performanceMonitor.endMeasuring("NeuralProcessing")
                    
                    // If we get neural results, update visualization
                    if let results = neuralResults {
                        self.visualizer.updateNeuralData(results)
                    }
                    
                    // On M3 Pro, neural processing should be very efficient
                    XCTAssertLessThan(processingTime, 10.0, "Neural processing should be under 10ms on M3 Pro")
                }
                
                // Update visualizer with audio data
                self.visualizer.updateAudioData(fftData: fftData, levels: audioLevels)
                
                // Render frame
                self.performanceMonitor.beginMeasuring("NeuralRenderFrame")
                let texture = self.visualizer.renderFrame()
                let renderTime = self.performanceMonitor.endMeasuring("NeuralRenderFrame")
                
                XCTAssertNotNil(texture, "Frame should render with neural data")
                
                // Count frames
                framesRendered += 1
                if framesRendered >= 100 {
                    renderingExpectation.fulfill()
                }
            }
            .store(in: &cancellables)
        
        // Start audio
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for test completion
        wait(for: [neuralProcessingExpectation, renderingExpectation], timeout: 30.0)
        
        // Remove observers
        NotificationCenter.default.removeObserver(beatObserver)
        NotificationCenter.default.removeObserver(patternObserver)
        NotificationCenter.default.removeObserver(emotionObserver)
        
        // Verify neural engine integration worked
        XCTAssertGreaterThan(beatDetections, 0, "Neural engine should detect beats")
        XCTAssertGreaterThan(emotionalUpdates, 2, "Neural engine should provide emotional analysis")
        
        // Verify performance was acceptable
        let avgNeuralRenderTime = performanceMonitor.getAverageTime(for: "NeuralRenderFrame")
        let avgNeuralProcessingTime = performanceMonitor.getAverageTime(for: "NeuralProcessing")
        
        // On M3 Pro, these should be very efficient
        XCTAssertLessThan(avgNeuralRenderTime, 8.0, "Average neural render time should be under 8ms on M3 Pro")
        XCTAssertLessThan(avgNeuralProcessingTime, 5.0, "Average neural processing time should be under 5ms on M3 Pro")
        
        // Verify ANE utilization
        XCTAssertTrue(performanceMonitor.isUsingANE, "Should be using Apple Neural Engine on M3 Pro")
    }
    
    /// Tests end-to-end integration with real-time preset switching and neural processing
    func testEndToEndIntegration() {
        // Start performance monitoring
        performanceMonitor.startMonitoring()
        
        // Configure audio
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Create expectations
        let endToEndExpectation = XCTestExpectation(description: "End-to-end integration test")
        
        // Create several test presets with different settings
        let presets = createTestPresets()
        
        // Set up visualization pipeline
        var frameCount = 0
        var currentPresetIndex = 0
        let targetFrames = 200 // Run through a good number of frames
        let framesPerPreset = 50 // Switch presets every 50 frames
        
        // Connect audio to visualization
        audioEngine.fftPublisher
            .sink { [weak self] fftData in
                guard let self = self else { return }
                
                // Get audio levels
                let audioLevels = AudioLevels(
                    bass: self.audioEngine.bassLevel,
                    mid: self.audioEngine.midLevel,
                    treble: self.audioEngine.trebleLevel,
                    left: self.audioEngine.leftLevel,
                    right: self.audioEngine.rightLevel
                )
                
                // Switch presets periodically
                if frameCount % framesPerPreset == 0 && frameCount > 0 {
                    currentPresetIndex = (currentPresetIndex + 1) % presets.count
                    let preset = presets[currentPresetIndex]
                    self.presetManager.applyPreset(preset)
                }
                
                // Process with neural engine if enabled
                if self.settings.neuralEngineEnabled {
                    self.audioEngine.processAudioWithNeuralEngine(fftData: fftData, levels: audioLevels) { neuralResults in
                        if let results = neuralResults {
                            self.visualizer.updateNeuralData(results)
                        }
                    }
                }
                
                // Update visualizer with audio data
                self.visualizer.updateAudioData(fftData: fftData, levels: audioLevels)
                
                // Render frame
                let texture = self.visualizer.renderFrame()
                
                // Verify frame was rendered
                XCTAssertNotNil(texture, "Frame should render in end-to-end test")
                
                // Count frames
                frameCount += 1
                if frameCount >= targetFrames {
                    endToEndExpectation.fulfill()
                }
            }
            .store(in: &cancellables)
        
        // Start audio
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for test completion
        wait(for: [endToEndExpectation], timeout: 60.0)
        
        // Verify performance metrics for end-to-end test
        XCTAssertGreaterThan(performanceMonitor.fps, 30, "FPS should stay above 30 in end-to-end test")
        
        // Check final performance report
        let report = performanceMonitor.generatePerformanceReport()
        XCTAssertTrue(report.contains("AudioBloom Performance Report"), "Should generate comprehensive performance report")
        
        // Clean up
        for preset in presets {
            presetManager.deletePreset(preset)
        }
    }
    
    // MARK: - Helper Methods
    
    /// Handles memory warnings for testing memory pressure scenarios
    /// 
    /// This method simulates the system's memory warning notification handling,
    /// allowing tests to verify the application correctly responds to memory pressure.
    @objc func handleMemoryWarning() {
        performanceMonitor.simulateMemoryWarning()
        
        // Verify application responds to memory pressure
        XCTAssertLessThan(performanceMonitor.currentQualityLevel, .ultra, "Quality should be reduced under memory pressure")
    }
    
    /// Generates test audio for integration testing
    /// 
    /// Creates a synthetic audio file containing predictable patterns including
    /// sine waves, harmonics and rhythmic beats that can be used to test both
    /// audio processing and neural analysis features of the application.
    /// The generated audio contains frequency and time-domain features that
    /// can trigger known responses in the visualization engine.
    private func generateTestAudio() {
 task that simulates additional processing
    /// load on the CPU. This is used to verify the application's ability to handle
    /// varying levels of system load while maintaining audiovisual performance.
    /// 
    /// - Returns: The time in seconds that the workload took to complete
    private func performExtraWorkload() -> Double {
        // Create a computationally intensive task to simulate extra load
        let workloadSize = 1000
        var result = 0.0
        
        // Start timing
        let startTime = CACurrentMediaTime()
        
        // Perform matrix-like operations
        for i in 0..<workloadSize {
            for j in 0..<100 {
                result += sin(Double(i) * cos(Double(j)))
            }
        }
        
        // End timing
        let endTime = CACurrentMediaTime()
        
        return endTime - startTime
    }
    
    /// Creates multiple test presets for integration testing
    /// 
    /// Generates a diverse set of preset configurations that exercise different
    /// aspects of the application's settings system. These presets range from
    /// low-CPU configurations to neural-intensive settings for comprehensive testing.
    /// 
    /// - Returns: An array of created presets for testing
    private func createTestPresets() -> [Preset] {
        let presetConfigs: [(String, VisualTheme, Bool, Float)] = [
            ("Classic Integration", .classic, false, 0.7),
            ("Neon Neural", .neon, true, 0.85),
            ("Cosmic High Motion", .cosmic, true, 1.0),
            ("Monochrome Low CPU", .monochrome, false, 0.5)
        ]
        
        var createdPresets: [Preset] = []
        
        for (name, theme, neuralEnabled, sensitivity) in presetConfigs {
            let preset = Preset(
                name: name,
                description: "Integration test preset",
                visualSettings: VisualizationSettings(
                    theme: theme,
                    sensitivity: sensitivity,
                    motionIntensity: sensitivity,
                    showFPS: true,
                    showBeatIndicator: neuralEnabled
                ),
                audioSettings: AudioSettings(
                    inputDevice: nil,
                    outputDevice: nil,
                    audioSource: "File",
                    micVolume: 0.8,
                    systemAudioVolume: 0.0,
                    mixInputs: false
                ),
                neuralSettings: NeuralSettings(
                    enabled: neuralEnabled,
                    beatSensitivity: sensitivity,
                    patternSensitivity: sensitivity * 0.9,
                    emotionalSensitivity: sensitivity * 0.8
                )
            )
            
            // Add to preset manager
            presetManager.addPreset(preset)
            
            // Add to created presets for return
            createdPresets.append(preset)
        }
        
return createdPresets
    }
    
    /// Generates test audio for integration testing
    private func generateTestAudio() {
        // Create a sine wave audio file with beats for testing
        let sampleRate: Double = 44100.0
        let duration: Double = 10.0  // 10 seconds
        let frequency: Double = 440.0  // A 440Hz
        let beatFrequency: Double = 2.0  // 2Hz for beat testing
        
        // Format settings
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        
        // Create audio buffer
        let frameCount = AVAudioFrameCount(sampleRate * duration)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        buffer.frameLength = frameCount
        
        // Fill buffer with sine wave data
        for frame in 0..<Int(frameCount) {
            let time = Double(frame) / sampleRate
            
            // Create sine wave with beats and harmonics for neural engine to detect
            let sineValue = sin(2.0 * .pi * frequency * time)
            let beatValue = abs(sin(2.0 * .pi * beatFrequency * time))
            let harmonic1 = 0.3 * sin(2.0 * .pi * frequency * 2 * time) // First harmonic
            let harmonic2 = 0.15 * sin(2.0 * .pi * frequency * 3 * time) // Second harmonic
            
            // Create pattern every 2 seconds
            let patternValue = abs(sin(2.0 * .pi * 0.5 * time))
            
            // Combine all components
            let value = Float(sineValue * (0.5 + 0.5 * beatValue) + harmonic1 + harmonic2 * patternValue)
            
            // Set left and right channels
            buffer.floatChannelData?[0][frame] = value
            buffer.floatChannelData?[1][frame] = value
        }
        
        // Write buffer to file
        try? AVAudioFile(forWriting: testAudioURL, settings: format.settings)
            .write(from: buffer)
    }
    
    /// Creates multiple test presets for integration testing
    private func createTestPresets() -> [Preset] {
        let presetConfigs: [(String, VisualTheme, Bool, Float)] = [
            ("Classic Integration", .classic, false, 0.7),
            ("Neon Neural", .neon, true, 0.85),
            ("Cosmic High Motion", .cosmic, true, 1.0),
            ("Monochrome Low CPU", .monochrome, false, 0.5)
        ]
        
        var createdPresets: [Preset] = []
        
        for (name, theme, neuralEnabled, sensitivity) in presetConfigs {
            let preset = Preset(
                name: name,
                description: "Integration test preset",
                visualSettings: VisualizationSettings(
                    theme: theme,
                    sensitivity: sensitivity,
                    motionIntensity: sensitivity,
                    showFPS: true,
                    showBeatIndicator: neuralEnabled
                ),
                audioSettings: AudioSettings(
                    inputDevice: nil,
                    outputDevice: nil,
                    audioSource: "File",
                    micVolume: 0.8,
                    systemAudioVolume: 0.0,
                    mixInputs: false
                ),
                neuralSettings: NeuralSettings(
                    enabled: neuralEnabled,
                    beatSensitivity: sensitivity,
                    patternSensitivity: sensitivity * 0.9,
                    emotionalSensitivity: sensitivity * 0.8
                )
            )
                neuralSettings: Neural
        
        // Create test audio
        let tempDir = NSTemporaryDirectory()
        testAudioURL = URL(fileURLWithPath: tempDir).appendingPathComponent("integration_test_audio.wav")
        generateTestAudio()
        
        // Set up Metal
        metalDevice = MTLCreateSystemDefaultDevice()
        
        // Initialize settings
        settings = AudioBloomSettings()
        
        // Initialize components in order of dependency
        performanceMonitor = PerformanceMonitor(metalDevice: metalDevice)
        audioEngine = AudioEngine(settings: settings)
        visualizer = AudioVisualizer(device: metalDevice, settings: settings)
        presetManager = PresetManager(settings: settings)
        
        // Register for memory warnings to test high-pressure scenarios
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )
    }
    
    override func tearDown() {
        // Clean up
        cancellables.removeAll()
        
        // Stop all active processes
        audioEngine.stop()
        performanceMonitor.stopMonitoring()
        
        // Clean up test audio
        try? FileManager.default.removeItem(at: testAudioURL)
        
        // Release components in reverse order
        presetManager = nil
        visualizer = nil
        audioEngine = nil
        performanceMonitor = nil
        settings = nil
        metalDevice = nil
        
        super.tearDown()
    }
    
    // MARK: - Integration Tests
    
    /// Tests the full audio to visualization pipeline
    func testAudioToVisualizationPipeline() {
        // Create expectation for rendering completion
        let expectation = XCTestExpectation(description: "Audio visualization pipeline")
        
        // Start performance monitoring
        performanceMonitor.startMonitoring()
        
        // Configure audio engine for file playback
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Set up visualization for Classic theme
        settings.currentTheme = .classic
        visualizer.updateTheme(.classic)
        
        // Create a counter for received frames
        var framesRendered = 0
        
        // Connect audio engine FFT output to visualizer
        audioEngine.fftPublisher
            .sink { [weak self] fftData in
                guard let self = self else { return }
                
                // Get audio levels from engine
                let audioLevels = AudioLevels(
                    bass: self.audioEngine.bassLevel,
                    mid: self.audioEngine.midLevel,
                    treble: self.audioEngine.trebleLevel,
                    left: self.audioEngine.leftLevel,
                    right: self.audioEngine.rightLevel
                )
                
                // Update visualizer with audio data
                self.visualizer.updateAudioData(fftData: fftData, levels: audioLevels)
                
                // Render a frame
                self.performanceMonitor.beginMeasuring("IntegrationRenderFrame")
                let texture = self.visualizer.renderFrame()
                self.performanceMonitor.endMeasuring("IntegrationRenderFrame")
                
                XCTAssertNotNil(texture, "Rendered texture should not be nil")
                
                framesRendered += 1
                
                // After we've processed a few frames, consider the test successful
                if framesRendered >= 30 {
                    expectation.fulfill()
                }
            }
            .store(in: &cancellables)
        
        // Start audio playback
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for frames to be rendered
        wait(for: [expectation], timeout: 10.0)
        
        // Verify performance was tracked
        XCTAssertGreaterThan(performanceMonitor.fps, 0, "FPS should be tracked during rendering")
        
        // Verify the integration metrics
        let report = performanceMonitor.generatePerformanceReport()
        XCTAssertTrue(report.contains("AudioBloom Performance Report"), "Performance report should be generated")
    }
    
    /// Tests preset application and its effect on visualization
    func testPresetApplicationToVisualization() {
        // Create a test preset with specific settings
        let testPreset = Preset(
            name: "Integration Test Preset",
            description: "For testing preset application in the visualization pipeline",
            visualSettings: VisualizationSettings(
                theme: .neon,
                sensitivity: 0.75,
                motionIntensity: 0.85,
                showFPS: true,
                showBeatIndicator: true
            ),
            audioSettings: AudioSettings(
                inputDevice: nil,
                outputDevice: nil,
                audioSource: "File",
                micVolume: 0.8,
                systemAudioVolume: 0.0,
                mixInputs: false
            ),
            neuralSettings: NeuralSettings(
                enabled: true,
                beatSensitivity: 0.7,
                patternSensitivity: 0.6,
                emotionalSensitivity: 0.5
            )
        )
        
        // Add the preset to the preset manager
        presetManager.addPreset(testPreset)
        
        // Apply the preset - this should update settings and visualizer
        presetManager.applyPreset(testPreset)
        
        // Verify settings were applied
        XCTAssertEqual(settings.currentTheme, .neon, "Theme setting should be applied")
        XCTAssertEqual(settings.audioSensitivity, 0.75, "Sensitivity setting should be applied")
        XCTAssertEqual(settings.motionIntensity, 0.85, "Motion intensity setting should be applied")
        
        // Verify visualizer reflects the new settings
        XCTAssertEqual(visualizer.currentTheme, .neon, "Visualizer theme should match preset")
        XCTAssertEqual(visualizer.audioSensitivity, 0.75, "Visualizer sensitivity should match preset")
        
        // Test that the neural engine settings were applied
        XCTAssertTrue(settings.neuralEngineEnabled, "Neural engine should be enabled from preset")
        XCTAssertEqual(settings.beatSensitivity, 0.7, "Beat sensitivity should match preset")
        
        // Create test audio data
        let mockFFTData = Array(repeating: Float(0.5), count: 1024)
        let mockAudioLevels = AudioLevels(bass: 0.7, mid: 0.6, treble: 0.5, left: 0.65, right: 0.65)
        
        // Update visualizer with the mock data
        visualizer.updateAudioData(fftData: mockFFTData, levels: mockAudioLevels)
        
        // Render a frame with the preset settings
        let texture = visualizer.renderFrame()
        XCTAssertNotNil(texture, "Should render a frame with preset settings")
        
        // Clean up
        presetManager.deletePreset(testPreset)
    }
    
    /// Tests the performance impact of theme switching during active visualization
    func testThemeSwitchingPerformance() {
        // Skip detailed performance test on non-M3 hardware
        guard isM3Hardware else {
            print("Skipping detailed theme switching performance test on non-M3 hardware")
            return
        }
        
        // Start performance monitoring
        performanceMonitor.startMonitoring()
        
        // Set up audio playback
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Create expectations
        let audioStartedExpectation = XCTestExpectation(description: "Audio started")
        let themeSwitchingExpectation = XCTestExpectation(description: "Theme switching completed")
        
        // Monitor audio pipeline
        audioEngine.fftPublisher
            .sink { [weak self] fftData in
                guard let self = self else { return }
                
                // Get audio levels
                let audioLevels = AudioLevels(
                    bass: self.audioEngine.bassLevel,
                    mid: self.audioEngine.midLevel,
                    treble: self.audioEngine.trebleLevel,
                    left: self.audioEngine.leftLevel,
                    right: self.audioEngine.rightLevel
                )
                
                // Update visualizer
                self.visualizer.updateAudioData(fftData: fftData, levels: audioLevels)
                
                // Render frame
                _ = self.visualizer.renderFrame()
                
                // Fulfill audio started expectation on first frame
                audioStartedExpectation.fulfill()
            }
            .store(in: &cancellables)
        
        // Start audio
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for audio to start
        wait(for: [audioStartedExpectation], timeout: 5.0)
        
        // Create baseline performance metrics
        let baselineTime = Date()
        let baselineFPS = performanceMonitor.fps
        
        // Switch through all themes rapidly while rendering continues
        DispatchQueue.global().async {
            // Array to store theme switch times
            var themeSwitchTimes: [Double] = []
            
            // Switch through all themes
            for theme in VisualTheme.allCases {
                self.performanceMonitor.beginMeasuring("ThemeSwitch")
                
                // Switch theme
                DispatchQueue.main.sync {
                    self.settings.currentTheme = theme
                    self.visualizer.updateTheme(theme)
                }
                
                // Force render a frame with new theme
                _ = self.visualizer.renderFrame()
                
                // Record switch time
                let switchTime = self.performanceMonitor.endMeasuring("ThemeSwitch")
                themeSwitchTimes.append(switchTime)
                
                // Short delay to allow rendering to occur
                Thread.sleep(forTimeInterval: 0.1)
            }
            
            // Calculate average theme switch time
            let avgSwitchTime = themeSwitchTimes.reduce(0, +) / Double(themeSwitchTimes.count)
            
            // On M3 Pro hardware, theme switching should be very fast
            XCTAssertLessThan(avgSwitchTime, 50.0, "Theme switching should be under 50ms on M3 Pro")
            
            // Fulfill expectation
            themeSwitchingExpectation.fulfill()
        }
        
        // Wait for theme switching to complete
        wait(for: [themeSwitchingExpectation], timeout: 10.0)
        
        // Check FPS impact of theme switching
        let postSwitchFPS = performanceMonitor.fps
        
        // On M3 Pro, theme switching should have minimal impact on FPS
        XCTAssertGreaterThan(postSwitchFPS, baselineFPS * 0.8, "FPS drop during theme switching should be less than 20% on M3 Pro")
    }
    
    /// Tests audio to visualization pipeline with performance monitoring under load
    func testPerformanceUnderLoad() {
        // Set up a scenario with high CPU/GPU load
        settings.currentTheme = .cosmic // Most intensive theme
        settings.neuralEngineEnabled = true
        settings.audioSensitivity = 1.0 // Maximum sensitivity
        settings.motionIntensity = 1.0 // Maximum motion
        
        // Start performance monitoring
        performanceMonitor.startMonitoring()
        
        // Configure audio
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Create expectations
        let loadTestExpectation = XCTestExpectation(description: "Load test completed")
        
        // Set up visualization pipeline
        var frameCount = 0
        let targetFrames = 100 // Render 100 frames under load
        
        // Connect audio to visualization
        audioEngine.fftPublisher
            .sink { [weak self] fftData in
                guard let self = self else { return }
                
                // Simulate additional load
                let extraLoad = self.performExtraWorkload()
                
                // Get audio levels
                let audioLevels = AudioLevels(
                    bass: self.audioEngine.bassLevel,
                    mid: self.audioEngine.midLevel,
                    treble: self.audioEngine.trebleLevel,
                    left: self.audioEngine.leftLevel,
                    right: self.audioEngine.rightLevel
                )
                
                // Update visualizer
                self.visualizer.updateAudioData(fftData: fftData, levels: audioLevels)
                
                // Render frame
                self.performanceMonitor.beginMeasuring("LoadTestRender")
                let texture = self.visualizer.renderFrame()
                let renderTime = self.performanceMonitor.endMeasuring("LoadTestRender")
                
                // Verify frame was rendered
                XCTAssertNotNil(texture, "Frame should render under load")
                
                // Count frames
                frameCount += 1
                if frameCount >= targetFrames {
                    loadTestExpectation.fulfill()
                }
            }
            .store(in: &cancellables)
        
        // Start audio
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for load test to complete
        wait(for: [loadTestExpectation], timeout: 30.0)
        
        // Get performance metrics
        let fps = performanceMonitor.fps
        let cpuUsage = performanceMonitor.cpuUsage
        let memoryUsage = performanceMonitor.memoryUsage
        
        // Verify the application handles load appropriately
        XCTAssertGreaterThan(fps, 30.0, "FPS should stay above 30 even under load")
        
        // Check if auto-optimization kicked in
        if performanceMonitor.autoOptimizationEnabled {
            let recommendations =

