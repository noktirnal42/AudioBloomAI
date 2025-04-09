    // MARK: - Audio Engine Tests
    
    /// Tests basic audio engine initialization and setup
    func testAudioEngineInitialization() {
        XCTAssertNotNil(audioEngine, "Audio engine should be properly initialized")
        XCTAssertEqual(audioEngine.isRunning, false, "Audio engine should not be running initially")
    }
    
    /// Tests audio engine input/output device management
    func testAudioDeviceManagement() {
        let devices = audioEngine.getAvailableDevices()
        XCTAssertFalse(devices.inputs.isEmpty, "Should detect at least one input device")
        XCTAssertFalse(devices.outputs.isEmpty, "Should detect at least one output device")
        
        // Test default device selection
        XCTAssertNotNil(audioEngine.currentInputDevice, "Should have a default input device")
        XCTAssertNotNil(audioEngine.currentOutputDevice, "Should have a default output device")
    }
    
    /// Tests audio engine start/stop functionality
    func testAudioEngineStartStop() {
        // Start the engine
        do {
            try audioEngine.start()
            XCTAssertTrue(audioEngine.isRunning, "Audio engine should be running after start")
            
            // Test volume control
            audioEngine.setInputVolume(0.5)
            XCTAssertEqual(audioEngine.inputVolume, 0.5, accuracy: 0.01, "Input volume should be set correctly")
            
            // Stop the engine
            audioEngine.stop()
            XCTAssertFalse(audioEngine.isRunning, "Audio engine should not be running after stop")
        } catch {
            XCTFail("Audio engine start failed with error: \(error)")
        }
    }
    
    /// Tests audio processing and FFT analysis
    func testAudioProcessing() {
        // Configure for file playback
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Create expectation for audio processing
        let expectation = XCTestExpectation(description: "Audio processing")
        
        // Subscribe to FFT updates
        audioEngine.fftPublisher
            .sink { fftData in
                // Verify FFT data format and contents
                XCTAssertEqual(fftData.count, 1024, "FFT data should have 1024 frequency bins")
                XCTAssertFalse(fftData.allSatisfy { $0 == 0 }, "FFT data should not be all zeros")
                
                // Verify calculated audio levels
                XCTAssertGreaterThan(self.audioEngine.bassLevel, 0, "Bass level should be greater than 0")
                XCTAssertGreaterThan(self.audioEngine.midLevel, 0, "Mid level should be greater than 0")
                XCTAssertGreaterThan(self.audioEngine.trebleLevel, 0, "Treble level should be greater than 0")
                
                expectation.fulfill()
            }
            .store(in: &cancellables)
        
        // Start playback
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for audio processing to complete
        wait(for: [expectation], timeout: 5.0)
    }
    
    /// Tests system audio capture on supported platforms
    func testSystemAudioCapture() {
        #if os(macOS)
        XCTAssertNoThrow(try audioEngine.enableSystemAudioCapture(), "System audio capture should be enabled without errors on macOS")
        
        // Test system audio volume
        audioEngine.setSystemAudioVolume(0.7)
        XCTAssertEqual(audioEngine.systemAudioVolume, 0.7, accuracy: 0.01, "System audio volume should be set correctly")
        #endif
    }
    
    // MARK: - Visualization Tests
    
    /// Tests basic visualization initialization
    func testVisualizationInitialization() {
        XCTAssertNotNil(audioVisualizer, "Audio visualizer should be properly initialized")
        XCTAssertNotNil(audioVisualizer.renderer, "Metal renderer should be properly initialized")
    }
    
    /// Tests visualization theme switching
    func testVisualizationThemes() {
        // Test each theme
        for theme in VisualTheme.allCases {
            settings.currentTheme = theme
            audioVisualizer.updateTheme(theme)
            
            // Check that renderer settings match
            XCTAssertEqual(audioVisualizer.currentTheme, theme, "Current theme should match requested theme")
        }
    }
    
    /// Tests frame rendering with audio data
    func testFrameRendering() {
        // Create mock audio data
        let mockFFTData = Array(repeating: Float(0.5), count: 1024)
        let mockAudioLevels = AudioLevels(
            bass: 0.7,
            mid: 0.5,
            treble: 0.3,
            left: 0.6,
            right: 0.6
        )
        
        // Create expectation for rendering
        let expectation = XCTestExpectation(description: "Frame rendering")
        
        // Start performance measuring
        performanceMonitor.beginMeasuring("RenderFrame")
        
        // Render frame with mock data
        audioVisualizer.updateAudioData(fftData: mockFFTData, levels: mockAudioLevels)
        
        // Run rendering on a background queue to simulate real rendering
        DispatchQueue.global().async {
            let texture = self.audioVisualizer.renderFrame()
            DispatchQueue.main.async {
                // End performance measuring
                let renderTime = self.performanceMonitor.endMeasuring("RenderFrame")
                
                // Verify texture was created
                XCTAssertNotNil(texture, "Rendered texture should not be nil")
                
                // Check render performance based on
        // Format settings
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        
        // Create audio buffer
        let frameCount = AVAudioFrameCount(sampleRate * duration)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        buffer.frameLength = frameCount
        
        // Fill buffer with sine wave data
        for frame in 0..<Int(frameCount) {
            let time = Double(frame) / sampleRate
            
            // Create sine wave with beats
            let sineValue = sin(2.0 * .pi * frequency * time)
            let beatValue = abs(sin(2.0 * .pi * beatFrequency * time))
            let value = Float(sineValue * (0.5 + 0.5 * beatValue))
            
            // Set left and right channels
            buffer.floatChannelData?[0][frame] = value
            buffer.floatChannelData?[1][frame] = value
        }
        
        // Write buffer to file
        try? AVAudioFile(forWriting: testAudioURL, settings: format.settings)
            .write(from: buffer)
    }
    
    /// Creates test presets for testing
    private func createTestPresets() {
        // Create a few standard presets for testing
        let presetThemes: [(String, VisualTheme)] = [
            ("Classic Test", .classic),
            ("Neon Test", .neon),
            ("Monochrome Test", .monochrome),
            ("Cosmic Test", .cosmic)
        ]
        
        for (name, theme) in presetThemes {
            settings.currentTheme = theme
            let preset = presetManager.createPresetFromCurrentSettings(
                name: name,
                description: "Auto-generated test preset"
            )
            testPresets.append(preset)
        }
    }
ing all major components
final class AudioBloomTests: XCTestCase {
    // Test environment components
    var audioEngine: AudioEngine!
    var presetManager: PresetManager!
    var settings: AudioBloomSettings!
    var performanceMonitor: PerformanceMonitor!
    var metalDevice: MTLDevice!
    var audioVisualizer: AudioVisualizer!
    
    // Test resources
    var testAudioURL: URL!
    var testPresets: [Preset] = []
    var cancellables = Set<AnyCancellable>()
    
    // Test flags
    var isM3Hardware: Bool = false
    
    // M3 specific properties
    var aneCoreML: Bool = false // Apple Neural Engine availability
    
    override func setUp() {
        super.setUp()
        
        // Detect hardware
        let hostInfo = Host.current().localizedName ?? ""
        isM3Hardware = hostInfo.contains("M3")
        
        // Create test audio file URL
        let tempDir = NSTemporaryDirectory()
        testAudioURL = URL(fileURLWithPath: tempDir).appendingPathComponent("test_audio.wav")
        generateTestAudio()
        
        // Initialize Metal device
        metalDevice = MTLCreateSystemDefaultDevice()
        
        // Check ANE capability
        if #available(macOS 12.0, *) {
            let mlconfig = MLModelConfiguration()
            mlconfig.computeUnits = .all
            aneCoreML = true
        }
        
        // Initialize test components
        settings = AudioBloomSettings()
        performanceMonitor = PerformanceMonitor(metalDevice: metalDevice)
        audioEngine = AudioEngine(settings: settings)
        presetManager = PresetManager(settings: settings)
        audioVisualizer = AudioVisualizer(device: metalDevice, settings: settings)
        
        // Generate test presets
        createTestPresets()
    }
    
    override func tearDown() {
        // Clean up resources
        cancellables.removeAll()
        
        // Remove test audio file
        try? FileManager.default.removeItem(at: testAudioURL)
        
        // Clean up test presets
        for preset in testPresets {
            presetManager.deletePreset(preset)
        }
        
        // Reset components
        audioEngine.stop()
        audioEngine = nil
        presetManager = nil
        settings = nil
        performanceMonitor = nil
        metalDevice = nil
        audioVisualizer = nil
        
        super.tearDown()
    }
    
    // MARK: - Helper Methods
    
    /// Generates test audio for testing
    private func generateTestAudio() {
        // Create a simple sine wave audio file for testing
        let sampleRate: Double = 44100.0
        let duration: Double = 5.0  // 5 seconds
        let frequency: Double = 440.0  // A 440Hz
        let beatFrequency: Double = 2.0  // 2Hz for beat testing
        
        // Format settings
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
    var presetManager: PresetManager!
    var settings: AudioBloomSettings!
    var performanceMonitor: PerformanceMonitor!
    var metalDevice: MTLDevice!
    var audioEngine: AudioEngine!
    var presetManager: PresetManager!
    var settings: AudioBloomSettings!
    var performanceMonitor: PerformanceMonitor!
    var metalDevice: MTLDevice!
    var audioVisualizer: AudioVisualizer!
    
    // Test resources
    var testAudioURL: URL!
    var testPresets: [Preset] = []
    var cancellables = Set<AnyCancellable>()
    
    // Test flags
    var isM3Hardware: Bool = false
    
    // M3 specific properties
    var aneCoreML: Bool = false // Apple Neural Engine availability
        let hostInfo = Host.current().localizedName ?? ""
        isM3Hardware = hostInfo.contains("M3")
        
        // Create test audio file URL
        let tempDir = NSTemporaryDirectory()
        testAudioURL = URL(fileURLWithPath: tempDir).appendingPathComponent("test_audio.wav")
        generateTestAudio()
        
        // Initialize Metal device
        metalDevice = MTLCreateSystemDefaultDevice()
        
        // Check ANE capability
        if #available(macOS 12.0, *) {
            let mlconfig = MLModelConfiguration()
            mlconfig.computeUnits = .all
            aneCoreML = true
        }
        
        // Initialize test components
        settings = AudioBloomSettings()
        performanceMonitor = PerformanceMonitor(metalDevice: metalDevice)
        audioEngine = AudioEngine(settings: settings)
        presetManager = PresetManager(settings: settings)
        audioVisualizer = AudioVisualizer(device: metalDevice, settings: settings)
        
        // Generate test presets
        createTestPresets()
    }
    
    override func tearDown() {
        // Clean up resources
        cancellables.removeAll()
        
        // Remove test audio file
        try? FileManager.default.removeItem(at: testAudioURL)
        
        // Clean up test presets
        for preset in testPresets {
            presetManager.deletePreset(preset)
        }
        
        // Reset components
        audioEngine.stop()
        audioEngine = nil
        presetManager = nil
        settings = nil
        performanceMonitor = nil
        metalDevice = nil
        audioVisualizer = nil
        
        super.tearDown()
    }
    
    // MARK: - Audio Engine Tests
    
    /// Tests preset creation and saving
    func testPresetCreationAndSaving() {
        // Create a new preset from current settings
        let presetName = "Test Preset \(UUID().uuidString)"
        let presetDescription = "A test preset created by AudioBloomTests"
        
        // Modify settings to ensure they're captured
        settings.currentTheme = .neon
        settings.audioSensitivity = 0.8
        settings.motionIntensity = 0.6
        settings.neuralEngineEnabled = true
        
        // Create preset
        let preset = presetManager.createPresetFromCurrentSettings(
            name: presetName,
            description: presetDescription
        )
        
        // Add to test presets for cleanup
        testPresets.append(preset)
        
        // Verify preset was created
        XCTAssertEqual(preset.name, presetName, "Preset name should match")
        XCTAssertEqual(preset.description, presetDescription, "Preset description should match")
        XCTAssertEqual(preset.visualSettings.theme, .neon, "Preset theme should match")
        XCTAssertEqual(preset.visualSettings.sensitivity, 0.8, "Preset sensitivity should match")
        XCTAssertEqual(preset.visualSettings.motionIntensity, 0.6, "Preset motion intensity should match")
        XCTAssertTrue(preset.neuralSettings.enabled, "Preset neural settings should be enabled")
        
        // Verify the preset was added to the manager
        XCTAssertTrue(presetManager.presets.contains(where: { $0.id == preset.id }), "Preset should be in the preset manager's collection")
        XCTAssertEqual(preset.visualSettings.motionIntensity, 0.6, "Preset motion intensity should match")
        XCTAssertTrue(preset.neuralSettings.enabled, "Preset neural settings should be enabled")
        
        // Verify the preset was added to the manager
        XCTAssertTrue(presetManager.presets.contains(where: { $0.id == preset.id }), "Preset should be in the preset manager's collection")
    }
d be under 2ms on M3 Pro hardware")
        XCTAssertLessThan(totalExecutionTime, iterations * 0.004, "Total execution should be efficient on M3 Pro")
        
        // Test that the texture cache is working effectively
        performanceMonitor.beginMeasuring("TextureCacheEfficiency")
        for _ in 0..<10 {
            _ = audioVisualizer.renderFrame()
        }
        let cacheTime = performanceMonitor.endMeasuring("TextureCacheEfficiency")
        XCTAssertLessThan(cacheTime, 20.0, "Texture cache operations should be under 20ms for 10 frames on M3 Pro")
    }
    
    /// Tests M3 Pro memory optimization strategies
    func testM3ProMemoryOptimizations() {
        // Skip if not on M3 hardware
        guard isM3Hardware else {
            print("Skipping M3 memory optimization test on non-M3 hardware")
            return
        }
        
        // Test memory buffer reuse
        let bufferCount = 20
        var bufferIDs: [UUID] = []
        
        // Register a series of buffers
        for _ in 0..<bufferCount {
            let bufferID = UUID()
            bufferIDs.append(bufferID)
            performanceMonitor.registerAudioBuffer(identifier: bufferID, byteSize: 512 * 1024) // 512KB each
        }
        
        // Check initial memory allocation
        let initialMemory = performanceMonitor.audioBufferUsage
        XCTAssertGreaterThanOrEqual(initialMemory, 10.0, "Should have allocated at least 10MB")
        
        // Unregister half the buffers
        for i in 0..<bufferCount/2 {
            performanceMonitor.unregisterAudioBuffer(identifier: bufferIDs[i])
        }
        
        // Wait for memory tracking to update
        let expectation = XCTestExpectation(description: "Memory optimization")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            // Verify memory usage decreased
            let midMemory = self.performanceMonitor.audioBufferUsage
            XCTAssertLessThan(midMemory, initialMemory, "Memory usage should decrease after releasing buffers")
            
            // Create high memory pressure
            self.performanceMonitor.simulateMemoryWarning()
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                // Check if the application properly responded to memory pressure
                let recommendations = self.performanceMonitor.getOptimizationRecommendations()
                XCTAssertTrue(recommendations.keys.contains("reduceBufferSize"), "Should recommend buffer size reduction under memory pressure")
                
                // Clean up remaining buffers
                for i in bufferCount/2..<bufferCount {
                    self.performanceMonitor.unregisterAudioBuffer(identifier: bufferIDs[i])
                }
                
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 2.0)
    }
    
    /// Tests neural engine utilization on M3 Pro hardware
    func testM3ProNeuralEngineUtilization() {
        // Skip if not on M3 hardware or no ANE available
        guard isM3Hardware && aneCoreML else {
            print("Skipping neural engine utilization test on unsupported hardware")
            return
        }
        
        // Enable neural processing
        settings.neuralEngineEnabled = true
        settings.beatSensitivity = 0.8
        settings.emotionalSensitivity = 0.9
        
        // Create mock audio data that should trigger neural processing
        let mockFFTData = Array(repeating: Float(0.5), count: 1024)
        let mockAudioLevels = AudioLevels(
            bass: 0.8,
            mid: 0.7,
            treble: 0.6,
            left: 0.75,
            right: 0.75
        )
        
        // Prepare expectation for ANE utilization
        let expectation = XCTestExpectation(description: "ANE utilization")
        
        // Start performance measuring
        performanceMonitor.beginMeasuring("NeuralProcessing")
        
        // Process audio with neural engine
        audioEngine.processAudioWithNeuralEngine(fftData: mockFFTData, levels: mockAudioLevels) { neuralResults in
            // End performance measuring
            let processingTime = self.performanceMonitor.endMeasuring("NeuralProcessing")
            
            // Verify neural results
            XCTAssertNotNil(neuralResults, "Neural results should not be nil")
            XCTAssertGreaterThan(neuralResults?.beatIntensity ?? 0, 0, "Beat intensity should be detected")
            
            // Check processing time on M3 Pro (should be very fast due to ANE)
            XCTAssertLessThan(processingTime, 10.0, "Neural processing should be under 10ms on M3 Pro with ANE")
            
            // Check that we're getting proper ANE utilization
            XCTAssertTrue(self.performanceMonitor.isUsingANE, "Should be using the Apple Neural Engine")
            
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
CTAssertFalse(audioEngine.isRunning, "Audio engine should not be running after stop")
        } catch {
            XCTFail("Audio engine start failed with error: \(error)")
        }
    }
    
    /// Tests preset updating
    func testPresetUpdate() {
        // Create a preset first
        let preset = presetManager.createPresetFromCurrentSettings(
            name: "Update Test Preset",
            description: "Will be updated"
        )
        testPresets.append(preset)
        
        // Modify the preset
        var updatedPreset = preset
        updatedPreset.name = "Updated Name"
        updatedPreset.description = "Updated description"
        updatedPreset.visualSettings.theme = .cosmic
        updatedPreset.visualSettings.sensitivity = 0.9
        
        // Update the preset in the manager
        presetManager.updatePreset(presetId: preset.id, updatedPreset: updatedPreset)
        
        // Get the updated preset
        guard let retrievedPreset = presetManager.presets.first(where: { $0.id == preset.id }) else {
            XCTFail("Updated preset should exist in the preset manager")
            return
        }
        
        // Verify the updates
        XCTAssertEqual(retrievedPreset.name, "Updated Name", "Preset name should be updated")
        XCTAssertEqual(retrievedPreset.description, "Updated description", "Preset description should be updated")
        XCTAssertEqual(retrievedPreset.visualSettings.theme, .cosmic, "Preset theme should be updated")
        XCTAssertEqual(retrievedPreset.visualSettings.sensitivity, 0.9, "Preset sensitivity should be updated")
    }
    
    /// Tests preset loading and applying
    func testPresetApplying() {
        // Create a preset with specific settings
        let preset = Preset(
            name: "Apply Test Preset",
            description: "For testing preset application",
            visualSettings: VisualizationSettings(
                theme: .monochrome,
                sensitivity: 0.75,
                motionIntensity: 0.6,
                showFPS: true,
                showBeatIndicator: false
            ),
            audioSettings: AudioSettings(
                inputDevice: nil,
                outputDevice: nil,
                audioSource: "Microphone",
                micVolume: 0.8,
                systemAudioVolume: 0.0,
                mixInputs: false
            ),
            neuralSettings: NeuralSettings(
                enabled: false,
                beatSensitivity: 0.5,
                patternSensitivity: 0.5,
                emotionalSensitivity: 0.5
            )
        )
        
        // Add to presetManager
        presetManager.addPreset(preset)
        testPresets.append(preset)
        
        // Apply the preset
        presetManager.applyPreset(preset)
        
        // Verify settings were applied
        XCTAssertEqual(settings.currentTheme, .monochrome, "Theme should be applied")
        XCTAssertEqual(settings.audioSensitivity, 0.75, "Sensitivity should be applied")
        XCTAssertEqual(settings.motionIntensity, 0.6, "Motion intensity should be applied")
        XCTAssertFalse(settings.neuralEngineEnabled, "Neural engine setting should be applied")
        
        // Verify current preset is set
        XCTAssertEqual(presetManager.currentPreset?.id, preset.id, "Current preset should be set")
    }
    
    /// Tests preset export and import
    func testPresetExportImport() {
        // Create a preset to export
        let exportPreset = presetManager.createPresetFromCurrentSettings(
            name: "Export Test Preset",
            description: "For testing preset export"
        )
        testPresets.append(exportPreset)
        
        // Export the preset
        let exportURL = presetManager.exportPreset(exportPreset)
        XCTAssertNotNil(exportURL, "Export URL should not be nil")
        
        // Delete the preset
        presetManager.deletePreset(exportPreset)
        XCTAssertFalse(presetManager.presets.contains(where: { $0.id == exportPreset.id }), "Preset should be deleted")
        
        // Import the preset back
        guard let exportURL = exportURL else { return }
        let importedPreset = presetManager.importPreset(from: exportURL)
        XCTAssertNotNil(importedPreset, "Imported preset should not be nil")
        
        // Add to test presets for cleanup if needed
        if let importedPreset = importedPreset {
            testPresets.append(importedPreset)
            
            // Verify imported preset matches original
            XCTAssertEqual(importedPreset.name, exportPreset.name, "Imported preset name should match")
            XCTAssertEqual(importedPreset.description, exportPreset.description, "Imported preset description should match")
            XCTAssertEqual(importedPreset.visualSettings.theme, exportPreset.visualSettings.theme, "Imported preset theme should match")
        }
        
        // Clean up export file
        try? FileManager.default.removeItem(at: exportURL)
    }
    
    /// Tests preset categorization
    func testPresetCategorization() {
        // Create presets with different themes for categorization
        let presetClassic = presetManager.createPresetFromCurrentSettings(
            name: "Classic Theme Preset",
            description: "Uses classic theme"
        )
        settings.currentTheme = .classic
        testPresets.append(presetClassic)
        
        let presetNeon = presetManager.createPresetFromCurrentSettings(
            name: "Neon Theme Preset",
            description: "Uses neon theme"
        )
        settings.currentTheme = .neon
        testPresets.append(presetNeon)
        
        // Get preset categories
        let categories = presetManager.getPresetsByCategory()
        
        // Verify categorization
        XCTAssertTrue(categories.keys.contains("Classic"), "Should have Classic category")
        XCTAssertTrue(categories.keys.contains("Neon"), "Should have Neon category")
        
        if let classicPresets = categories["Classic"] {
            XCTAssertTrue(classicPresets.contains(where: { $0.id == presetClassic.id }), "Classic preset should be in Classic category")
        } else {
            XCTFail("Classic category should exist")
        }
        
        if let neonPresets = categories["Neon"] {
            XCTAssertTrue(neonPresets.contains(where: { $0.id == presetNeon.id }), "Neon preset should be in Neon category")
        } else {
            XCTFail("Neon category should exist")
        }
    }
    
    // MARK: - Performance Monitor Tests
    
    /// Tests performance monitoring initialization
    func testPerformanceMonitorInitialization() {
        XCTAssertNotNil(performanceMonitor, "Performance monitor should be properly initialized")
        XCTAssertGreaterThanOrEqual(performanceMonitor.fps, 0, "FPS should be initialized to a valid value")
    }
    
    /// Tests performance metrics tracking
    func testPerformanceMetricsTracking() {
        // Create expectation for metrics collection
        let expectation = XCTestExpectation(description: "Performance metrics tracking")
        
        // Simulate rendering to generate metrics
        let testDuration = 3.0 // 3 seconds
        let startTime = Date()
        
        // Run a timer to check for metrics
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { timer in
            if Date().timeIntervalSince(startTime) >= testDuration {
                timer.invalidate()
                expectation.fulfill()
            }
            
            // Force some performance metrics by doing work
            self.performanceMonitor.beginMeasuring("TestOperation")
            // Simulate work by rendering a frame
            _ = self.audioVisualizer.renderFrame()
            self.performanceMonitor.endMeasuring("TestOperation")
        }
        
        // Wait for metrics to be collected
        wait(for: [expectation], timeout: testDuration + 1.0)
        
        // Verify metrics were collected
        XCTAssertGreaterThan(performanceMonitor.fps, 0, "FPS should be tracked")
        XCTAssertGreaterThan(performanceMonitor.cpuUsage, 0, "CPU usage should be tracked")
        XCTAssertGreaterThan(performanceMonitor.memoryUsage, 0, "Memory usage should be tracked")
    }
    
    /// Tests quality level settings and adjustments
    func testQualityLevelAdjustment() {
        // Set initial quality level
        performanceMonitor.setQualityLevel(.high)
        XCTAssertEqual(performanceMonitor.currentQualityLevel, .high, "Quality level should be set to high")
        
        // Attempt to set too high (above max)
        performanceMonitor.maxQualityLevel = .medium
        performanceMonitor.setQualityLevel(.ultra)
        XCTAssertEqual(performanceMonitor.currentQualityLevel, .medium, "Quality level should be limited by max level")
        
        // Reset max quality
        performanceMonitor.maxQualityLevel = .ultra
        
        // Test quality level change publisher
        let expectation = XCTestExpectation(description: "Quality level change")
        
        performanceMonitor.qualityLevelDidChange
            .sink { level in
                XCTAssertEqual(level, .low, "Published quality level should match set level")
                expectation.fulfill()
            }
            .store(in: &cancellables)
        
        performanceMonitor.setQualityLevel(.low)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    /// Tests audio buffer memory tracking
    func testAudioBufferMemoryTracking() {
        // Register some test buffers
        let buffer1ID = UUID()
        let buffer2ID = UUID()
        
        performanceMonitor.registerAudioBuffer(identifier: buffer1ID, byteSize: 1024 * 1024) // 1MB
        performanceMonitor.registerAudioBuffer(identifier: buffer2ID, byteSize: 2 * 1024 * 1024) // 2MB
        
        // Wait a bit for async operations
        let expectation = XCTestExpectation(description: "Audio buffer memory tracking")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            // Verify total registered memory
            XCTAssertGreaterThanOrEqual(self.performanceMonitor.audioBufferUsage, 3.0, "Should track at least 3MB of audio buffer usage")
            
            // Unregister one buffer
            self.performanceMonitor.unregisterAudioBuffer(identifier: buffer1ID)
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                // Verify decreased memory usage
                XCTAssertLessThan(self.performanceMonitor.audioBufferUsage, 3.0, "Audio buffer usage should decrease after unregistering")
                XCTAssertGreaterThanOrEqual(self.performanceMonitor.audioBufferUsage, 2.0, "Should still track remaining buffer memory")
                
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 2.0)
    }
    
    // MARK: - M3 Pro Specific Performance Tests
    
    /// Tests M3 Pro specific optimizations with Metal shader performance
    func testM3ProMetalOptimizations() {
        // Skip if not on M3 hardware
        guard isM3Hardware else {
            print("Skipping M3 specific test on non-M3 hardware")
            return
        }
        
        // Test shader performance on M3 Pro
        let iterations = 100
        var totalTime: Double = 0
        
        // Start performance measurement
        let start = CACurrentMediaTime()
        
        // Run multiple render iterations to test shader performance
        for _ in 0..<iterations {
            performanceMonitor.beginMeasuring("M3Rendering")
            _ = audioVisualizer.renderFrame()
            let frameTime = performanceMonitor.endMeasuring("M3Rendering")
            totalTime += frameTime
        }
        
        let end = CACurrentMediaTime()
        let totalExecutionTime = end - start
        
        // Calculate average render time
        let avgRenderTime = totalTime / Double(iterations)
        
        // M3 Pro shoul
            XCTAssertEqual(audioEngine.inputVolume, 0.5, accuracy: 0.01, "Input volume should be set correctly")
            
            // Stop the engine
            audioEngine.stop()
            XCTAssertFalse(audioEngine.isRunning, "Audio engine should not be running after stop")
        } catch {
            XCTFail("Audio engine start failed with error: \(error)")
        }
    }
    
    /// Tests audio processing and FFT analysis
    func testAudioProcessing() {
        // Configure for file playback
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Create expectation for audio processing
        let expectation = XCTestExpectation(description: "Audio processing")
        
        // Subscribe to FFT updates
        audioEngine.fftPublisher
            .sink { fftData in
                // Verify FFT data format and contents
                XCTAssertEqual(fftData.count, 1024, "FFT data should have 1024 frequency bins")
                XCTAssertFalse(fftData.allSatisfy { $0 == 0 }, "FFT data should not be all zeros")
                
                // Verify calculated audio levels
                XCTAssertGreaterThan(self.audioEngine.bassLevel, 0, "Bass level should be greater than 0")
                XCTAssertGreaterThan(self.audioEngine.midLevel, 0, "Mid level should be greater than 0")
                XCTAssertGreaterThan(self.audioEngine.trebleLevel, 0, "Treble level should be greater than 0")
                
                expectation.fulfill()
            }
            .store(in: &cancellables)
        
        // Start playback
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for audio processing to complete
        wait(for: [expectation], timeout: 5.0)
    }
    
    /// Tests system audio capture on supported platforms
    func testSystemAudioCapture() {
        #if os(macOS)
        XCTAssertNoThrow(try audioEngine.enableSystemAudioCapture(), "System audio capture should be enabled without errors on macOS")
        
        // Test system audio volume
        audioEngine.setSystemAudioVolume(0.7)
        XCTAssertEqual(audioEngine.systemAudioVolume, 0.7, accuracy: 0.01, "System audio volume should be set correctly")
        #endif
    }
    
    // MARK: - Visualization Tests
    
    /// Tests basic visualization initialization
    func testVisualizationInitialization() {
        XCTAssertNotNil(audioVisualizer, "Audio visualizer should be properly initialized")
        XCTAssertNotNil(audioVisualizer.renderer, "Metal renderer should be properly initialized")
    }
    
    /// Tests visualization theme switching
    func testVisualizationThemes() {
        // Test each theme
        for theme in VisualTheme.allCases {
            settings.currentTheme = theme
            audioVisualizer.updateTheme(theme)
            
            // Check that renderer settings match
            XCTAssertEqual(audioVisualizer.currentTheme, theme, "Current theme should match requested theme")
        }
    }
    
    /// Tests frame rendering with audio data
    func testFrameRendering() {
        // Create mock audio data
        let mockFFTData = Array(repeating: Float(0.5), count: 1024)
        let mockAudioLevels = AudioLevels(
            bass: 0.7,
            mid: 0.5,
            treble: 0.3,
            left: 0.6,
            right: 0.6
        )
        
        // Create expectation for rendering
        let expectation = XCTestExpectation(description: "Frame rendering")
        
        // Start performance measuring
        performanceMonitor.beginMeasuring("RenderFrame")
        
        // Render frame with mock data
        audioVisualizer.updateAudioData(fftData: mockFFTData, levels: mockAudioLevels)
        
        // Run rendering on a background queue to simulate real rendering
        DispatchQueue.global().async {
            let texture = self.audioVisualizer.renderFrame()
            DispatchQueue.main.async {
                // End performance measuring
                let renderTime = self.performanceMonitor.endMeasuring("RenderFrame")
                
                // Verify texture was created
                XCTAssertNotNil(texture, "Rendered texture should not be nil")
                
                // Check render performance based on hardware
                if self.isM3Hardware {
                    XCTAssertLessThan(renderTime, 5.0, "Render time should be under 5ms on M3 hardware")
                } else {
                    XCTAssertLessThan(renderTime, 16.0, "Render time should be under 16ms on all hardware")
                }
                
                expectation.fulfill()
            }
        }
        
        // Wait for rendering to complete
        wait(for: [expectation], timeout: 5.0)
    }
    
    /// Tests shader compilation for all themes
    func testShaderCompilation() {
        // Create expectation for shader compilation
        let expectation = XCTestExpectation(description: "Shader compilation")
        
        // Start performance measuring
        performanceMonitor.beginMeasuring("ShaderCompilation")
        
        // Force shader recompilation by setting a different theme
        audioVisualizer.updateTheme(.neon) // Use neon theme to stress the shader more
        
        DispatchQueue.global().async {
            // Create a custom render pipeline to force shader compilation
            let library = self.audioVisualizer.renderer.metalLibrary
            
            // Try to get vertex and fragment functions
            let vertexFunction = library?.makeFunction(name: "audio_vertex_shader")
            let fragmentFunction = library?.makeFunction(name: "audio_fragment_shader")
            
            // Check if functions were created successfully
            XCTAssertNotNil(vertexFunction, "Vertex shader function should be created")
            XCTAssertNotNil(fragmentFunction, "Fragment shader function should be created")
            
            // End performance measuring
            let compileTime = self.performanceMonitor.endMeasuring("ShaderCompilation")
            
            // Check compile performance based on hardware
            if self.isM3Hardware {
                XCTAssertLessThan(compileTime, 100.0, "Shader compile time should be under 100ms on M3 hardware")
            } else {
                XCTAssertLessThan(compileTime, 500.0, "Shader compile time should be under 500ms on all hardware")
            }
            
            expectation.fulfill()
        }
        
        // Wait for shader compilation to complete
        wait(for: [expectation], timeout: 10.0)
    }
    
    // MARK: - Neural Engine Tests
    
    /// Tests neural engine integration for beat detection
    func testNeuralBeatDetection() {
        guard aneCoreML else {
            print("Skipping neural engine test on unsupported hardware")
            return
        }
        
        // Enable neural engine
        settings.neuralEngineEnabled = true
        settings.beatSensitivity = 0.8
        
        // Create expectation for neural processing
        let expectation = XCTestExpectation(description: "Neural beat detection")
        
        // Set up audio with known beats
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Start monitoring neural data
        var beatDetected = false
        
        // Set up listener for neural engine beat detection
        NotificationCenter.default.addObserver(forName: Notification.Name("BeatDetected"), object: nil, queue: nil) { _ in
            beatDetected = true
            expectation.fulfill()
        }
        
        // Start playback (assuming test audio has beats)
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for neural engine to detect beats
        wait(for: [expectation], timeout: 10.0)
        
        // Verify beat detection
        XCTAssertTrue(beatDetected, "Neural engine should detect beats in test audio")
    }
    
    /// Tests emotional analysis on M3 Pro hardware
    func testEmotionalAnalysis() {
        // Skip test if not on M3 hardware or no ANE available
        guard isM3Hardware && aneCoreML else {
            print("Skipping emotional analysis test on unsupported hardware")
            return
        }
        
        // Enable neural engine with focus on emotional analysis
        settings.neuralEngineEnabled = true
        settings.emotionalSensitivity = 1.0
        
        // Create expectation for emotional analysis
        let expectation = XCTestExpectation(description: "Emotional analysis")
        
        // Set up audio with known emotional content
        audioEngine.loadAudioFile(url: testAudioURL)
        
        // Start monitoring for emotional data
        var emotionalData: [String: Float] = [:]
        
        // Set up listener for emotional analysis
        NotificationCenter.default.addObserver(forName: Notification.Name("EmotionalAnalysisComplete"), object: nil, queue: nil) { notification in
            if let data = notification.userInfo as? [String: Float] {
                emotionalData = data
                expectation.fulfill()
            }
        }
        
        // Start playback
        do {
            try audioEngine.start()
            audioEngine.play()
        } catch {
            XCTFail("Failed to start audio engine: \(error)")
        }
        
        // Wait for emotional analysis to complete
        wait(for: [expectation], timeout: 15.0)
        
        // Verify emotional data
        XCTAssertFalse(emotionalData.isEmpty, "Emotional data should not be empty")
        XCTAssertNotNil(emotionalData["valence"], "Valence should be present in emotional data")
        XCTAssertNotNil(emotionalData["arousal"], "Arousal should be present in emotional data")
    }
    
    // MARK: - Preset Management Tests
    
    /// Tests preset creation and saving
    func testPresetCreationAndSaving() {
        // Create a new preset from current settings
        let presetName = "Test Preset \(UUID().uuidString)"
        let presetDescription = "A test preset created by AudioBloomTests"
        
        // Modify settings to ensure they're captured
        settings.currentTheme = .neon
        settings.audioSensitivity = 0.8
        settings.motionIntensity = 0.6
        settings.neuralEngineEnabled = true
        
        // Create preset
        let preset = presetManager.createPresetFromCurrentSettings(
            name: presetName,
            description: presetDescription
        )
        
        // Add to test presets for cleanup
        testPresets.append(preset)
        
        // Verify preset was created
        XCTAssertEqual(preset.name, presetName, "Preset name should match")
        XCTAssertEqual(preset.description, presetDescription, "Preset description should match")
        XCTAssertEqual(preset.visualSettings.theme, .neon, "Preset theme should match")
        XCT

import XCTest
@testable import AudioBloomCore

final class AudioBloomTests: XCTestCase {
    func testExample() throws {
        // This is a placeholder test case
        XCTAssert(true)
    }
    
    static var allTests = [
        ("testExample", testExample),
    ]
}

