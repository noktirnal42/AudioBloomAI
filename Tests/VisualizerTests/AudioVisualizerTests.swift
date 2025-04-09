import XCTest
import Combine
@testable import Visualizer
@testable import AudioBloomCore

// MARK: - Mock Objects

/// Mock audio data provider for testing
class MockAudioEngine: AudioDataProvider {
    var setupCalled = false
    var startCalled = false
    var stopCalled = false
    var shouldThrowError = false
    var lastAudioData: [Float] = []
    var lastLevels: (left: Float, right: Float) = (0, 0)
    
    // Publisher for audio data
    private let audioDataPublisher = CurrentValueSubject<(audioData: [Float], levels: (left: Float, right: Float)), Never>(([], (0, 0)))
    
    func setupAudioSession() async throws {
        setupCalled = true
        if shouldThrowError {
            throw VisualizerError.engineInitFailure
        }
    }
    
    func startCapture() throws {
        startCalled = true
        if shouldThrowError {
            throw VisualizerError.engineInitFailure
        }
    }
    
    func stopCapture() {
        stopCalled = true
    }
    
    // Simulate sending audio data
    func sendMockAudioData(_ data: [Float], levels: (left: Float, right: Float)) {
        lastAudioData = data
        lastLevels = levels
        audioDataPublisher.send((data, levels))
    }
    
    func getAudioDataPublisher() -> CurrentValueSubject<(audioData: [Float], levels: (left: Float, right: Float)), Never> {
        return audioDataPublisher
    }
}

/// Mock neural engine for testing
class MockNeuralEngine: MLProcessing {
    var prepareModelCalled = false
    var processDataCalled = false
    var lastAudioData: [Float] = []
    var shouldThrowError = false
    var beatDetected = false
    var outputData: [Float] = Array(repeating: 0, count: 32)
    
    // For ObservableObject compliance
    let objectWillChange = PassthroughSubject<Void, Never>()
    
    func prepareMLModel() {
        prepareModelCalled = true
    }
    
    func processAudioData(_ data: [Float]) async {
        processDataCalled = true
        lastAudioData = data
        
        // Simulate some ML processing
        beatDetected = data.contains { $0 > 0.8 }
        
        // Generate random output data
        for i in 0..<outputData.count {
            outputData[i] = Float.random(in: 0...1)
        }
        
        // Notify changes
        DispatchQueue.main.async {
            self.objectWillChange.send()
        }
    }
}

/// Mock visualization renderer for testing
class MockVisualizationRenderer: VisualizationRenderer, VisualizationParameterReceiver {
    var isReady = true
    var prepareCalled = false
    var updateCalled = false
    var renderCalled = false
    var cleanupCalled = false
    var lastParameters: [String: Any] = [:]
    var lastAudioData: [Float] = []
    var lastLevels: (left: Float, right: Float) = (0, 0)
    var shouldThrowError = false
    
    // For tracking updates
    var updateCount = 0
    var renderCount = 0
    
    func prepareRenderer() {
        prepareCalled = true
        if shouldThrowError {
            isReady = false
        } else {
            isReady = true
        }
    }
    
    func update(audioData: [Float], levels: (left: Float, right: Float)) {
        updateCalled = true
        updateCount += 1
        lastAudioData = audioData
        lastLevels = levels
    }
    
    func render() {
        renderCalled = true
        renderCount += 1
    }
    
    func updateParameters(_ parameters: [String: Any]) {
        lastParameters = parameters
    }
    
    func cleanup() {
        cleanupCalled = true
    }
}

// MARK: - Tests

final class AudioVisualizerTests: XCTestCase {
    var audioEngine: MockAudioEngine!
    var visualRenderer: MockVisualizationRenderer!
    var neuralEngine: MockNeuralEngine!
    var settings: AudioBloomSettings!
    var cancellables = Set<AnyCancellable>()
    
    override func setUp() {
        super.setUp()
        audioEngine = MockAudioEngine()
        visualRenderer = MockVisualizationRenderer()
        neuralEngine = MockNeuralEngine()
        settings = AudioBloomSettings()
        cancellables = Set<AnyCancellable>()
    }
    
    override func tearDown() {
        audioEngine = nil
        visualRenderer = nil
        neuralEngine = nil
        settings = nil
        cancellables.removeAll()
        super.tearDown()
    }
    
    // MARK: - Test Initialization and Setup
    
    func testInitialization() {
        // Test basic initialization
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Verify initial state
        XCTAssertFalse(visualizer.isActive)
        XCTAssertEqual(visualizer.currentTheme, settings.currentTheme)
        XCTAssertEqual(visualizer.audioSensitivity, settings.audioSensitivity)
        XCTAssertEqual(visualizer.motionIntensity, settings.motionIntensity)
        XCTAssertEqual(visualizer.neuralEngineEnabled, settings.neuralEngineEnabled)
        XCTAssertFalse(visualizer.beatDetected)
    }
    
    func testStartAndStop() async {
        // Test starting and stopping visualization
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Start visualization
        do {
            try await visualizer.start()
            
            // Verify start actions were called
            XCTAssertTrue(audioEngine.setupCalled)
            XCTAssertTrue(audioEngine.startCalled)
            XCTAssertTrue(visualRenderer.prepareCalled)
            XCTAssertTrue(neuralEngine.prepareModelCalled)
            XCTAssertTrue(visualizer.isActive)
            
            // Stop visualization
            visualizer.stop()
            
            // Verify stop actions were called
            XCTAssertTrue(audioEngine.stopCalled)
            XCTAssertFalse(visualizer.isActive)
        } catch {
            XCTFail("Start should not throw error: \(error)")
        }
    }
    
    func testStartWithError() async {
        // Test error handling during start
        audioEngine.shouldThrowError = true
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Start should throw error
        do {
            try await visualizer.start()
            XCTFail("Start should throw error")
        } catch {
            // Verify error was thrown
            XCTAssertFalse(visualizer.isActive)
            XCTAssertTrue(error is VisualizerError)
        }
    }
    
    // MARK: - Test Theme Switching and Parameter Updates
    
    func testThemeSwitching() {
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Test setting theme directly
        visualizer.currentTheme = .neon
        XCTAssertEqual(visualizer.currentTheme, .neon)
        XCTAssertEqual(settings.currentTheme, .neon)
        XCTAssertTrue(visualRenderer.lastParameters.keys.contains("theme"))
        XCTAssertEqual(visualRenderer.lastParameters["theme"] as? String, "Neon")
        
        // Test cycling through themes
        let originalThemeCount = AudioBloomCore.VisualTheme.allCases.count
        
        // Cycle through all themes
        for _ in 0..<originalThemeCount {
            let currentTheme = visualizer.currentTheme
            visualizer.cycleToNextTheme()
            
            // Verify theme was changed
            if AudioBloomCore.VisualTheme.allCases.count > 1 {
                XCTAssertNotEqual(visualizer.currentTheme, currentTheme)
            }
            
            // Verify renderer was updated
            XCTAssertEqual(visualRenderer.lastParameters["theme"] as? String, visualizer.currentTheme.rawValue)
        }
        
        // After cycling through all themes, we should be back to the first one
        XCTAssertEqual(visualizer.currentTheme, AudioBloomCore.VisualTheme.allCases.first)
    }
    
    func testParameterUpdates() {
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Test updating sensitivity
        visualizer.audioSensitivity = 0.9
        XCTAssertEqual(visualizer.audioSensitivity, 0.9)
        XCTAssertEqual(settings.audioSensitivity, 0.9)
        XCTAssertEqual(visualRenderer.lastParameters["sensitivity"] as? Float, 0.9)
        
        // Test updating motion intensity
        visualizer.motionIntensity = 0.6
        XCTAssertEqual(visualizer.motionIntensity, 0.6)
        XCTAssertEqual(settings.motionIntensity, 0.6)
        XCTAssertEqual(visualRenderer.lastParameters["motionIntensity"] as? Float, 0.6)
        
        // Test updating neural engine enabled
        visualizer.neuralEngineEnabled = !visualizer.neuralEngineEnabled
        XCTAssertEqual(visualizer.neuralEngineEnabled, settings.neuralEngineEnabled)
        XCTAssertEqual(visualRenderer.lastParameters["neuralEngineEnabled"] as? Bool, visualizer.neuralEngineEnabled)
    }
    
    // MARK: - Test Audio Data Processing and Visualization Updates
    
    func testAudioDataProcessing() {
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Set up expectations for async processing
        let expectation = XCTestExpectation(description: "Audio data processed")
        
        // Subscribe to frame rate updates to know when processing is complete
        visualizer.$framesPerSecond
            .dropFirst() // Skip initial value
            .sink { _ in
                expectation.fulfill()
            }
            .store(in: &cancellables)
        
        // Send mock audio data
        let mockData: [Float] = Array(repeating: 0, count: 1024).enumerated().map { i, _ in
            return sin(Float(i) / 100.0) * 0.5 + 0.5
        }
        let mockLevels: (left: Float, right: Float) = (0.7, 0.8)
        
        // Simulate audio data coming in
        audioEngine.sendMockAudioData(mockData, levels: mockLevels)
        
        // Wait for processing to complete
        wait(for: [expectation], timeout: 2.0)
        
        // Verify data was processed and visualization was updated
        XCTAssertTrue(visualRenderer.updateCalled)
        XCTAssertEqual(visualRenderer.lastAudioData, mockData)
        XCTAssertEqual(visualRenderer.lastLevels.left, mockLevels.left)
        XCTAssertEqual(visualRenderer.lastLevels.right, mockLevels.right)
        XCTAssertTrue(visualRenderer.renderCalled)
        
        // If neural engine is enabled, verify it was also called
        if visualizer.neuralEngineEnabled {
            XCTAssertTrue(neuralEngine.processDataCalled)
            XCTAssertEqual(neuralEngine.lastAudioData, mockData)
        }
    }
    
    func testBeatDetection() {
        // Enable neural engine
        settings.neuralEngineEnabled = true
        
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Set up expectation for beat detection
        let expectation = XCTestExpectation(description: "Beat detected")
        
        // Track beat detection changes
        visualizer.$beatDetected
            .dropFirst() // Skip initial value
            .sink { beatDetected in
                if beatDetected {
                    expectation.fulfill()
                }
            }
            .store(in: &cancellables)
        
        // Create data with peak to trigger beat detection
        var mockData = Array(repeating: Float(0.5), count: 1024)
        mockData[100] = 0.9 // Add a peak to trigger beat detection
        
        // Send the data to audio engine
        audioEngine.sendMockAudioData(mockData, levels: (0.7, 0.8))
        
        // Simulate neural engine detecting the beat
        neuralEngine.beatDetected = true
        neuralEngine.objectWillChange.send()
        
        // Wait for beat detection
        wait(for: [expectation], timeout: 2.0)
        
        // Verify beat was detected
        XCTAssertTrue(visualizer.beatDetected)
    }
    
    // MARK: - Test Settings Synchronization
    
    func testSettingsSynchronization() {
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Change settings directly
        settings.currentTheme = .cosmic
        settings.audioSensitivity = 0.65
        settings.motionIntensity = 0.45
        
        // Allow time for settings to propagate
        let expectation = XCTestExpectation(description: "Settings synchronized")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
        
        // Verify visualizer settings were updated
        XCTAssertEqual(visualizer.currentTheme, .cosmic)
        XCTAssertEqual(visualizer.audioSensitivity, 0.65)
        XCTAssertEqual(visualizer.motionIntensity, 0.45)
        
        // Verify renderer received updated parameters
        XCTAssertEqual(visualRenderer.lastParameters["theme"] as? String, "Cosmic")
        XCTAssertEqual(visualRenderer.lastParameters["sensitivity"] as? Float, 0.65)
        XCTAssertEqual(visualRenderer.lastParameters["motionIntensity"] as? Float, 0.45)
    }
    
    // MARK: - Test Performance and Frame Rate Monitoring
    
    /// Tests the frame rate monitoring functionality
    func testFrameRateMonitoring() {
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Set expectation for frame rate updates
        let expectation = XCTestExpectation(description: "Frame rate updated")
        
        // Start with zero FPS
        XCTAssertEqual(visualizer.framesPerSecond, 0.0)
        
        // Subscribe to FPS updates
        visualizer.$framesPerSecond
            .dropFirst() // Skip initial value
            .sink { fps in
                if fps > 0 {
                    expectation.fulfill()
                }
            }
            .store(in: &cancellables)
        
        // Start visualizer
        Task {
            do {
                try await visualizer.start()
                
                // Simulate multiple renders for frame rate calculation
                for _ in 0..<30 {
                    // Send audio data and trigger render
                    let mockData = Array(repeating: Float(0.5), count: 1024)
                    audioEngine.sendMockAudioData(mockData, levels: (0.5, 0.5))
                    
                    // Small delay to simulate real-time rendering
                    try await Task.sleep(nanoseconds: 16_000_000) // ~60 FPS
                }
            } catch {
                XCTFail("Failed to start visualizer: \(error)")
            }
        }
        
        // Wait for frame rate updates
        wait(for: [expectation], timeout: 2.0)
        
        // Verify frame rate is being calculated
        XCTAssertGreaterThan(visualizer.framesPerSecond, 0.0)
    }
    
    /// Tests performance under high load conditions
    func testPerformanceUnderLoad() {
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Start visualizer
        Task {
            do {
                try await visualizer.start()
            } catch {
                XCTFail("Failed to start visualizer: \(error)")
            }
        }
        
        // Use XCTest performance metrics
        measure {
            // Simulate high load by sending 100 audio updates in quick succession
            for i in 0..<100 {
                // Create dynamic audio data
                let mockData = Array(repeating: Float(0), count: 1024).enumerated().map { index, _ in
                    return sin(Float(index + i) / 50.0) * 0.5 + 0.5
                }
                
                // Send the data
                audioEngine.sendMockAudioData(mockData, levels: (0.6, 0.7))
            }
        }
        
        // Verify renderer was called appropriate number of times
        XCTAssertGreaterThanOrEqual(visualRenderer.updateCount, 100)
    }
    
    // MARK: - Test Cleanup and Resource Management
    
    /// Tests that resources are properly cleaned up
    func testCleanupAndResourceManagement() {
        // Create a scope to control visualizer lifetime
        var visualizer: AudioVisualizer? = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Start visualizer
        Task {
            do {
                try await visualizer?.start()
            } catch {
                XCTFail("Failed to start visualizer: \(error)")
            }
        }
        
        // Send some data
        let mockData = Array(repeating: Float(0.5), count: 1024)
        audioEngine.sendMockAudioData(mockData, levels: (0.5, 0.5))
        
        // Stop explicitly to trigger cleanup
        visualizer?.stop()
        
        // Verify resources were released
        XCTAssertTrue(audioEngine.stopCalled)
        
        // Set visualizer to nil to trigger deinit
        visualizer = nil
        
        // Give time for deinit to complete
        let expectation = XCTestExpectation(description: "Deinitialization completed")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
    }
    
    /// Tests that the renderer is properly cleaned up
    func testRendererCleanup() {
        // Create a temporary visualizer
        var visualizer: AudioVisualizer? = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Start and then explicitly stop
        Task {
            do {
                try await visualizer?.start()
                visualizer?.stop()
            } catch {
                XCTFail("Failed to start visualizer: \(error)")
            }
        }
        
        // Release visualizer
        visualizer = nil
        
        // Wait for cleanup to occur
        let expectation = XCTestExpectation(description: "Cleanup completed")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
        
        // Verify renderer's cleanup was called
        XCTAssertTrue(visualRenderer.cleanupCalled)
    }
    
    // MARK: - Memory Leak Detection Tests
    
    /// Tests that the visualizer doesn't cause memory leaks
    func testNoMemoryLeaks() {
        // Use weak reference to detect potential leaks
        weak var weakVisualizer: AudioVisualizer?
        
        // Create a scope for controlled lifetime
        autoreleasepool {
            let visualizer = AudioVisualizer(
                audioEngine: audioEngine,
                visualizer: visualRenderer,
                neuralEngine: neuralEngine,
                settings: settings
            )
            
            weakVisualizer = visualizer
            
            // Verify the weak reference exists
            XCTAssertNotNil(weakVisualizer)
        }
        
        // Give time for autorelease
        let expectation = XCTestExpectation(description: "Autorelease completed")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
        
        // The weak reference should now be nil if there are no retain cycles
        XCTAssertNil(weakVisualizer, "Memory leak detected: AudioVisualizer instance was not deallocated")
    }
    
    /// Tests that subscriptions are properly cleaned up to avoid memory leaks
    func testSubscriptionCleanup() {
        // Use weak reference to check for proper cleanup
        weak var weakVisualizer: AudioVisualizer?
        
        // Create a scope for controlled lifetime
        autoreleasepool {
            let visualizer = AudioVisualizer(
                audioEngine: audioEngine,
                visualizer: visualRenderer,
                neuralEngine: neuralEngine,
                settings: settings
            )
            
            weakVisualizer = visualizer
            
            // Start to create subscriptions
            Task {
                do {
                    try await visualizer.start()
                } catch {
                    XCTFail("Failed to start visualizer: \(error)")
                }
            }
            
            // Send data to trigger subscription activity
            audioEngine.sendMockAudioData(Array(repeating: 0.5, count: 1024), levels: (0.5, 0.5))
            
            // Stop to cleanup subscriptions
            visualizer.stop()
        }
        
        // Give time for autorelease
        let expectation = XCTestExpectation(description: "Autorelease completed")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
        
        // Verify that visualizer was properly deallocated
        XCTAssertNil(weakVisualizer, "Memory leak detected: AudioVisualizer instance with subscriptions was not deallocated")
    }
    
    // MARK: - Error Handling Tests
    
    /// Tests handling of renderer initialization failure
    func testRendererInitializationFailure() {
        // Set renderer to fail
        visualRenderer.shouldThrowError = true
        
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Attempt to start
        Task {
            do {
                try await visualizer.start()
                XCTFail("Should have thrown an error")
            } catch {
                // Verify error is of correct type
                XCTAssertTrue(error is VisualizerError)
                if let vizError = error as? VisualizerError {
                    XCTAssertEqual(vizError, VisualizerError.rendererInitFailure)
                }
            }
        }
    }
    
    /// Tests handling of neural engine errors
    func testNeuralEngineErrorHandling() {
        // Enable neural engine
        settings.neuralEngineEnabled = true
        
        // Set neural engine to fail
        neuralEngine.shouldThrowError = true
        
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Start visualizer
        Task {
            do {
                try await visualizer.start()
                
                // Send data to trigger neural processing
                let mockData = Array(repeating: Float(0.5), count: 1024)
                audioEngine.sendMockAudioData(mockData, levels: (0.5, 0.5))
                
                // Wait a moment for processing
                try await Task.sleep(nanoseconds: 100_000_000)
                
                // Verify visualization still works even if neural engine fails
                XCTAssertTrue(visualizer.isActive)
                XCTAssertTrue(visualRenderer.updateCalled)
                XCTAssertTrue(visualRenderer.renderCalled)
            } catch {
                XCTFail("Visualization should continue despite neural engine errors: \(error)")
            }
        }
    }
    
    /// Tests recovery from audio data errors
    func testAudioDataErrorRecovery() {
        let visualizer = AudioVisualizer(
            audioEngine: audioEngine,
            visualizer: visualRenderer,
            neuralEngine: neuralEngine,
            settings: settings
        )
        
        // Start visualizer
        Task {
            do {
                try await visualizer.start()
                
                // First send invalid data (empty array)
                audioEngine.sendMockAudioData([], levels: (0, 0))
                
                // Wait a moment
                try await Task.sleep(nanoseconds: 50_000_000)
                
                // Then send valid data
                let validData = Array(repeating: Float(0.5), count: 1024)
                audioEngine.sendMockAudioData(validData, levels: (0.5, 0.5))
                
                // Wait for processing
                try await Task.sleep(nanoseconds: 50_000_000)
                
                // Verify the visualizer recovered and processed the valid data
                XCTAssertTrue(visualizer.isActive)
                XCTAssertEqual(visualRenderer.lastAudioData, validData)
                XCTAssertEqual(visualRenderer.lastLevels.left, 0.5)
                XCTAssertEqual(visualRenderer.lastLevels.right, 0.5)
755
