import XCTest
import AVFoundation
import Combine
@testable import AudioProcessor
@testable import AudioBloomCore

final class AudioProcessorTests: XCTestCase {
    // AudioEngine instance for testing
    private var audioEngine: AudioEngine!
    
    // Subscription set for Combine publishers
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Setup and Teardown
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Create a fresh AudioEngine instance for each test
        audioEngine = AudioEngine()
        
        // Allow time for device enumeration
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
    }
    
    override func tearDown() async throws {
        // Clean up the audio engine
        if audioEngine.isCapturing {
            audioEngine.stopCapture()
        }
        
        // Clear any subscriptions
        cancellables.removeAll()
        
        // Release the audio engine
        audioEngine = nil
        
        try await super.tearDown()
    }
    
    // MARK: - Device Enumeration Tests
    
    func testDeviceEnumeration() async throws {
        // Verify that the audio engine can enumerate audio devices
        audioEngine.refreshAudioDevices()
        
        // Wait for device enumeration to complete
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        
        // Check that we have at least one input and output device on macOS
        #if os(macOS)
        XCTAssertFalse(audioEngine.availableInputDevices.isEmpty, "Should have at least one input device")
        XCTAssertFalse(audioEngine.availableOutputDevices.isEmpty, "Should have at least one output device")
        #endif
        
        // Verify selected device properties
        XCTAssertNotNil(audioEngine.selectedInputDevice, "Should have a default input device selected")
        if let selectedDevice = audioEngine.selectedInputDevice {
            XCTAssertTrue(selectedDevice.isInput, "Selected input device should be marked as input")
            XCTAssertFalse(selectedDevice.id.isEmpty, "Device ID should not be empty")
            XCTAssertFalse(selectedDevice.name.isEmpty, "Device name should not be empty")
        }
    }
    
    func testDeviceSelection() async throws {
        // Skip test if no devices are available
        guard !audioEngine.availableInputDevices.isEmpty else {
            throw XCTSkip("No input devices available for testing")
        }
        
        // Get the first available input device
        let testDevice = audioEngine.availableInputDevices[0]
        
        // Set it as the active device
        audioEngine.setActiveDevice(testDevice, reconfigureAudio: true)
        
        // Verify the device was selected
        XCTAssertEqual(audioEngine.selectedInputDevice?.id, testDevice.id, "AudioEngine should use the selected device")
    }
    
    // MARK: - Audio Capture Tests
    
    func testAudioCaptureInitialization() async throws {
        // Set up the audio session
        try await audioEngine.setupAudioSession()
        
        // Start capture
        try audioEngine.startCapture()
        
        // Verify the engine is running
        XCTAssertTrue(audioEngine.isCapturing, "Audio engine should be capturing")
        
        // Wait briefly to ensure capture is stable
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        
        // Stop capture
        audioEngine.stopCapture()
        
        // Verify the engine stopped
        XCTAssertFalse(audioEngine.isCapturing, "Audio engine should not be capturing after stop")
    }
    
    func testAudioSourceSelection() async throws {
        // Test each audio source type
        for sourceType in AudioEngine.AudioSourceType.allCases {
            // Set the audio source
            audioEngine.setAudioSource(sourceType)
            
            // Verify the source was correctly set
            XCTAssertEqual(audioEngine.activeAudioSource, sourceType, "Audio source should be set to \(sourceType)")
            
            // Start and stop capture to ensure configuration works
            try audioEngine.startCapture()
            XCTAssertTrue(audioEngine.isCapturing, "Audio engine should be capturing")
            
            // Wait briefly
            try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
            
            audioEngine.stopCapture()
            XCTAssertFalse(audioEngine.isCapturing, "Audio engine should not be capturing after stop")
        }
    }
    
    // MARK: - FFT Processing Tests
    
    func testFFTProcessing() async throws {
        // Create an expectation for frequency data
        let expectation = expectation(description: "Received frequency data")
        
        // Subscribe to the audio data publisher
        audioEngine.getAudioDataPublisher()
            .sink { audioData in
                // Verify frequency data format
                XCTAssertFalse(audioData.frequencyData.isEmpty, "Frequency data should not be empty")
                XCTAssertEqual(audioData.frequencyData.count, AudioBloomCore.Constants.defaultFFTSize / 2, 
                              "Frequency data should have correct size")
                
                // Verify data values are in expected range (0.0-1.0)
                for value in audioData.frequencyData {
                    XCTAssertGreaterThanOrEqual(value, 0.0, "Frequency data should be >= 0.0")
                    XCTAssertLessThanOrEqual(value, 1.0, "Frequency data should be <= 1.0")
                }
                
                expectation.fulfill()
            }
            .store(in: &cancellables)
        
        // Start audio capture
        try audioEngine.startCapture()
        
        // Wait for frequency data or timeout
        await fulfillment(of: [expectation], timeout: 5.0)
        
        // Stop capture
        audioEngine.stopCapture()
    }
    
    // MARK: - Audio Level Tests
    
    func testAudioLevelCalculation() async throws {
        // Create an expectation for audio level data
        let expectation = expectation(description: "Received audio level data")
        
        // Subscribe to the audio data publisher
        audioEngine.getAudioDataPublisher()
            .sink { audioData in
                // Verify level format - we expect a tuple of (left, right)
                XCTAssertNotNil(audioData.levels.0, "Left level should not be nil")
                XCTAssertNotNil(audioData.levels.1, "Right level should not be nil")
                
                // Verify level values are in expected range (0.0-1.0)
                XCTAssertGreaterThanOrEqual(audioData.levels.0, 0.0, "Left level should be >= 0.0")
                XCTAssertLessThanOrEqual(audioData.levels.0, 1.0, "Left level should be <= 1.0")
                
                XCTAssertGreaterThanOrEqual(audioData.levels.1, 0.0, "Right level should be >= 0.0")
                XCTAssertLessThanOrEqual(audioData.levels.1, 1.0, "Right level should be <= 1.0")
                
                expectation.fulfill()
            }
            .store(in: &cancellables)
        
        // Start audio capture
        try audioEngine.startCapture()
        
        // Wait for level data or timeout
        await fulfillment(of: [expectation], timeout: 5.0)
        
        // Stop capture
        audioEngine.stopCapture()
    }
    
    // MARK: - Configuration Tests
    
    func testVolumeAdjustment() async throws {
        // Test various volume settings
        let testVolumes: [(mic: Float, system: Float)] = [
            (0.0, 0.0),  // Muted
            (0.5, 0.5),  // Half volume
            (1.0, 1.0),  // Full volume
            (0.3, 0.7),  // Mixed levels
            (1.5, -0.5)  // Out of range (should be clamped)
        ]
        
        for (micVolume, systemVolume) in testVolumes {
            // Adjust volumes
            audioEngine.adjustVolumes(micVolume: micVolume, systemVolume: systemVolume)
            
            // Start and stop capture to ensure configuration is applied
            try audioEngine.startCapture()
            
            // Wait briefly
            try await Task.sleep(nanoseconds: 200_000_000) // 0.2 seconds
            
            audioEngine.stopCapture()
            
            // No assertions needed as long as it doesn't crash
            // The actual volume values are internal state in the audio engine
        }
    }
    
    // MARK: - Error Handling Tests
    
    func testErrorHandlingDuringStartStopCapture() async throws {
        // Starting twice should not throw an error
        try audioEngine.startCapture()
        try audioEngine.startCapture() // Should be safe to call again
        
        // Stopping twice should not throw an error
        audioEngine.stopCapture()
        audioEngine.stopCapture() // Should be safe to call again
    }
    
    func testInvalidDeviceHandling() async throws {
        // Create a non-existent device
        let invalidDevice = AudioEngine.AudioDevice(
            id: "invalid-device-id",
            name: "Invalid Device",
            manufacturer: "Test",
            isInput: true,
            sampleRate: 48000,
            channelCount: 2
        )
        
        // Try to set it as the active device
        // This should not cause a crash but might log an error
        audioEngine.setActiveDevice(invalidDevice)
        
        // Try to start capture, which may fail gracefully
        do {
            try audioEngine.startCapture()
            // If it didn't throw, we should be able to stop cleanly
            audioEngine.stopCapture()
        } catch {
            // Failure is acceptable here, as long as it's a controlled error
            XCTAssertTrue(error is AudioBloomCore.Error, "Should throw a typed error")
        }
    }
    
    // MARK: - Simulated Audio Tests
    
    func testSimulatedAudioProcessing() async throws {
        // This test creates a mock audio processor to validate FFT and level calculation
        // without relying on actual audio capture
        
        // Create a simple sine wave at 440 Hz with amplitude 0.5
        let sampleRate = 44100.0
        let frequency = 440.0
        let duration = 1.0
        let amplitude = Float(0.5)
        
        let sampleCount = Int(sampleRate * duration)
        var sineWave = [Float](repeating: 0.0, count: sampleCount)
        
        for i in 0..<sampleCount {
            let phase = 2.0 * Float.pi * Float(frequency) * Float(i) / Float(sampleRate)
            sineWave[i] = amplitude * sin(phase)
        }
        
        // Create an FFT helper with the same size as used in the AudioEngine
        let fftHelper = FFTHelper(fftSize: AudioBloomCore.Constants.defaultFFTSize)
        
        // Process the sine wave with FFT
        let fftData = fftHelper.performFFT(data: sineWave)
        
        // A 440 Hz tone should have a peak around index 440 / (sampleRate/fftSize)
        let expectedPeakIndex = Int(frequency / (sampleRate / Double(AudioBloomCore.Constants.defaultFFTSize)))
        let peakRange = (expectedPeakIndex - 2)...(expectedPeakIndex + 2)
        
        // Find the actual peak
        var maxValue: Float = 0.0
        var maxIndex = 0
        
        for i in 0..<fftData.count {
            if fftData[i] > maxValue {
                maxValue = fftData[i]
                maxIndex = i
            }
        }
        
        // Verify the peak is in the expected range
        XCTAssertTrue(peakRange.contains(maxIndex), "FFT should detect the 440 Hz frequency peak")
        
        // Now test level calculation
        // For a sine wave with amplitude 
