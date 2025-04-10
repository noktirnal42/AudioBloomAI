// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import XCTest
@testable import AudioProcessor

/// Tests focusing on audio processing functionality in the pipeline.
@available(macOS 15.0, *)
final class AudioPipelineProcessingTests: XCTestCase {
    var pipeline: AudioPipeline!
    var mockSource: MockAudioSource!
    var mockSink: MockAudioSink!
    
    override func setUp() {
        super.setUp()
        mockSource = MockAudioSource()
        mockSink = MockAudioSink()
        pipeline = AudioPipeline()
    }
    
    override func tearDown() {
        pipeline = nil
        mockSource = nil
        mockSink = nil
        super.tearDown()
    }
    
    // MARK: - Processing Tests
    
    /// Tests pipeline with FFT processing
    func testFFTProcessing() {
        // Set up a pipeline: source -> fft -> ifft -> sink
        let fftProcessor = FFTProcessor(id: "fft")
        let ifftProcessor = IFFTProcessor(id: "ifft")
        
        // Add nodes
        pipeline.addNode(mockSource)
        pipeline.addNode(fftProcessor)
        pipeline.addNode(ifftProcessor)
        pipeline.addNode(mockSink)
        
        // Connect nodes
        XCTAssertNoThrow(try pipeline.connect(from: mockSource, to: fftProcessor))
        XCTAssertNoThrow(try pipeline.connect(from: fftProcessor, to: ifftProcessor))
        XCTAssertNoThrow(try pipeline.connect(from: ifftProcessor, to: mockSink))
        
        // Configure test data - a simple sine wave
        let sampleRate =

