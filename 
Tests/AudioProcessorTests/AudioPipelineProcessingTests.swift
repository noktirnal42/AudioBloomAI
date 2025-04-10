        // Verify channel handling - ChannelAwareNode applies different gain to each channel
        // Left channel (even indices) should be amplified by 2.0
        // Right channel (odd indices) should be attenuated by 0.5
        let stereoSampleCount = processedBuffer.count / 2
        for samplePairIndex in 0..<stereoSampleCount {
            let leftIndex = samplePairIndex * 2
            let rightIndex = samplePairIndex * 2 + 1
            
            let expectedLeftSample = stereoBuffer[leftIndex] * 2.0
            let expectedRightSample = stereoBuffer[rightIndex] * 0.5
            
            XCTAssertEqual(processedBuffer[leftIndex], expectedLeftSample, accuracy: 0.001)
            XCTAssertEqual(processedBuffer[rightIndex], expectedRightSample, accuracy: 0.001)
        }
        // Verify individual samples
        let sampleCount = min(testBuffer.count, processedBuffer.count)
        for sampleIndex in 0..<sampleCount {
            let expectedSample = testBuffer[sampleIndex] * 2.0
            XCTAssertEqual(processedBuffer[sampleIndex], expectedSample, accuracy: 0.001)
        }
        var stereoBuffer = [Float](repeating: 0.0, count: leftChannel.count * 2)
        for sampleIndex in 0..<leftChannel.count {
            let leftBufferIndex = sampleIndex * 2
            let rightBufferIndex = leftBufferIndex + 1
            
            stereoBuffer[leftBufferIndex] = leftChannel[sampleIndex]       // Left channel
            stereoBuffer[rightBufferIndex] = rightChannel[sampleIndex]     // Right channel
        }
        // ChannelMapperNode should swap left and right channels
        for channelPairIndex in 0..<leftChannel.count {
            // Calculate indices for left and right channels
            let leftOutputIndex = channelPairIndex * 2
            let rightOutputIndex = leftOutputIndex + 1
            
            let leftInputIndex = rightOutputIndex // swapped in output
            let rightInputIndex = leftOutputIndex // swapped in output
            
            // Processed buffer should have channels swapped
            XCTAssertEqual(
                processedBuffer[leftOutputIndex], 
                stereoBuffer[leftInputIndex], 
                accuracy: 0.001
            )
            XCTAssertEqual(
                processedBuffer[rightOutputIndex], 
                stereoBuffer[rightInputIndex], 
                accuracy: 0.001
            )
        }
            var result = buffer
            
            // Process each stereo pair (left and right channels)
            for channelIndex in stride(from: 0, to: buffer.count, by: 2) {
                // Check if left channel index is valid
                if channelIndex < buffer.count {
                    // Amplify left channel
                    result[channelIndex] = buffer[channelIndex] * 2.0
                }
                
                // Check if right channel index is valid
                let rightChannelIndex = channelIndex + 1
                if rightChannelIndex < buffer.count {
                    // Attenuate right channel
                    result[rightChannelIndex] = buffer[rightChannelIndex] * 0.5
        // Verify the signal went through both nodes
        // First it should have been reduced in amplitude, then inverted
        for sampleIndex in 0..<min(testBuffer.count, processedBuffer.count) {
        // Stop the pipeline
        try audioPipeline.stop()
        
        // Verify the signal went through both nodes
        // First it should have been reduced in amplitude, then inverted
        for sampleIndex in 0..<min(testBuffer.count, processedBuffer.count) {
            let expectedSample = -0.5 * testBuffer[sampleIndex] // Half ampl    /// Tests signal flow through multiple nodes in a processing chain
    /// - Verifies that:
    ///   - Multiple nodes process the signal in sequence
    ///   - GainNode reduces amplitude by half
    ///   - InvertNode inverts the phase of the signal
    ///   - Combined effect properly applies both transformations
    func testSignalFlow() throws {
 that:
    ///   - Pipeline correctly processes a sine wave input
    ///   - Output buffer size matches input buffer size
    ///   - Signal passes    /// Tests stereo channel handling capabilities
    /// - Verifies that:
    ///   - Pipeline processes stereo signals correctly
    ///   - ChannelAwareNode applies correct gain to each channel:
    ///     * Left channel is amplified by 2.0
    ///     * Right channel is attenuated by 0.5
    ///   - Channel integrity is maintained throughout processing
    func testChannelHandling() throws {
    ///   - GainNode reduces amplitude by half
    ///   - InvertNode inverts the phase of the signal
    ///   - Combined effect properly applies both transformations
    func testSignalFlow() throws {
    /// - Left channel should be amplified by 2.0
= try audioPipeline.allocateBuffer(
            size: 1024 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        
        // Track buffers for cleanup
        var allocatedBuffers = [inputBuffer, outputBuffer]
        defer {
            // Clean up allocated buffers
            for bufferId in allocatedBuffers {
                audioPipeline.releaseBuffer(id: bufferId)
            }
        }
        
        // Create a test signal
        let testData = generateSineWave(frequency: 440.0, duration: 0.1, amplitude: 0.5)
        
        // Fill input buffer
        try testData.withUnsafeBytes { bytes in
            try audioPipeline.updateBuffer(
                id: inputBuffer,
                data: bytes.baseAddress!,
                size: testData.count * MemoryLayout<Float>.stride,
                options: [.waitForCompletion]
            )
        }
        
        // Configure pipeline with gain node
        try audioPipeline.reset()
        let gainNode = GainNode(name: "Gain Node", gain: 2.0) // Double the amplitude
        try audioPipeline.addNode(gainNode, connections: [])
        
        // Start the pipeline
        try audioPipeline.startStream()
        
        // Process the buffer
        let success = try audioPipeline.process(
            inputBuffers: [inputBuffer],
            outputBuffers: [outputBuffer],
            context: audioPipeline
        )
        
        XCTAssertTrue(success, "Processing should succeed")
        
        // Verify output
        let processedData = try getBufferContents(outputBuffer)
        
        // Verify RMS level
        let inputRMS = calculateRMSLevel(testData)
        let outputRMS = calculateRMSLevel(processedData)
        
        XCTAssertEqual(
            outputRMS,
            inputRMS * 2.0, 
            accuracy: 0.01,
            "Output RMS level should be doubled"
        )
        
        // Verify individual samples
        for sampleIndex in 0..<min(testData.count, processedData.count) {
            let expectedSample = testData[sampleIndex] * 2.0
            XCTAssertEqual(
                processedData[sampleIndex],
                expectedSample,
                accuracy: 0.001,
                "Each sample should be multiplied by the gain factor"
            )
        }
    }
            XCTAssertEqual(
                processedBuffer[rightChannelIndex], 
                expectedRightSample, 
                accuracy: 0.001,
                "Right channel should be attenuated by 0.5"
            )
        }
            let rightChannelIndex = leftChannelIndex + 1
            
            let expectedLeftSample = stereoBuffer[leftChannelIndex] * 2.0
            let expectedRightSample = stereoBuffer[rightChannelIndex] * 0.5
            
            XCTAssertEqual(
                processedBuffer[leftChannelIndex], 
                expectedLeftSample, 
                accuracy: 0.001,
                "Left channel should be amplified by 2.0"
            )
            XCTAssertEqual(
                processedBuffer[rightChannelIndex], 
                expectedRightSample, 
                accuracy: 0.001,
                "Right channel should be attenuated by 0.5"
            )
        }
            // Calculate indices for left and right channels
            let leftChannelIndex = channelPairIndex * 2
            let rightChannelIndex = leftChannelIndex + 1
            
            // Swapped indices for verification
            let originalRightChannelIndex = rightChannelIndex
            let originalLeftChannelIndex = leftChannelIndex
            
            // Processed buffer should have channels swapped
        // Verify that gain was applied correctly (RMS should be doubled)
        XCTAssertEqual(
            outputRMS, 
            inputRMS * 2.0, 
            accuracy: 0.01,
            "Output RMS level should be doubled"
        )

        // Verify individual samples
        let sampleCount = min(testBuffer.count, processedBuffer.count)
        for sampleIndex in 0..<sampleCount {
            let expectedSample = testBuffer[sampleIndex] * 2.0
            XCTAssertEqual(
                processedBuffer[sampleIndex], 
                expectedSample, 
                accuracy: 0.001,
                "Each sample should be multiplied by the gain factor"
            )
        }
                processedBuffer[rightChannelIndex], 
                stereoBuffer[originalLeftChannelIndex], 
                accuracy: 0.001,
                "Right channel in output should contain left channel from input"
            )
        }
final class OperationCounterNode: AudioProcessingNode {
    var processCount = 0
    /// Tests sample rate conversion processing
    /// - Verifies that:
    ///   - Output buffer size changes proportionally to the sample rate change
    ///   - Doubling the sample rate doubles the output buffer size
    ///   - Peak amplitude is preserved in the conversion
    func testSampleRateConversion() throws {
    ///   - GainNode multiplies all samples by the gain factor
    ///   - Both RMS level and individual samples show correct amplification
    ///   - Input signal is preserved in shape but changed in amplitude
    func testGainAdjustment() throws {
    /// Tests that sample rate conversion is handled correctly
    /// - Verifies that:
    ///   - Output buffer size changes proportionally to the sample rate change
    ///   - Doubling the sample rate doubles the output buffer size
    /// Tests channel mapping operations
    /// - Verifies that:
    ///   - ChannelMapperNode correctly swaps left and right channels
    ///   - Different frequencies in each channel are correctly swapped
    ///   - Channel mapping preserves signal quality
    func testChannelMapping() throws {
        XCTAssertEqual(
            outputRMS, 
            inputRMS * 2.0, 
            accuracy: 0.01,
            "Output RMS level should be doubled"
        )

        XCTAssertEqual(
            processedBuffer.count, 
            inputSampleCount * 2, 
            accuracy: 10,
            "Output buffer size should double when sample rate doubles"
        )
        let sampleCount = min(testBuffer.count, processedBuffer.count)
        for sampleIndex in 0..<sampleCount {
            let expectedSample = testBuffer[sampleIndex] * 2.0
            XCTAssertEqual(
                processedBuffer[sampleIndex], 
        XCTAssertEqual(
            outputPeak, 
            inputPeak, 
            accuracy: 0.1,
            "Peak amplitude should be preserved during sample rate conversion"
        )
                accuracy: 0.001,
                "Each sample should be multiplied by the gain factor"
            )
        }
(
            processedBuffer.count, 
            inputSampleCount * 2, 
            accuracy: 10,
            "Output buffer size should double when sample rate doubles"
        )
t = min(testBuffer.count, processedBuffer.count)
        for sampleIndex in 0..<sampleCount {
            let expectedSample = testBuffer[sampleIndex] * 2.0
            XCTAssertEqual(
        var stereoBuffer = [Float](repeating: 0.0, count: leftChannel.count * 2)
        for sampleIndex in 0..<leftChannel.count {
            let leftChannelIndex = sampleIndex * 2
            let rightChannelIndex = leftChannelIndex + 1
            
            stereoBuffer[leftChannelIndex] = leftChannel[sampleIndex]       // Left channel
            stereoBuffer[rightChannelIndex] = rightChannel[sampleIndex]     // Right channel
        }
        }
       let leftChannelIndex = sampleIndex * 2
            let rightChannelIndex = leftChannelIndex + 1
            
            stereoBuffer[leftChannelIndex] = leftChannel[sampleIndex]       // Left channel
            stereoBuffer[rightChannelIndex] = rightChannel[sampleIndex]     // Right channel
        // ChannelMapperNode should swap left and right channels
        for channelPairIndex in 0..<leftChannel.count {
            // Calculate indices for left and right channels
            let leftChannelIndex = channelPairIndex * 2
        // ChannelMapperNode should swap left and right channels
        for channelPairIndex in 0..<leftChannel.count {
            // Calculate indices for left and right channels
            let leftChannelIndex = channelPairIndex * 2
            let rightChannelIndex = leftChannelIndex + 1
            
            // Swapped indices for verification
            let originalRightChannelIndex = rightChannelIndex
            let originalLeftChannelIndex = leftChannelIndex
            
            // Processed buffer should have channels swapped
            XCTAssertEqual(
                processedBuffer[leftChannelIndex], 
                stereoBuffer[originalRightChannelIndex], 
                accuracy: 0.001,
                "Left channel in output should contain right channel from input"
            )
            XCTAssertEqual(
                processedBuffer[rightChannelIndex], 
                stereoBuffer[originalLeftChannelIndex], 
                accuracy: 0.001,
                "Right channel in output should contain left channel from input"
            )
        }
            // Processed buffer should have channels swapped
            XCTAssertEqual(
                processedBuffer[leftChannelIndex], 
                stereoBuffer[originalRightChannelIndex], 
                accuracy: 0.001,
                "Left channel in output should contain right channel from input"
            )
            XCTAssertEqual(
                processedBuffer[rightChannelIndex], 
                stereoBuffer[originalLeftChannelIndex], 
                accuracy: 0.001,
                "Right channel in output should contain left channel from input"
        XCTAss        XC    /// Tests handling of invalid buffer sizes
    /// - Verifies that:
    ///   - Processing a buffer with invalid size throws an appropriate error
    ///   - Processing an empty buffer throws an appropriate error
    ///   - The error type is AudioPipelineError as expected
    func testInvalidBufferHandling() throws {
hrow
eError, 
                "Should throw an AudioPipelineError for invalid buffer size"
            )
        }
    func testChannelMapping() throws {
    /// Tests that the pipeline handles invalid buffer sizes correctly
    /// - Verifies that processing an invalid buffer size throws an appropriate error
    /// - Verifies that processing an empty buffer throws an appropriate error
    func testInvalidBufferHandling() throws {
    /// Tests that processing errors from nodes are handled correctly
    /// - Verifies that errors thrown by processing nodes propagate to the caller
    /// - Checks that the specific error type is preserved in propagation
    func testProcessingErrors() throws {
    /// Tests channel mapping operations
    /// - Verifies that:
    ///   - ChannelMapperNode correctly swaps left and right channels
    ///   - Different frequencies in each channel are correctly swapped
    ///   - Channel mapping preserves signal quality
    func testChannelMapping() throws {
    /// Tests that errors propagate correctly through the pipeline
    /// - Verifies that:
    ///   - Nodes before an error-throwing node are processed
    ///   - Nodes after an error-throwing node are not processed
    ///   - The error from the throwing node is propagated to the caller
    func testErrorPropagation() throws {
wn by processing nodes propagate to the caller
    ///   - The specific error type from the node is preserved
    ///   - The pipeline properly reports the failing node's error
    func testProcessingErrors() throws {
 a buffer with invalid size throws an appropriate error
    ///   - Processing an empty buffer throws an appropriate error
    ///   - The error type is AudioPipelineError as expected
    func testInvalidBufferHandling() throws {
