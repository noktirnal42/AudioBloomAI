        for sampleIndex in 0..<min(testBuffer.count, processedBuffer.count) {
            let expectedSample = -0.5 * testBuffer[sampleIndex] // Half amplitude and inverted
            XCTAssertEqual(processedBuffer[sampleIndex], expectedSample, accuracy: 0.001)
        for frameIndex in 0..<processedBuffer.count/2 {
            let leftIndex = frameIndex * 2
            let rightIndex = frameIndex * 2 + 1
        for sampleIndex in 0..<min(testBuffer.count, processedBuffer.count) {
            XCTAssertEqual(processedBuffer[sampleIndex], testBuffer[sampleIndex] * 2.0, accuracy: 0.001)
        for frameIndex in 0..<leftChannel.count {
            stereoBuffer[frameIndex * 2] = leftChannel[frameIndex]       // Left channel
            stereoBuffer[frameIndex * 2 + 1] = rightChannel[frameIndex]  // Right channel
        for frameIndex in 0..<leftChannel.count {
            // Processed buffer should have channels swapped
            XCTAssertEqual(processedBuffer[frameIndex * 2], stereoBuffer[frameIndex * 2 + 1], accuracy: 0.001)
            XCTAssertEqual(processedBuffer[frameIndex * 2 + 1], stereoBuffer[frameIndex * 2], accuracy: 0.001)
            for channelIndex in stride(from: 0, to: buffer.count, by: 2) where channelIndex < buffer.count {
                result[channelIndex] = buffer[channelIndex] * 2.0 // Amplify left channel
                
                if channelIndex + 1 < buffer.count {
            for channelIndex in stride(from: 0, to: buffer.count, by: 2) where channelIndex + 1 < buffer.count {
                      let outputSize = Int(Double(buffer.count) * inputToOutputRatio)
            var output = [Float](repeating: 0.0, count: outputSize)

            for outputIndex in 0..<outputSize {
                let inputIndex = Double(outputIndex) / inputToOutputRatio
                let inputIndexInt = Int(inputIndex)

                if inputIndexInt < buffer.count {
                    output[outputIndex] = buffer[inputIndexInt]
                }
            }
     }
