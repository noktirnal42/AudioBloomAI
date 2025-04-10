// Updated AudioBloomCore types
import Foundation

struct ProcessingResult {
    let frameCount: Int
    let bufferSize: Int
    let errorCount: Int
}

extension AudioBloomCore {
    func getProcessingStats() -> ProcessingResult {
        return ProcessingResult(
            frameCount: currentFrameCount,
            bufferSize: currentBufferSize,
            errorCount: currentErrorCount
        )
    }
}
