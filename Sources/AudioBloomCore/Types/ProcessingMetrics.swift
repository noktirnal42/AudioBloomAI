// AudioBloomCore types
import Foundation

struct ProcessingMetrics {
    let frameCount: Int
    let bufferSize: Int
    let errorCount: Int
    let processingTime: TimeInterval
}

extension AudioBloomCore {
    func getMetrics() -> ProcessingMetrics {
        return ProcessingMetrics(
            frameCount: frameCount,
            bufferSize: bufferSize,
            errorCount: errorCount,
            processingTime: processingTime
        )
    }
}
