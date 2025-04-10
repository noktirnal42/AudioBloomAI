// AudioVisualizerBridge extensions
import Foundation

extension AudioVisualizerBridge {
    func updateFrameCount() {
        frameCount += 1
    }
    
    func updateProcessingTime(elapsed: TimeInterval) {
        processingTime += elapsed
    }
}
