import Foundation
import AVFoundation
import Accelerate
import Combine

@MainActor final class SimpleAudioCapture: ObservableObject {
    struct Levels { let left: Float; let right: Float }

    private let engine = AVAudioEngine()
    private let sessionQueue = DispatchQueue(label: "com.audiobloom.simpleaudio")

    let frequencyPublisher = PassthroughSubject<[Float], Never>()
    let levelsPublisher = PassthroughSubject<Levels, Never>()

    private var fftSetup: vDSP_DFT_Setup?
    private let fftSize: Int = 1024
    private var window: [Float]

    init() {
        window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(fftSize), vDSP_DFT_Direction.FORWARD)
    }


    func start() throws {
        let input = engine.inputNode
        let format = input.inputFormat(forBus: 0)
        let bufferSize: AVAudioFrameCount = AVAudioFrameCount(fftSize)

        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: bufferSize, format: format) { [weak self] buffer, _ in
            self?.process(buffer: buffer)
        }
        try engine.start()
    }

    func stop() {
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
    }

    private func process(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData, let fftSetup = fftSetup else { return }
        let frameCount = Int(buffer.frameLength)
        let count = min(frameCount, fftSize)

        // Mono mix of first two channels if available
        let left = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        let right: [Float] = (buffer.format.channelCount > 1) ? Array(UnsafeBufferPointer(start: channelData[1], count: frameCount)) : left

        var mono = [Float](repeating: 0, count: fftSize)
        for i in 0..<count { mono[i] = 0.5 * (left[i] + right[i]) * window[i] }

        // FFT
        let realIn = mono
        let imagIn = [Float](repeating: 0, count: fftSize)
        var realOut = [Float](repeating: 0, count: fftSize)
        var imagOut = [Float](repeating: 0, count: fftSize)
        vDSP_DFT_Execute(fftSetup, realIn, imagIn, &realOut, &imagOut)

        var mag = [Float](repeating: 0, count: fftSize/2)
        for i in 0..<fftSize/2 { mag[i] = sqrt(realOut[i]*realOut[i] + imagOut[i]*imagOut[i]) }
        var scale = Float(1.0 / Float(fftSize))
        vDSP_vsmul(mag, 1, &scale, &mag, 1, vDSP_Length(fftSize/2))

        // Simple dB scale normalize 0..1
        var norm = [Float](repeating: 0, count: fftSize/2)
        for i in 0..<fftSize/2 {
            let db = 10.0 * log10f(mag[i] + 1e-8)
            norm[i] = min(max((db + 90.0) / 90.0, 0.0), 1.0)
        }

        // Levels (RMS)
        var l: Float = 0, r: Float = 0
        vDSP_rmsqv(left, 1, &l, vDSP_Length(count))
        vDSP_rmsqv(right, 1, &r, vDSP_Length(count))

        DispatchQueue.main.async {
            self.frequencyPublisher.send(norm)
            self.levelsPublisher.send(Levels(left: l, right: r))
        }
    }
}