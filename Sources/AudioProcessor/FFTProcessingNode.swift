import Foundation
import AVFoundation
import Accelerate
import Logging
import AudioBloomCore

/// FFT Processing Node for real-time frequency analysis of audio data
@available(macOS 15.0, *)
public class FFTProcessingNode: AudioProcessingNode {
    // MARK: - AudioProcessingNode Protocol Properties
    
    /// The unique identifier for this node
    public let id: UUID
    
    /// The display name of this node
    public var name: String
    
    /// Whether this node is enabled in the processing chain
    public var isEnabled: Bool
    
    /// Input requirements for this node
    public var inputRequirements: AudioNodeIORequirements
    
    /// Output capabilities of this node
    public var outputCapabilities: AudioNodeIORequirements
    
    // MARK: - FFT Processing Properties
    
    /// Size of the FFT operation (must be a power of 2)
    private var fftSize: Int
    
    /// Logger instance
    private let logger = Logger(label: "com.audiobloom.fft-processor")
    
    /// Window function type for FFT
    public enum WindowFunction: String, Codable {
        case hann
        case hamming
        case blackman
        case none
    }
    
    /// Current window function
    private var windowFunction: WindowFunction
    
    /// Window buffer for windowing the signal before FFT
    private var window: [Float]
    
    /// FFT setup for real signal
    private var fftSetup: vDSP_DFT_Setup?
    
    /// Temporary buffers for FFT computation
    private var realInput: [Float]
    private var imagInput: [Float]
    private var realOutput: [Float]
    private var imagOutput: [Float]
    private var magnitude: [Float]
    
    /// Frequency band results
    private var bassLevel: Float = 0.0
    private var midLevel: Float = 0.0
    private var trebleLevel: Float = 0.0
    
    /// Band frequency ranges in Hz
    private var bassRange: ClosedRange<Float> = 20.0...250.0
    private var midRange: ClosedRange<Float> = 250.0...4000.0
    private var trebleRange: ClosedRange<Float> = 4000.0...20000.0
    
    /// Sample rate for frequency calculations
    private var sampleRate: Double = AudioBloomCore.Constants.defaultSampleRate
    
    /// Real and imaginary pointers for DSPSplitComplex
    private var realPtr: UnsafeMutablePointer<Float>?
    private var imagPtr: UnsafeMutablePointer<Float>?
    
    // MARK: - Initialization
    
    /// Initializes a new FFT processing node with the specified parameters
    /// - Parameters:
    ///   - id: Optional UUID for the node, auto-generated if not provided
    ///   - name: Name of the node
    ///   - fftSize: Size of the FFT operation (default: 2048)
    ///   - windowFunction: Window function to apply to input samples (default: hann)
    public init(
        id: UUID = UUID(),
        name: String = "FFT Processor",
        fftSize: Int = 2048,
        windowFunction: WindowFunction = .hann
    ) {
        self.id = id
        self.name = name
        self.isEnabled = true
        
        // Ensure FFT size is a power of 2
        self.fftSize = 1 << Int(log2(Double(max(64, min(16384, fftSize)))).rounded())
        
        // Initialize window function
        self.windowFunction = windowFunction
        
        // Create FFT setup
        self.fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(self.fftSize),
            vDSP_DFT_Direction.FORWARD
        )
        
        // Create window for better frequency resolution
        self.window = [Float](repeating: 0, count: self.fftSize)
        self.createWindow()
        
        // Initialize working buffers
        self.realInput = [Float](repeating: 0, count: self.fftSize)
        self.imagInput = [Float](repeating: 0, count: self.fftSize)
        self.realOutput = [Float](repeating: 0, count: self.fftSize)
        self.imagOutput = [Float](repeating: 0, count: self.fftSize)
        self.magnitude = [Float](repeating: 0, count: self.fftSize/2)
        
        // Allocate memory for real and imaginary parts
        self.realPtr = UnsafeMutablePointer<Float>.allocate(capacity: self.fftSize)
        self.imagPtr = UnsafeMutablePointer<Float>.allocate(capacity: self.fftSize)
        
        // Initialize to zero
        self.realPtr?.initialize(repeating: 0, count: self.fftSize)
        self.imagPtr?.initialize(repeating: 0, count: self.fftSize)
        
        // Configure input requirements and output capabilities
        self.inputRequirements = AudioNodeIORequirements(
            supportedFormats: [
                AVAudioFormat(
                    standardFormatWithSampleRate: AudioBloomCore.Constants.defaultSampleRate,
                    channels: 1
                )!,
                AVAudioFormat(
                    standardFormatWithSampleRate: AudioBloomCore.Constants.defaultSampleRate,
                    channels: 2
                )!
            ],
            channels: .oneOrMore,
            bufferSize: AudioNodeIORequirements.CountRange(min: 64, max: 16384),
            sampleRates: [44100, 48000, 88200, 96000]
        )
        
        // Output capabilities - produces frequency data
        self.outputCapabilities = AudioNodeIORequirements(
            supportedFormats: [
                AVAudioFormat(
                    standardFormatWithSampleRate: AudioBloomCore.Constants.defaultSampleRate,
                    channels: 1
                )!
            ],
            channels: .one,
            bufferSize: AudioNodeIORequirements.CountRange(min: self.fftSize / 2, max: self.fftSize / 2),
            sampleRates: [44100, 48000, 88200, 96000]
        )
        
        logger.info("Initialized FFT processing node: id=\(id), size=\(self.fftSize)")
    }
    
    deinit {
        // Clean up FFT setup
        if let fftSetup = fftSetup {
            vDSP_DFT_DestroySetup(fftSetup)
        }
        
        // Clean up allocated memory
        if let realPtr = realPtr {
            realPtr.deinitialize(count: fftSize)
            realPtr.deallocate()
        }
        
        if let imagPtr = imagPtr {
            imagPtr.deinitialize(count: fftSize)
            imagPtr.deallocate()
        }
        
        logger.info("FFT processing node deinitialized: id=\(id)")
    }
    
    // MARK: - AudioProcessingNode Protocol Implementation
    
    /// Configure this node with parameters
    /// - Parameter parameters: Configuration parameters for this node
    public func configure(parameters: [String: Any]) throws {
        logger.debug("Configuring FFT node: \(parameters)")
        
        // Update FFT size if provided
        if let newSize = parameters["fftSize"] as? Int {
            // Ensure FFT size is a power of 2
            let validSize = 1 << Int(log2(Double(max(64, min(16384, newSize)))).rounded())
            
            if validSize != self.fftSize {
                self.fftSize = validSize
                
                // Re-create FFT setup
                if let fftSetup = fftSetup {
                    vDSP_DFT_DestroySetup(fftSetup)
                }
                
                self.fftSetup = vDSP_DFT_zop_CreateSetup(
                    nil,
                    vDSP_Length(self.fftSize),
                    vDSP_DFT_Direction.FORWARD
                )
                
                // Re-initialize buffers
                self.window = [Float](repeating: 0, count: self.fftSize)
                self.realInput = [Float](repeating: 0, count: self.fftSize)
                self.imagInput = [Float](repeating: 0, count: self.fftSize)
                self.realOutput = [Float](repeating: 0, count: self.fftSize)
                self.imagOutput = [Float](repeating: 0, count: self.fftSize)
                self.magnitude = [Float](repeating: 0, count: self.fftSize/2)
                
                // Reallocate memory
                if let realPtr = self.realPtr {
                    realPtr.deinitialize(count: fftSize)
                    realPtr.deallocate()
                }
                
                if let imagPtr = self.imagPtr {
                    imagPtr.deinitialize(count: fftSize)
                    imagPtr.deallocate()
                }
                
                self.realPtr = UnsafeMutablePointer<Float>.allocate(capacity: self.fftSize)
                self.imagPtr = UnsafeMutablePointer<Float>.allocate(capacity: self.fftSize)
                
                self.realPtr?.initialize(repeating: 0, count: self.fftSize)
                self.imagPtr?.initialize(repeating: 0, count: self.fftSize)
                
                // Update window
                self.createWindow()
                
                logger.info("FFT size updated: \(self.fftSize)")
            }
        }
        
        // Update window function if provided
        if let windowName = parameters["windowFunction"] as? String,
           let newWindowFunction = WindowFunction(rawValue: windowName.lowercased()) {
            if newWindowFunction != self.windowFunction {
                self.windowFunction = newWindowFunction
                self.createWindow()
                logger.info("Window function updated: \(self.windowFunction)")
            }
        }
        
        // Update sample rate if provided
        if let newSampleRate = parameters["sampleRate"] as? Double {
            self.sampleRate = newSampleRate
            logger.info("Sample rate updated: \(self.sampleRate)")
        }
        
        // Update frequency bands if provided
        if let bassMin = parameters["bassMin"] as? Float,
           let bassMax = parameters["bassMax"] as? Float {
            self.bassRange = bassMin...bassMax
        }
        
        if let midMin = parameters["midMin"] as? Float,
           let midMax = parameters["midMax"] as? Float {
            self.midRange = midMin...midMax
        }
        
        if let trebleMin = parameters["trebleMin"] as? Float,
           let trebleMax = parameters["trebleMax"] as? Float {
            self.trebleRange = trebleMin...trebleMax
        }
    }
    
    /// Process incoming audio data
    /// - Parameters:
    ///   - inputBuffers: The input audio buffer identifiers
    ///   - outputBuffers: The output audio buffer identifiers
    ///   - context: The processing context
    /// - Returns: Whether processing was successful
    public func process(
        inputBuffers: [AudioBufferID],
        outputBuffers: [AudioBufferID],
        context: AudioProcessingContext
    ) async throws -> Bool {
        // Skip processing if disabled
        guard isEnabled else {
            logger.debug("Skipping disabled FFT node: id=\(id)")
            return true
        }
        
        logger.debug("Processing FFT node: id=\(id), input=\(inputBuffers.count) buffers, output=\(outputBuffers.count) buffers")
        
        // Ensure we have at least one input buffer
        guard !inputBuffers.isEmpty else {
            logger.warning("No input buffers provided to FFT node")
            return false
        }
        
        // Ensure we have at least one output buffer
        guard !outputBuffers.isEmpty else {
            logger.warning("No output buffers provided to FFT node")
            return false
        }
        
        // Get the input buffer
        let inputBuffer: AudioBuffer
        do {
            inputBuffer = try context.bufferManager.getBuffer(id: inputBuffers[0])
        } catch {
            logger.error("Failed to get input buffer: \(error)")
            return false
        }
        
        // Get the output buffer
        let outputBuffer: AudioBuffer
        do {
            outputBuffer = try context.bufferManager.getBuffer(id: outputBuffers[0])
        } catch {
            logger.error("Failed to get output buffer: \(error)")
            return false
        }
        
        // Ensure input buffer has CPU-accessible data
        guard let inputData = inputBuffer.cpuBuffer else {
            logger.error("Input buffer has no CPU-accessible data")
            return false
        }
        
        // Ensure output buffer has CPU-accessible data
        guard let outputData = outputBuffer.cpuBuffer else {
            logger.error("Output buffer has no CPU-accessible data")
            return false
        }
        
        // Perform the FFT
        let success = performFFT(
            input: inputData.assumingMemoryBound(to: Float.self),
            inputSize: inputBuffer.size / MemoryLayout<Float>.size,
            output: outputData.assumingMemoryBound(to: Float.self),
            outputSize: outputBuffer.size / MemoryLayout<Float>.size,
            format: context.format
        )
        
        // If metal buffer is used, make sure to synchronize
        if outputBuffer.metalBuffer != nil {
            try context.bufferManager.synchronizeBuffer(id: outputBuffers[0])
        }
        
        return success
    }
    
    /// Reset the state of this node
    public func reset() {
        // Clear all FFT data
        self.realInput = [Float](repeating: 0, count: self.fftSize)
        self.imagInput = [Float](repeating: 0, count: self.fftSize)
        self.realOutput = [Float](repeating: 0, count: self.fftSize)
        self.imagOutput = [Float](repeating: 0, count: self.fftSize)
        self.magnitude = [Float](repeating: 0, count: self.fftSize/2)
        
        // Reset frequency band levels
        self.bassLevel = 0.0
        self.midLevel = 0.0
        self.trebleLevel = 0.0
        
        logger.debug("FFT node reset: id=\(id)")
    }
    
    // MARK: - FFT Implementation
    
    /// Creates the window function buffer based on the selected window type
    private func createWindow() {
        switch windowFunction {
        case .hann:
            vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        case .hamming:
            vDSP_hamm_window(&window, vDSP_Length(fftSize), 0)
        case .blackman:
            vDSP_blkman_window(&window, vDSP_Length(fftSize), 0)
        case .none:
            // No window function - use rectangular window (all 1's)
            for i in 0..<fftSize {
                window[i] = 1.0
            }
        }
    }
    
    /// Performs FFT on the provided audio data
    /// - Parameters:
    ///   - input: Pointer to input audio samples
    ///   - inputSize: Number of samples in the input buffer
    ///   - output: Pointer to output buffer for FFT results
    ///   - outputSize: Size of the output buffer
    ///   - format: Audio format of the input data
    /// - Returns: Whether the FFT was successful
    private func performFFT(
        input: UnsafePointer<Float>,
        inputSize: Int,
        output: UnsafeMutablePointer<Float>,
        outputSize: Int,
        format: AVAudioFormat
    ) -> Bool {
        guard let fftSetup = fftSetup,
              let realPtr = realPtr,
              let imagPtr = imagPtr else {
            logger.error("FFT setup is not initialized")
            return false
        }
        
        // Check buffer sizes
        let outputElementCount = min(outputSize, fftSize / 2)
        guard outputElementCount > 0 else {
            logger.error("Output buffer too small: size=\(outputSize)")
            return false
        }
        
        // Extract samples from the input buffer
        let count = min(inputSize, fftSize)
        for i in 0..<count {
            realInput[i] = input[i] * window[i]  // Apply window function
        }
        
        // Zero-padding if needed
        if count < fftSize {
            for i in count..<fftSize {
                realInput[i] = 0
            }
        }
        
        // Clear imaginary input (we're analyzing real signals)
        for i in 0..<fftSize {
            imagInput[i] = 0
        }
        
        // Perform the FFT
        vDSP_DFT_Execute(
            fftSetup,
            realInput, imagInput,
            &realOutput, &imagOutput
        )
        
        // Copy output to our persistent buffers
        for i in 0..<fftSize {
            realPtr[i] = realOutput[i]
            imagPtr[i] = imagOutput[i]
        }
        
        // Compute magnitude using the proper approach for complex numbers
        for i in 0..<fftSize/2 {
            let real = realPtr[i]
            let imag = imagPtr[i]
            magnitude[i] = sqrt(real * real + imag * imag)
        }
        
        // Scale the magnitudes (normalize)
        var scale = Float(1.0 / Float(fftSize))
        vDSP_vsmul(magnitude, 1, &scale, &magnitude, 1, vDSP_Length(fftSize/2))
        
        // Apply logarithmic scaling for better visualization
        var scaledMagnitude = [Float](repeating: 0, count: fftSize/2)
        for i in 0..<fftSize/2 {
            // Convert to dB with some scaling and clamping
            let logValue = 10.0 * log10f(magnitude[i] + 1e-9)
            // Normalize to 0.0-1.0 range
            let normalizedValue = (logValue + 90.0) / 90.0
            scaledMagnitude[i] = min(max(normalizedValue, 0.0), 1.0)
        }
        
        // Copy the results to the output buffer
        for i in 0..<outputElementCount {
            output[i] = scaledMagnitude[i]
        }
        
        // Update frequency band levels
        calculateFrequencyBands(scaledMagnitude: scaledMagnitude, format: format)
        
        return true
    }
    
    /// Calculates frequency band levels from FFT data
    /// - Parameters:
    ///   - scaledMagnitude: The scaled magnitude values from FFT
    ///   - format: The audio format
    private func calculateFrequencyBands(scaledMagnitude: [Float], format: AVAudioFormat) {
        // Reset levels
        var bassSum: Float = 0.0
        var midSum: Float = 0.0
        var trebleSum: Float = 0.0
        
        var bassCount: Int = 0
        var midCount: Int = 0
        var trebleCount: Int = 0
        
        // Calculate the frequency resolution (Hz per bin)
        let binWidth = Float(format.sampleRate) / Float(fftSize)
        
        // Sum magnitudes in each frequency band
        for i in 0..<min(scaledMagnitude.count, fftSize/2) {
            let frequency = Float(i) * binWidth
            
            if bassRange.contains(frequency) {
                bassSum += scaledMagnitude[i]
                bassCount += 1
            } else if midRange.contains(frequency) {
                midSum += scaledMagnitude[i]
                midCount += 1
            } else if trebleRange.contains(frequency) {
                trebleSum += scaledMagnitude[i]
                trebleCount += 1
            }
        }
        
        // Calculate average levels, avoiding division by zero
        bassLevel = bassCount > 0 ? bassSum / Float(bassCount) : 0.0
        midLevel = midCount > 0 ? midSum / Float(midCount) : 0.0
        trebleLevel = trebleCount > 0 ? trebleSum / Float(trebleCount) : 0.0
        
        // Apply some smoothing to avoid rapid fluctuations
        let smoothingFactor: Float = 0.7
        bassLevel = smoothingFactor * bassLevel + (1.0 - smoothingFactor) * bassLevel
        midLevel = smoothingFactor * midLevel + (1.0 - smoothingFactor) * midLevel
        trebleLevel = smoothingFactor * trebleLevel + (1.0 - smoothingFactor) * trebleLevel
        
        // Clamp values to 0.0-1.0 range
        bassLevel = min(max(bassLevel, 0.0), 1.0)
        midLevel = min(max(midLevel, 0.0), 1.0)
        trebleLevel = min(max(trebleLevel, 0.0), 1.0)
    }
    
    /// Gets frequency band levels
    /// - Returns: Tuple containing bass, mid, and treble levels
    public func getFrequencyBandLevels() -> (bass: Float, mid: Float, treble: Float) {
        return (bass: bassLevel, mid: midLevel, treble: trebleLevel)
    }
    
    /// Converts bin index to frequency in Hz
    /// - Parameters:
    ///   - binIndex: The FFT bin index
    ///   - sampleRate: The audio sample rate
    /// - Returns: The frequency in Hz
    private func binIndexToFrequency(binIndex: Int, sampleRate: Double) -> Float {
        return Float(binIndex) * Float(sampleRate) / Float(fftSize)
    }
    
    /// Finds the bin index for a given frequency
    /// - Parameters:
    ///   - frequency: The frequency in Hz
    ///   - sampleRate: The audio sample rate
    /// - Returns: The closest bin index
    private func frequencyToBinIndex(frequency: Float, sampleRate: Double) -> Int {
        let index = Int(Float(fftSize) * frequency / Float(sampleRate))
        return min(max(0, index), fftSize/2 - 1)
    }
}
