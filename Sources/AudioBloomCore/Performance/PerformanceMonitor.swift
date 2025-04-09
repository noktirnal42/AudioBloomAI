import Foundation
import Metal
import MetalPerformanceShaders
import QuartzCore
import Combine

/// Performance metric types tracked by the monitor
public enum PerformanceMetric {
    case fps                  // Frames per second
    case cpuUsage             // CPU utilization percentage
    case gpuUsage             // GPU utilization percentage
    case memoryUsage          // Memory usage in MB
    case audioBufferUsage     // Audio buffer memory usage in MB
    case shaderCompileTime    // Shader compilation time in ms
    case renderTime           // Frame render time in ms
    case audioProcessingTime  // Audio processing time in ms
}

/// Quality level settings that can be automatically adjusted
public enum QualityLevel: Int, CaseIterable {
    case low = 0
    case medium = 1
    case high = 2
    case ultra = 3
    
    public var description: String {
        switch self {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .ultra: return "Ultra"
        }
    }
}

/// Performance warning level
public enum PerformanceWarningLevel: Int {
    case none = 0
    case low = 1
    case medium = 2
    case high = 3
    
    public var description: String {
        switch self {
        case .none: return "None"
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        }
    }
}

/// Manages performance monitoring and optimization for AudioBloom
public class PerformanceMonitor: ObservableObject {
    // MARK: - Published Properties
    
    /// Current frames per second
    @Published public private(set) var fps: Double = 0
    
    /// CPU usage percentage
    @Published public private(set) var cpuUsage: Double = 0
    
    /// GPU usage percentage (if available)
    @Published public private(set) var gpuUsage: Double = 0
    
    /// Total memory usage in megabytes
    @Published public private(set) var memoryUsage: Double = 0
    
    /// Audio buffer memory usage in megabytes
    @Published public private(set) var audioBufferUsage: Double = 0
    
    /// Current quality level
    @Published public private(set) var currentQualityLevel: QualityLevel = .high
    
    /// Auto optimization enabled flag
    @Published public var autoOptimizationEnabled: Bool = true {
        didSet {
            UserDefaults.standard.set(autoOptimizationEnabled, forKey: "autoOptimizationEnabled")
        }
    }
    
    /// Current warning level
    @Published public private(set) var warningLevel: PerformanceWarningLevel = .none
    
    /// Most recent performance snapshot
    @Published public private(set) var performanceSnapshot: [PerformanceMetric: Double] = [:]
    
    /// User-selected maximum quality level
    @Published public var maxQualityLevel: QualityLevel = .ultra {
        didSet {
            UserDefaults.standard.set(maxQualityLevel.rawValue, forKey: "maxQualityLevel")
            if currentQualityLevel.rawValue > maxQualityLevel.rawValue {
                self.currentQualityLevel = maxQualityLevel
                qualityLevelDidChange.send(currentQualityLevel)
            }
        }
    }
    
    // MARK: - Publishers
    
    /// Publisher for quality level changes
    public let qualityLevelDidChange = PassthroughSubject<QualityLevel, Never>()
    
    /// Publisher for performance warnings
    public let performanceWarningDidChange = PassthroughSubject<PerformanceWarningLevel, Never>()
    
    // MARK: - Private Properties
    
    /// Metal device for performance monitoring
    private let device: MTLDevice?
    
    /// GPU counter set
    private var counterSet: MTLCounterSet?
    
    /// CADisplayLink for frame timing
    private var displayLink: CADisplayLink?
    
    /// Performance history for tracking trends
    private var metricHistory: [PerformanceMetric: [Double]] = [:]
    
    /// Maximum history size for performance metrics
    private let maxHistorySize = 60 // Approximately 1 second at 60fps
    
    /// Last timestamp for fps calculation
    private var lastFrameTimestamp: CFTimeInterval = 0
    
    /// Optimization timer to periodically adjust quality settings
    private var optimizationTimer: Timer?
    
    /// Sampling timestamps for render timing
    private var renderTimings: [String: CFTimeInterval] = [:]
    
    /// Queue for performance monitoring tasks
    private let performanceQueue = DispatchQueue(label: "com.audiobloom.performanceMonitor", qos: .utility)
    
    /// Process info for CPU stats
    private let processInfo = ProcessInfo.processInfo
    
    /// Flag to check if running on M-series hardware
    private let isAppleSilicon: Bool
    
    /// Type of Apple Silicon (M1, M2, M3, etc.)
    private let chipType: String
    
    /// Metal performance sampler
    private var sampler: MTLCaptureManager?
    
    /// Counter sample buffer
    private var counterSampleBuffer: MTLCounterSampleBuffer?
    
    /// Audio buffer memory pool
    private var audioBufferMemoryPool: [UUID: Int] = [:]
    
    /// Last optimization timestamp
    private var lastOptimizationTime: CFTimeInterval = 0
    
    // Frame time history for smoothing
    private var frameTimeHistory: [CFTimeInterval] = []
    private let frameTimeHistoryMaxSize = 10
    
    // MARK: - Initialization
    
    /// Initialize performance monitor
    /// - Parameter metalDevice: Metal device to monitor
    public init(metalDevice: MTLDevice?) {
        self.device = metalDevice
        
        // Detect Apple Silicon
        let systemInfo = Host.current().localizedName ?? ""
        self.isAppleSilicon = systemInfo.contains("Apple") || systemInfo.contains("M1") || systemInfo.contains("M2") || systemInfo.contains("M3")
        
        // Determine chip type
        if systemInfo.contains("M3") {
            self.chipType = "M3"
        } else if systemInfo.contains("M2") {
            self.chipType = "M2"
        } else if systemInfo.contains("M1") {
            self.chipType = "M1"
        } else {
            self.chipType = "Unknown"
        }
        
        // Initialize metric history
        PerformanceMetric.allCases.forEach { metric in
            metricHistory[metric] = []
        }
        
        // Load user preferences
        self.autoOptimizationEnabled = UserDefaults.standard.bool(forKey: "autoOptimizationEnabled")
        if let savedQualityLevel = QualityLevel(rawValue: UserDefaults.standard.integer(forKey: "maxQualityLevel")) {
            self.maxQualityLevel = savedQualityLevel
        }
        
        setupCounters()
        startMonitoring()
    }
    
    deinit {
        stopMonitoring()
    }
    
    // MARK: - Public Methods
    
    /// Start performance monitoring
    public func startMonitoring() {
        if displayLink == nil {
            displayLink = CADisplayLink(target: self, selector: #selector(displayLinkDidFire))
            displayLink?.add(to: .main, forMode: .common)
        }
        
        if optimizationTimer == nil && autoOptimizationEnabled {
            optimizationTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
                self?.performOptimization()
            }
        }
    }
    
    /// Stop performance monitoring
    public func stopMonitoring() {
        displayLink?.invalidate()
        displayLink = nil
        
        optimizationTimer?.invalidate()
        optimizationTimer = nil
    }
    
    /// Manually adjust quality level
    /// - Parameter level: Desired quality level
    public func setQualityLevel(_ level: QualityLevel) {
        guard level.rawValue <= maxQualityLevel.rawValue else { return }
        
        if currentQualityLevel != level {
            currentQualityLevel = level
            qualityLevelDidChange.send(level)
        }
    }
    
    /// Start measuring performance for specific operation
    /// - Parameter operation: Name of the operation being measured
    public func beginMeasuring(_ operation: String) {
        renderTimings[operation] = CACurrentMediaTime()
    }
    
    /// End measuring performance for specific operation
    /// - Parameter operation: Name of the operation that was being measured
    /// - Returns: Duration in milliseconds
    @discardableResult
    public func endMeasuring(_ operation: String) -> Double {
        guard let startTime = renderTimings[operation] else { return 0 }
        
        let duration = (CACurrentMediaTime() - startTime) * 1000 // Convert to milliseconds
        renderTimings.removeValue(forKey: operation)
        
        // Update specific metrics
        switch operation {
        case "ShaderCompilation":
            updateMetric(.shaderCompileTime, value: duration)
        case "RenderFrame":
            updateMetric(.renderTime, value: duration)
        case "AudioProcessing":
            updateMetric(.audioProcessingTime, value: duration)
        default:
            break
        }
        
        return duration
    }
    
    /// Register audio buffer allocation for memory tracking
    /// - Parameters:
    ///   - identifier: Unique ID for the buffer
    ///   - byteSize: Size in bytes
    public func registerAudioBuffer(identifier: UUID, byteSize: Int) {
        performanceQueue.async { [weak self] in
            self?.audioBufferMemoryPool[identifier] = byteSize
            self?.recalculateAudioBufferUsage()
        }
    }
    
    /// Unregister audio buffer when released
    /// - Parameter identifier: Unique ID of the buffer to unregister
    public func unregisterAudioBuffer(identifier: UUID) {
        performanceQueue.async { [weak self] in
            self?.audioBufferMemoryPool.removeValue(forKey: identifier)
            self?.recalculateAudioBufferUsage()
        }
    }
    
    /// Get optimization recommendation for current state
    /// - Returns: Dictionary of recommended optimizations
    public func getOptimizationRecommendations() -> [String: Any] {
        var recommendations: [String: Any] = [:]
        
        // Base recommendations on current performance metrics
        if fps < 40 {
            recommendations["reduceQuality"] = true
            recommendations["suggestedQualityLevel"] = min(currentQualityLevel.rawValue - 1, 0)
        }
        
        if memoryUsage > 1000 {
            recommendations["reduceBufferSize"] = true
            recommendations["suggestedBufferReduction"] = 0.5
        }
        
        if cpuUsage > 80 {
            recommendations["offloadToGPU"] = true
        }
        
        return recommendations
    }
    
    /// Get performance report for the current session
    /// - Returns: Formatted performance report
    public func generatePerformanceReport() -> String {
        var report = "AudioBloom Performance Report\n"
        report += "===============================\n"
        report += "Date: \(Date())\n"
        report += "Device: \(chipType)\n\n"
        
        report += "Performance Metrics (Average):\n"
        report += "- FPS: \(String(format: "%.1f", averageForMetric(.fps)))\n"
        report += "- CPU: \(String(format: "%.1f", averageForMetric(.cpuUsage)))%\n"
        report += "- GPU: \(String(format: "%.1f", averageForMetric(.gpuUsage)))%\n"
        report += "- Memory: \(String(format: "%.1f", averageForMetric(.memoryUsage))) MB\n"
        report += "- Audio Memory: \(String(format: "%.1f", averageForMetric(.audioBufferUsage))) MB\n"
        report += "- Render Time: \(String(format: "%.2f", averageForMetric(.renderTime))) ms\n"
        report += "- Audio Processing Time: \(String(format: "%.2f", averageForMetric(.audioProcessingTime))) ms\n\n"
        
        report += "Quality Settings:\n"
        report += "- Current Quality: \(currentQualityLevel.description)\n"
        report += "- Max Quality: \(maxQualityLevel.description)\n"
        report += "- Auto-Optimization: \(autoOptimizationEnabled ? "Enabled" : "Disabled")\n\n"
        
        report += "Warning Level: \(warningLevel.description)\n"
        
        return report
    }
    
    /// Reset all performance metrics
    public func resetMetrics() {
        performanceQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.metricHistory.removeAll()
            PerformanceMetric.allCases.forEach { metric in
                self.metricHistory[metric] = []
            }
            self.frameTimeHistory.removeAll()
            
            DispatchQueue.main.async {
                self.fps = 0
                self.cpuUsage = 0
                self.gpuUsage = 0
                self.memoryUsage = 0
                self.audioBufferUsage = 0
                self.performanceSnapshot = [:]
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func setupCounters() {
        guard let device = device else { return }
        
        if #available(macOS 10.15, iOS 13.0, *) {
            counterSet = device.counterSets.first
            sampler = MTLCaptureManager.shared()
        }
    }
    
    @objc private func displayLinkDidFire(_ link: CADisplayLink) {
        // Calculate frame time
        let currentTime = link.timestamp
        let frameTime = currentTime - lastFrameTimestamp
        lastFrameTimestamp = currentTime
        
        if frameTime > 0 {
            // Add to frame time history for smoothing
            frameTimeHistory.append(frameTime)
            if frameTimeHistory.count > frameTimeHistoryMaxSize {
                frameTimeHistory.removeFirst()
            }
            
            // Calculate average frame time
            let avgFrameTime = frameTimeHistory.reduce(0, +) / Double(frameTimeHistory.count)
            let currentFPS = 1.0 / avgFrameTime
            
            updateMetric(.fps, value: currentFPS)
        }
        
        // Only sample other metrics every 10 frames to reduce overhead
        if Int(fps) % 10 == 0 {
            performanceQueue.async { [weak self] in
                self?.samplePerformanceMetrics()
            }
        }
    }
    
    private func updateMetric(_ metric: PerformanceMetric, value: Double) {
        performanceQueue.async { [weak self] in
            guard let self = self else

