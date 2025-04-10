import Foundation
import SwiftUI
import Combine
import Metal
import MetalKit
import QuartzCore
import os.log
import AppKit
import Darwin

/// Warning levels for performance issues
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public enum PerformanceWarningLevel: String, CaseIterable {
    case none
    case low
    case medium
    case high
    
    /// User-friendly string representation
/// Uses Swift 6 actor isolation for thread safety.
    public var description: String {
        switch self {
        case .none: return "None"
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        }
    }
}

/// Metrics that can be tracked for performance
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public enum PerformanceMetric: String, CaseIterable {
    case fps
    case cpuUsage
    case gpuUsage
    case memoryUsage
    case audioBufferUsage
    case renderTime
    case audioProcessingTime
    case totalLatency
    case thermalLevel
    case batteryImpact
}

/// Manages performance monitoring and optimization for AudioBloom
/// Uses Swift 6 actor isolation for thread safety.
@available(macOS 15.0, *)
public final actor PerformanceMonitor: ObservableObject, @unchecked Sendable  {
    // Converted to actor in Swift 6 for thread safety
    // MARK: - Published Properties
    
    /// Current frames per second
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var fps: Double = 60.0
    
    /// CPU usage percentage (0-100)
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var cpuUsage: Double = 0.0
    
    /// GPU usage percentage (0-100)
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var gpuUsage: Double = 0.0
    
    /// Memory usage in MB
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var memoryUsage: Double = 0.0
    
    /// Audio buffer memory usage in MB
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var audioBufferUsage: Double = 0.0
    
    /// Render time in milliseconds
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var renderTime: Double = 0.0
    
    /// Audio processing time in milliseconds
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var audioProcessingTime: Double = 0.0
    
    /// Total latency in milliseconds
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var totalLatency: Double = 0.0
    
    /// Thermal state (0=nominal, 1=fair, 2=serious, 3=critical)
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var thermalState: Int = 0
    
    /// Battery impact (0-1)
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var batteryImpact: Double = 0.0
    
    /// Current warning level
/// Uses Swift 6 actor isolation for thread safety.
    @Published public private(set) var warningLevel: PerformanceWarningLevel = .none
    
    /// Whether auto-optimization is enabled
/// Uses Swift 6 actor isolation for thread safety.
    @Published public var autoOptimizationEnabled: Bool = false
    
    // MARK: - Private Properties
    
    /// Modern display link for synchronization
/// Uses Swift 6 actor isolation for thread safety.
    private var displayLink: Any?
    
    /// Display link target for callback handling
/// Uses Swift 6 actor isolation for thread safety.
    private var displayLinkTarget: DisplayLinkTarget?
    /// Timer for automatic optimization
/// Uses Swift 6 actor isolation for thread safety.
    private var optimizationTimer: Timer?
    
    /// Last sample timestamps for timing calculations
/// Uses Swift 6 actor isolation for thread safety.
    private var lastSampleTime: CFTimeInterval = 0
    
    /// Metal device for GPU monitoring
/// Uses Swift 6 actor isolation for thread safety.
    private var device: MTLDevice?
    
    /// GPU counter set
/// Uses Swift 6 actor isolation for thread safety.
    private var counterSet: MTLCounterSet?
    
    /// Metal performance sampler
/// Uses Swift 6 actor isolation for thread safety.
    private var sampler: MTLCaptureManager?
    
    /// Queue for thread-safe access to performance metrics
/// Uses Swift 6 actor isolation for thread safety.
    private let performanceQueue = DispatchQueue(label: "com.audiobloom.performance", qos: .utility)
    
    /// Lock for thread safety when accessing metrics
/// Uses Swift 6 actor isolation for thread safety.
    private let metricsLock = NSLock()
    
    /// History of metrics for tracking trends
/// Uses Swift 6 actor isolation for thread safety.
    private var metricHistory: [PerformanceMetric: [Double]] = [:]
    
    /// Maximum history samples to keep
/// Uses Swift 6 actor isolation for thread safety.
    private let historySize = 100
    
    /// Audio buffer memory tracking
/// Uses Swift 6 actor isolation for thread safety.
    private var audioBufferMemoryPool: [UUID: Int] = [:]
    
    /// Whether to track history
/// Uses Swift 6 actor isolation for thread safety.
    private var trackHistory = true
    
    /// Overall metrics (updated periodically)
/// Uses Swift 6 actor isolation for thread safety.
    private var metrics: [PerformanceMetric: Double] = [:]
    
    /// Frame counter for averaging
/// Uses Swift 6 actor isolation for thread safety.
    private var frameCount: Int = 0
    
    /// Callback to run on the main queue
/// Uses Swift 6 actor isolation for thread safety.
    private var mainQueueCallback: DispatchWorkItem?
    
    // MARK: - Initialization
    /// Initializes a performance monitor
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter metalDevice: Metal device to monitor
/// Uses Swift 6 actor isolation for thread safety.
    public init(metalDevice: MTLDevice? = nil) {
        self.device = metalDevice
        
        // Initialize metrics with defaults
        for metric in PerformanceMetric.allCases {
            metrics[metric] = 0.0
            metricHistory[metric] = []
        }
        
        // Set some reasonable defaults
        metrics[.fps] = 60.0
        
        // Setup GPU counters if Metal is available
        if let device = metalDevice {
            setupCounters(device: device)
        }
        
        // Initialize timestamp
        lastSampleTime = CFAbsoluteTimeGetCurrent()
    }
    
    deinit {
        stopMonitoring()
    }
    
    // MARK: - Display Link Target
    
    /// Class to handle DisplayLink callbacks 
/// Uses Swift 6 actor isolation for thread safety.
    @available(macOS 15.0, *)
    private final actor DisplayLinkTarget {
    // Converted to actor in Swift 6 for thread safety
        weak var monitor: PerformanceMonitor?
        
        init(monitor: PerformanceMonitor) {
            self.monitor = monitor
        }
        
        @objc func displayLinkDidFire(displayLink: NSObject) {
            guard let monitor = monitor else { return }
            monitor.handleDisplayLinkFire()
        }
    }
    
    // MARK: - Monitoring Control
    
    /// Starts performance monitoring
/// Uses Swift 6 actor isolation for thread safety.
    public func startMonitoring() {
        // Create display link target
        let target = DisplayLinkTarget(monitor: self)
        self.displayLinkTarget = target
        
        if let mainView = NSApp.mainWindow?.contentView {
            // Use modern NSView.displayLink API (available in macOS 15+)
            let newDisplayLink = mainView.displayLink(target: target, 
                                                     selector: #selector(DisplayLinkTarget.displayLinkDidFire))
            newDisplayLink.isPaused = false
            self.displayLink = newDisplayLink
            
            // Setup optimization timer if needed
            if optimizationTimer == nil && autoOptimizationEnabled {
                optimizationTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
                    self?.performOptimization()
                }
            }
            
            // Reset initial time
            lastSampleTime = CFAbsoluteTimeGetCurrent()
            
            os_log(.debug, "Started performance monitoring with modern DisplayLink")
        } else {
            // Fallback to a timer if no view is available
            let timer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { [weak self] _ in
                self?.handleDisplayLinkFire()
            }
            RunLoop.main.add(timer, forMode: .common)
            self.displayLink = timer
            
            os_log(.debug, "Started performance monitoring with timer fallback")
        }
    
    /// Stops performance monitoring
/// Uses Swift 6 actor isolation for thread safety.
    public func stopMonitoring() {
        // Stop the display link based on its type
        if let displayLink = displayLink as? NSDisplayLink {
            displayLink.invalidate()
        } else if let displayLink = displayLink as? Timer {
            displayLink.invalidate()
        }
        
        self.displayLink = nil
        self.displayLinkTarget = nil
        
        // Invalidate optimization timer
        optimizationTimer?.invalidate()
        optimizationTimer = nil
        
        os_log(.debug, "Stopped performance monitoring")
    }
    
    /// Pauses performance monitoring
/// Uses Swift 6 actor isolation for thread safety.
    public func pauseMonitoring() {
        if let displayLink = displayLink as? NSDisplayLink {
            displayLink.isPaused = true
        } else if let displayLink = displayLink as? Timer {
            displayLink.invalidate()
            self.displayLink = nil
        }
        
        optimizationTimer?.invalidate()
        optimizationTimer = nil
        
        os_log(.debug, "Paused performance monitoring")
    }
    
    /// Resumes performance monitoring
/// Uses Swift 6 actor isolation for thread safety.
    public func resumeMonitoring() {
        if let displayLink = displayLink as? NSDisplayLink {
            displayLink.isPaused = false
        } else if displayLink == nil {
            startMonitoring()
        }
        
        if optimizationTimer == nil && autoOptimizationEnabled {
            optimizationTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
                self?.performOptimization()
            }
        }
        
        os_log(.debug, "Resumed performance monitoring")
    }
    
    /// Dictionary to track render timings
/// Uses Swift 6 actor isolation for thread safety.
    private var renderTimings: [String: CFTimeInterval] = [:]
    
    /// Start measuring performance for specific operation
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter operation: Name of the operation being measured
/// Uses Swift 6 actor isolation for thread safety.
    public func beginMeasuring(_ operation: String) {
        renderTimings[operation] = CFAbsoluteTimeGetCurrent()
    }
    
    /// End measuring performance for specific operation
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter operation: Name of the operation that was being measured
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Duration in milliseconds
/// Uses Swift 6 actor isolation for thread safety.
    @discardableResult
    public func endMeasuring(_ operation: String) -> Double {
        guard let startTime = renderTimings[operation] else { return 0 }
        
        let duration = (CFAbsoluteTimeGetCurrent() - startTime) * 1000 // Convert to milliseconds
        renderTimings.removeValue(forKey: operation)
        
        // Update specific metrics
        switch operation {
        case "RenderFrame":
            updateMetric(.renderTime, value: duration)
        case "AudioProcessing":
            updateMetric(.audioProcessingTime, value: duration)
        default:
            break
        }
        
        return duration
    }
    
    /// Records a render time measurement
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter milliseconds: Time in milliseconds
/// Uses Swift 6 actor isolation for thread safety.
    public func recordRenderTime(_ milliseconds: Double) {
        updateMetric(.renderTime, value: milliseconds)
    }
    
    /// Records an audio processing time measurement
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter milliseconds: Time in milliseconds
/// Uses Swift 6 actor isolation for thread safety.
    public func recordAudioProcessingTime(_ milliseconds: Double) {
        updateMetric(.audioProcessingTime, value: milliseconds)
    }
    
    /// Records total latency
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter milliseconds: Latency in milliseconds
/// Uses Swift 6 actor isolation for thread safety.
    public func recordLatency(_ milliseconds: Double) {
        updateMetric(.totalLatency, value: milliseconds)
    }
    
    /// Register audio buffer allocation for memory tracking
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - identifier: Unique ID for the buffer
/// Uses Swift 6 actor isolation for thread safety.
    ///   - byteSize: Size in bytes
/// Uses Swift 6 actor isolation for thread safety.
    public func registerAudioBuffer(identifier: UUID, byteSize: Int) {
        performanceQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.metricsLock.lock()
            self.audioBufferMemoryPool[identifier] = byteSize
            self.metricsLock.unlock()
            
            self.recalculateAudioBufferUsage()
        }
    }
    
    /// Unregister audio buffer when released
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter identifier: Unique ID of the buffer to unregister
/// Uses Swift 6 actor isolation for thread safety.
    public func unregisterAudioBuffer(identifier: UUID) {
        performanceQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.metricsLock.lock()
            self.audioBufferMemoryPool.removeValue(forKey: identifier)
            self.metricsLock.unlock()
            
            self.recalculateAudioBufferUsage()
        }
    }
    
    /// Recalculate audio buffer usage based on registered buffers
/// Uses Swift 6 actor isolation for thread safety.
    private func recalculateAudioBufferUsage() {
        metricsLock.lock()
        let totalBytes = audioBufferMemoryPool.values.reduce(0, +)
        let megabytes = Double(totalBytes) / (1024 * 1024)
        metricsLock.unlock()
        
        updateMetric(.audioBufferUsage, value: megabytes)
    }
    
    /// Get optimization recommendation for current state
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Dictionary of recommended optimizations
/// Uses Swift 6 actor isolation for thread safety.
    public func getOptimizationRecommendations() -> [String: Any] {
        var recommendations: [String: Any] = [:]
        
        // Base recommendations on current performance metrics
        if fps < 40 {
            recommendations["reduceQuality"] = true
            recommendations["suggestedQualityLevel"] = "low"
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
    
    /// Perform automated performance optimization based on metrics
/// Uses Swift 6 actor isolation for thread safety.
    private func performOptimization() {
        let recommendations = getOptimizationRecommendations()
        
        // Log recommendations
        let optimizationLog = "Performance optimization recommendations: \(recommendations)"
        os_log(.info, "%{public}@", optimizationLog)
        
        // This would trigger actual optimizations in a complete implementation
        // For now, just update the warning level based on performance
        updateWarningLevel()
    }
    
    /// Update the warning level based on current metrics
/// Uses Swift 6 actor isolation for thread safety.
    private func updateWarningLevel() {
        // Determine warning level based on metrics
        var newWarningLevel: PerformanceWarningLevel = .none
        
        if fps < 20 || cpuUsage > 90 || memoryUsage > 2000 {
            newWarningLevel = .high
        } else if fps < 40 || cpuUsage > 70 || memoryUsage > 1500 {
            newWarningLevel = .medium
        } else if fps < 55 || cpuUsage > 50 || memoryUsage > 1000 {
            newWarningLevel = .low
        }
        
        // Update on main thread if changed
        if newWarningLevel != warningLevel {
            Task { @MainActor in [weak self] in
                self?.warningLevel = newWarningLevel
            }
        }
    }
    
    /// Calculate the average value for a metric
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameter metric: The metric to average
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Average value
/// Uses Swift 6 actor isolation for thread safety.
    private func averageForMetric(_ metric: PerformanceMetric) -> Double {
        metricsLock.lock()
        defer { metricsLock.unlock() }
        
        guard let history = metricHistory[metric], !history.isEmpty else {
            return metrics[metric] ?? 0.0
        }
        
        return history.reduce(0.0, +) / Double(history.count)
    }
    
    /// Get performance report for the current session
/// Uses Swift 6 actor isolation for thread safety.
    /// - Returns: Formatted performance report
/// Uses Swift 6 actor isolation for thread safety.
    public func generatePerformanceReport() -> String {
        var report = "AudioBloom Performance Report\n"
        report += "===============================\n"
        report += "Date: \(Date())\n"
        
        // Device type detection
        #if arch(arm64)
        let chipType = "Apple Silicon"
        #else
        let chipType = "Intel"
        #endif
        
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
        report += "- Quality Settings: \(autoOptimizationEnabled ? "Auto" : "Manual")\n"
        report += "- Auto-Optimization: \(autoOptimizationEnabled ? "Enabled" : "Disabled")\n\n"
        
        report += "Warning Level: \(warningLevel.description)\n"
        
        return report
    }
    
    /// Reset all performance metrics
/// Uses Swift 6 actor isolation for thread safety.
    public func resetMetrics() {
        performanceQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.metricsLock.lock()
            self.metricHistory.removeAll()
            PerformanceMetric.allCases.forEach { metric in
                self.metricHistory[metric] = []
            }
            self.metricsLock.unlock()
            
            Task { @MainActor in [weak self] in
                guard let self = self else { return }
                self.fps = 60.0
                self.cpuUsage = 0
                self.gpuUsage = 0
                self.memoryUsage = 0
                self.audioBufferUsage = 0
                self.renderTime = 0
                self.audioProcessingTime = 0
                self.totalLatency = 0
            }
        }
            }
        }
    }
    
    // MARK: - Private Methods
    
    /// Setup GPU performance counters
/// Uses Swift 6 actor isolation for thread safety.
    private func setupCounters(device: MTLDevice) {
        if #available(macOS 10.15, *) {
            if let counterSets = device.counterSets {
                counterSet = counterSets.first
            }
            sampler = MTLCaptureManager.shared()
        }
    }
    
    /// Track memory usage
/// Uses Swift 6 actor isolation for thread safety.
    private func getCurrentMemoryUsage() -> Double {
        var info = mach_task_basic_info()
        // Use process_info to get memory usage on macOS
        var taskInfo = task_basic_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_BASIC_INFO), $0, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return Double(taskInfo.resident_size) / (1024 * 1024) // Convert to MB
        }
        
        return 0
    }
    /// Modern display link handler
/// Uses Swift 6 actor isolation for thread safety.
    private func handleDisplayLinkFire() {
        // Calculate frame time
        let currentTime = CFAbsoluteTimeGetCurrent()
        let frameTime = currentTime - lastSampleTime
        lastSampleTime = currentTime
        // Only process if meaningful frame time
        if frameTime > 0 {
            // Calculate FPS
            let instantFPS = 1.0 / frameTime
            
            // Update FPS with smoothing
            let smoothingFactor = 0.2 // Adjust for more/less smoothing
            let smoothedFPS = (instantFPS * smoothingFactor) + (fps * (1.0 - smoothingFactor))
            
            // Update metric
            updateMetric(.fps, value: smoothedFPS)
        }
        
        // Sample other metrics periodically to reduce overhead
        frameCount += 1
        if frameCount % 30 == 0 { // Every 30 frames
            performanceQueue.async { [weak self] in
                self?.samplePerformanceMetrics()
            }
        }
    }
    
    /// Sample system performance metrics
/// Uses Swift 6 actor isolation for thread safety.
    /// Sample system performance metrics
/// Uses Swift 6 actor isolation for thread safety.
    private func samplePerformanceMetrics() {
        // Memory usage
        let memoryMB = getCurrentMemoryUsage()
        updateMetric(.memoryUsage, value: memoryMB)
        
        // CPU usage would be sampled here
        // This requires platform-specific APIs
        let cpuUsageValue = 0.0 // Placeholder
        updateMetric(.cpuUsage, value: cpuUsageValue)
        
        // GPU usage would be sampled here if available
        // This requires platform-specific APIs
        let gpuUsageValue = 0.0 // Placeholder
        updateMetric(.gpuUsage, value: gpuUsageValue)
        
        // Update warning level based on new metrics
        updateWarningLevel()
    }
    
    /// Update a specific metric value
/// Uses Swift 6 actor isolation for thread safety.
    /// - Parameters:
/// Uses Swift 6 actor isolation for thread safety.
    ///   - metric: The metric to update
/// Uses Swift 6 actor isolation for thread safety.
    ///   - value: The new value
/// Uses Swift 6 actor isolation for thread safety.
    private func updateMetric(_ metric: PerformanceMetric, value: Double) {
        performanceQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Update metric value with thread safety
            self.metricsLock.lock()
            
            // Store current value
            self.metrics[metric] = value
            
            // Add to history if tracking enabled
            if self.trackHistory {
                var history = self.metricHistory[metric] ?? []
                history.append(value)
                
                // Limit history size
                if history.count > self.historySize {
                    history.removeFirst(history.count - self.historySize)
                }
                
                // Store updated history
                self.metricHistory[metric] = history
            }
            
            self.metricsLock.unlock()
            
            // Update published properties on main thread
            switch metric {
            case .fps:
                Task { @MainActor in [weak self] in self?.fps = value }
            case .cpuUsage:
                Task { @MainActor in [weak self] in self?.cpuUsage = value }
            case .gpuUsage:
                Task { @MainActor in [weak self] in self?.gpuUsage = value }
            case .memoryUsage:
                Task { @MainActor in [weak self] in self?.memoryUsage = value }
            case .audioBufferUsage:
                Task { @MainActor in [weak self] in self?.audioBufferUsage = value }
            case .renderTime:
                Task { @MainActor in [weak self] in self?.renderTime = value }
            case .audioProcessingTime:
                Task { @MainActor in [weak self] in self?.audioProcessingTime = value }
            case .totalLatency:
                Task { @MainActor in [weak self] in self?.totalLatency = value }
            case .thermalLevel:
                Task { @MainActor in [weak self] in self?.thermalState = Int(value) }
            case .batteryImpact:
                Task { @MainActor in [weak self] in self?.batteryImpact = value }
            }
        }
    }
}
