#!/usr/bin/env swift

import Foundation

// MARK: - Configuration

struct ModernizationConfig {
    // Project paths
    let sourcesPath = "Sources"
    let testsPath = "Tests"
    
    // Swift version
    let swiftVersion = "6.0"
    
    // macOS version
    let macOSVersion = "15.0"
    
    // Flag to enable backup files
    let createBackups = true
    
    // List of files to exclude from processing
    let excludedFiles = [
        "Package.swift"
    ]
    
    // List of directories to exclude from processing
    let excludedDirs = [
        ".git",
        ".build",
        "AudioBloomAI.wiki"
    ]
}

// MARK: - File Processing

class SwiftFileModernizer {
    let config = ModernizationConfig()
    
    // Process a single Swift file
    func processFile(at path: String) throws {
        print("Processing: \(path)")
        
        // Check if file should be excluded
        if config.excludedFiles.contains(where: { path.hasSuffix($0) }) {
            print("  Skipping excluded file")
            return
        }
        
        // Read file contents
        guard var content = try? String(contentsOfFile: path, encoding: .utf8) else {
            print("  Failed to read file")
            return
        }
        
        // Create backup if enabled
        if config.createBackups {
            try? content.write(toFile: path + ".bak", atomically: true, encoding: .utf8)
        }
        
        // Apply transformations
        content = addAvailabilityAttribute(to: content)
        content = updateConcurrencyAnnotations(in: content)
        content = addActorIsolation(to: content)
        content = updateDocumentation(in: content)
        content = applySwift6Optimizations(to: content)
        
        // Write modified content back to file
        try? content.write(toFile: path, atomically: true, encoding: .utf8)
        print("  Updated successfully")
    }
    
    // Walk through directory and process Swift files
    func processDirectory(at path: String) throws {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(atPath: path)
        
        for item in contents {
            let itemPath = path + "/" + item
            
            // Skip excluded directories
            if config.excludedDirs.contains(where: { item.contains($0) }) {
                continue
            }
            
            var isDir: ObjCBool = false
            if fileManager.fileExists(atPath: itemPath, isDirectory: &isDir) {
                if isDir.boolValue {
                    try processDirectory(at: itemPath)
                } else if item.hasSuffix(".swift") {
                    try processFile(at: itemPath)
                }
            }
        }
    }
    
    // MARK: - Content Transformations
    
    // Add @available attribute to classes, structs, actors, and enums
    func addAvailabilityAttribute(to content: String) -> String {
        var lines = content.components(separatedBy: .newlines)
        var inComment = false
        var i = 0
        
        while i < lines.count {
            let line = lines[i]
            
            // Skip comment blocks
            if line.contains("/**") || line.contains("/*") {
                inComment = true
            }
            if line.contains("*/") {
                inComment = false
                continue
            }
            if inComment {
                continue
            }
            
            // Add @available attribute to type declarations
            if line.contains("class ") || line.contains("struct ") || 
               line.contains("enum ") || line.contains("protocol ") {
                // Only add if doesn't already have an @available attribute
                if !lines[max(0, i-1)].contains("@available") {
                    let lineWithIndentation = line.prefix(while: { $0 == " " || $0 == "\t" })
                    lines.insert("\(lineWithIndentation)@available(macOS \(config.macOSVersion), *)", at: i)
                    // Skip the inserted line
                    i += 1
                }
            }
            i += 1
        }
        
        return lines.joined(separator: "\n")
    }
    
    // Update concurrency annotations (async, await, Task)
    func updateConcurrencyAnnotations(in content: String) -> String {
        // Handle potential async/await issues
        var updatedContent = content
        
        // Add Sendable to appropriate types
        updatedContent = updatedContent.replacingOccurrences(
            of: "struct ([A-Za-z0-9_]+)( *[:{])",
            with: "struct $1: Sendable$2",
            options: .regularExpression
        )
        
        // Fix common concurrency patterns
        updatedContent = updatedContent.replacingOccurrences(
            of: "DispatchQueue.main.async \\{([^}]*)\\}",
            with: "Task { @MainActor in$1}",
            options: .regularExpression
        )
        
        // Add proper Task group handling
        updatedContent = updatedContent.replacingOccurrences(
            of: "func ([A-Za-z0-9_]+)\\(([^)]*)\\)( *-> *[A-Za-z0-9_<>,\\s]+)? *\\{\\s*Task",
            with: "func $1($2)$3 async {\n        Task",
            options: .regularExpression
        )
        
        return updatedContent
    }
    
    // Add actor isolation where appropriate
    func addActorIsolation(to content: String) -> String {
        var updatedContent = content
        
        // Convert classes with concurrency to actors where appropriate
        if updatedContent.contains("class") && 
           (updatedContent.contains("async") || updatedContent.contains("await")) {
            
            // Check for classes that manage state and use locks - candidates for actors
            if updatedContent.contains("NSLock") || 
               updatedContent.contains("DispatchQueue") && 
               updatedContent.contains("weak var") {
                
                // Convert to actor
                updatedContent = updatedContent.replacingOccurrences(
                    of: "class ([A-Za-z0-9_]+)( *: *[^{]+)?\\s*\\{",
                    with: "actor $1$2 {\n    // Converted to actor in Swift 6 for thread safety",
                    options: .regularExpression
                )
                
                // Remove lock objects
                updatedContent = updatedContent.replacingOccurrences(
                    of: "\\s+private let lock = NSLock\\(\\).*?\n",
                    with: "\n    // Lock removed - actor provides automatic isolation\n",
                    options: .regularExpression
                )
                
                // Remove lock/unlock calls
                updatedContent = updatedContent.replacingOccurrences(
                    of: "\\s+lock\\.lock\\(\\)\\s+",
                    with: " ",
                    options: .regularExpression
                )
                updatedContent = updatedContent.replacingOccurrences(
                    of: "\\s+lock\\.unlock\\(\\)\\s+",
                    with: " ",
                    options: .regularExpression
                )
            }
        }
        
        // Add @MainActor to UI-related classes
        if updatedContent.contains("import SwiftUI") || 
           updatedContent.contains("import UIKit") {
            updatedContent = updatedContent.replacingOccurrences(
                of: "class ([A-Za-z0-9_]+)( *: *[^{]+)?\\s*\\{",
                with: "@MainActor\nclass $1$2 {\n    // Added @MainActor for UI thread safety",
                options: .regularExpression
            )
        }
        
        return updatedContent
    }
    
    // Update documentation for Swift 6 features
    func updateDocumentation(in content: String) -> String {
        var updatedContent = content
        
        // Add Swift 6 version note to file headers
        if updatedContent.hasPrefix("//") || updatedContent.hasPrefix("import") {
            let header = """
                // Swift 6 optimized implementation
                // Requires macOS 15.0 or later
                // Updated for modern concurrency

                """
                
            if !updatedContent.contains("Swift 6") {
                updatedContent = header + updatedContent
            }
        }
        
        // Update doc comments to mention Swift 6 features where relevant
        if updatedContent.contains("actor") {
            updatedContent = updatedContent.replacingOccurrences(
                of: "/// ([^\n]+)",
                with: "/// $1\n/// Uses Swift 6 actor isolation for thread safety.",
                options: .regularExpression
            )
        }
        
        return updatedContent
    }
    
    // Apply Swift 6 specific optimizations
    func applySwift6Optimizations(to content: String) -> String {
        var updatedContent = content
        
        // Update to new function builder syntax
        updatedContent = updatedContent.replacingOccurrences(
            of: "@ViewBuilder func",
            with: "@ViewBuilder(resultType: some View) func",
            options: .regularExpression
        )
        
        // Fix potential issues with optional chaining
        updatedContent = updatedContent.replacingOccurrences(
            of: "if let ([a-zA-Z0-9_]+) = ([a-zA-Z0-9_]+)\\?.([a-zA-Z0-9_]+),",
            with: "if let $1 = $2?.$3,",
            options: .regularExpression
        )
        
        return updatedContent
    }
}

// MARK: - Script Entry Point

let modernizer = SwiftFileModernizer()
let projectPath = FileManager.default.currentDirectoryPath
let sourcesPath = projectPath + "/" + modernizer.config.sourcesPath
let testsPath = projectPath + "/" + modernizer.config.testsPath

print("AudioBloomAI Swift 6 Modernization Script")
print("----------------------------------------")
print("Swift Version: \(modernizer.config.swiftVersion)")
print("macOS Version: \(modernizer.config.macOSVersion)")
print("----------------------------------------")

do {
    print("Processing source files...")
    try modernizer.processDirectory(at: sourcesPath)
    
    print("Processing test files...")
    try modernizer.processDirectory(at: testsPath)
    
    print("Modernization complete!")
} catch {
    print("Error: \(error)")
}

