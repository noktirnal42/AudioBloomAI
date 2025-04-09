import SwiftUI
import AppKit
import Foundation

// Creates directory for resources if it doesn't exist
func ensureResourcesDirectoryExists() {
    let fileManager = FileManager.default
    let resourcesPath = "Resources"
    
    if !fileManager.fileExists(atPath: resourcesPath) {
        do {
            try fileManager.createDirectory(atPath: resourcesPath, withIntermediateDirectories: true, attributes: nil)
            print("Created Resources directory")
        } catch {
            print("Error creating Resources directory: \(error)")
        }
    }
}

// Ensure Resources directory exists
ensureResourcesDirectoryExists()

print("Starting logo generation...")

// Run the async task to generate the logo
Task {
    await generateLogo()
    exit(0)
}

// Keep the process running until the task completes
RunLoop.main.run()
