import Cocoa

class WindowController: NSWindowController {
    
    override func windowDidLoad() {
        super.windowDidLoad()
        
        // Set window title based on environment
        if let environmentName = Bundle.main.object(forInfoDictionaryKey: "EnvironmentName") as? String {
            if environmentName != "production" {
                window?.title = "\(window?.title ?? "AudioBloomAI") (\(environmentName))"
            }
        }
        
        // Configure window appearance
        window?.appearance = NSAppearance(named: .darkAqua)
        window?.backgroundColor = NSColor.black
        
        // Set minimum window size
        window?.minSize = NSSize(width: 800, height: 600)
        
        // Add custom toolbar items if needed
        configureToolbar()
    }
    
    private func configureToolbar() {
        guard let window = window else { return }
        
        // Create a toolbar
        let toolbar = NSToolbar(identifier: "AudioBloomToolbar")
        toolbar.displayMode = .iconAndLabel
        toolbar.delegate = self
        toolbar.allowsUserCustomization = true
        toolbar.autosavesConfiguration = true
        
        window.toolbar = toolbar
    }
}

// MARK: - NSToolbarDelegate
extension WindowController: NSToolbarDelegate {
    
    enum ToolbarItemIdentifier: String {
        case play = "com.audiobloom.toolbar.play"
        case open = "com.audiobloom.toolbar.open"
        case export = "com.audiobloom.toolbar.export"
        case settings = "com.audiobloom.toolbar.settings"
        
        var identifier: NSToolbarItem.Identifier {
            return NSToolbarItem.Identifier(rawValue)
        }
    }
    
    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        return [
            ToolbarItemIdentifier.open.identifier,
            ToolbarItemIdentifier.play.identifier,
            NSToolbarItem.Identifier.flexibleSpace,
            ToolbarItemIdentifier.export.identifier,
            ToolbarItemIdentifier.settings.identifier
        ]
    }
    
    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        return [
            ToolbarItemIdentifier.open.identifier,
            ToolbarItemIdentifier.play.identifier,
            ToolbarItemIdentifier.export.identifier,
            ToolbarItemIdentifier.settings.identifier,
            NSToolbarItem.Identifier.flexibleSpace,
            NSToolbarItem.Identifier.space
        ]
    }
    
    func toolbar(_ toolbar: NSToolbar, itemForItemIdentifier itemIdentifier: NSToolbarItem.Identifier, willBeInsertedIntoToolbar flag: Bool) -> NSToolbarItem? {
        // Create toolbar items based on identifiers
        if itemIdentifier.rawValue == ToolbarItemIdentifier.play.rawValue {
            let toolbarItem = NSToolbarItem(itemIdentifier: itemIdentifier)
            toolbarItem.label = "Play/Pause"
            toolbarItem.paletteLabel = "Play/Pause"
            toolbarItem.toolTip = "Play or pause audio visualization"
            toolbarItem.image = NSImage(systemSymbolName: "play.fill", accessibilityDescription: "Play")
            toolbarItem.action = #selector(playPauseAction(_:))
            return toolbarItem
            
        } else if itemIdentifier.rawValue == ToolbarItemIdentifier.open.rawValue {
            let toolbarItem = NSToolbarItem(itemIdentifier: itemIdentifier)
            toolbarItem.label = "Open"
            toolbarItem.paletteLabel = "Open Audio"
            toolbarItem.toolTip = "Open an audio file"
            toolbarItem.image = NSImage(systemSymbolName: "folder", accessibilityDescription: "Open")
            toolbarItem.action = #selector(openFileAction(_:))
            return toolbarItem
            
        } else if itemIdentifier.rawValue == ToolbarItemIdentifier.export.rawValue {
            let toolbarItem = NSToolbarItem(itemIdentifier: itemIdentifier)
            toolbarItem.label = "Export"
            toolbarItem.paletteLabel = "Export Visualization"
            toolbarItem.toolTip = "Export the current visualization"
            toolbarItem.image = NSImage(systemSymbolName: "square.and.arrow.up", accessibilityDescription: "Export")
            toolbarItem.action = #selector(exportAction(_:))
            return toolbarItem
            
        } else if itemIdentifier.rawValue == ToolbarItemIdentifier.settings.rawValue {
            let toolbarItem = NSToolbarItem(itemIdentifier: itemIdentifier)
            toolbarItem.label = "Settings"
            toolbarItem.paletteLabel = "Settings"
            toolbarItem.toolTip = "Adjust visualization settings"
            toolbarItem.image = NSImage(systemSymbolName: "gear", accessibilityDescription: "Settings")
            toolbarItem.action = #selector(settingsAction(_:))
            return toolbarItem
        }
        
        return nil
    }
    
    // MARK: - Toolbar Actions
    @objc func playPauseAction(_ sender: Any) {
        // Handle play/pause functionality
        print("Play/Pause action triggered")
    }
    
    @objc func openFileAction(_ sender: Any) {
        // Open file dialog
        let openPanel = NSOpenPanel()
        openPanel.title = "Open Audio File"
        openPanel.allowsMultipleSelection = false
        openPanel.canChooseDirectories = false
        openPanel.canCreateDirectories = false
        openPanel.allowedFileTypes = ["mp3", "wav", "aac", "flac", "m4a"]
        
        openPanel.beginSheetModal(for: window!) { response in
            if response == .OK, let url = openPanel.url {
                print("Selected file: \(url.path)")
                // Process the selected file
            }
        }
    }
    
    @objc func exportAction(_ sender: Any) {
        // Handle export functionality
        print("Export action triggered")
    }
    
    @objc func settingsAction(_ sender: Any) {
        // Show settings panel
        print("Settings action triggered")
    }
}

