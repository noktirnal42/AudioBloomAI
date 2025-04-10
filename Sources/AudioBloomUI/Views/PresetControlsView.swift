// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency
import SwiftUI
import AudioBloomCore
import Combine

/// SwiftUI view for managing and applying visualization presets
@available(macOS 15.0, *)
public struct PresetControlsView: Sendable: View {
    /// The preset manager to use
    @ObservedObject private var presetManager: PresetManager
    
    /// The application settings
    @ObservedObject private var settings: AudioBloomSettings
    
    /// View state for new preset creation
    @State private var isCreatingNewPreset = false
    @State private var newPresetName = ""
    @State private var newPresetDescription = ""
    
    /// View state for preset editing
    @State private var isEditingPreset = false
    @State private var editedPreset: Preset?
    
    /// View state for preset import/export
    @State private var isImporting = false
    @State private var isExporting = false
    @State private var selectedPresetsForExport: Set<UUID> = []
    
    /// Filter for preset display
    @State private var categoryFilter: String? = nil
    @State private var searchText = ""
    
    /// Stores subscription to preset changes
    private var cancellables = Set<AnyCancellable>()
    
    /// Initializes a new preset controls view
    /// - Parameters:
    ///   - presetManager: The preset manager to use
    ///   - settings: The application settings
    public init(presetManager: PresetManager, settings: AudioBloomSettings) {
        self.presetManager = presetManager
        self.settings = settings
    }
    
    public var body: some View {
public var body: some View {
        VStack(spacing: 20) {
            // Header with title and actions
            HStack {
                Text("Visualization Presets")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button(action: { isCreatingNewPreset = true }) {
                    Label("New Preset", systemImage: "plus")
                }
                
                Button(action: { isImporting = true }) {
                    Label("Import", systemImage: "square.and.arrow.down")
                }
                
                Button(action: { isExporting = true }) {
                    Label("Export", systemImage: "square.and.arrow.up")
                }
            }
            .padding(.bottom, 8)
            
            // Search and filter
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                
                TextField("Search presets", text: $searchText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                Picker("Filter", selection: $categoryFilter) {
                    Text("All Presets").tag(nil as String?)
                    
                    Divider()
                    
                    ForEach(Array(presetManager.getPresetsByCategory().keys.sorted()), id: \.self) { category in
                        Text(category).tag(category as String?)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .frame(width: 140)
            }
            
            // Presets list or empty state
            if presetManager.presets.isEmpty {
                EmptyPresetsView(onCreateNew: { isCreatingNewPreset = true })
                    .frame(height: 300)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(8)
            } else {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(filteredPresets) { preset in
                            PresetItemView(
                                preset: preset,
                                isSelected: presetManager.currentPreset?.id == preset.id,
                                onSelect: { presetManager.applyPreset(preset) },
                                onEdit: {
                                    editedPreset = preset
                                    isEditingPreset = true
                                },
                                onDelete: { presetManager.deletePreset(preset) },
                                isExportMode: isExporting,
                                isSelected_export: selectedPresetsForExport.contains(preset.id),
                                onToggleExport: {
                                    if selectedPresetsForExport.contains(preset.id) {
                                        selectedPresetsForExport.remove(preset.id)
                                    } else {
                                        selectedPresetsForExport.insert(preset.id)
                                    }
                                }
                            )
                        }
                    }
                    .padding(.vertical, 4)
                }
                .frame(height: 300)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)
            }
            
            // Current preset parameters section
            if let currentPreset = presetManager.currentPreset {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Adjusting: \(currentPreset.name)")
                        .font(.headline)
                    
                    // Visualization adjustments
                    Group {
                        HStack {
                            Text("Theme:")
                                .frame(width: 100, alignment: .leading)
                            
                            Picker("Theme", selection: $settings.currentTheme) {
                                ForEach(VisualTheme.allCases) { theme in
                                    Text(theme.displayName).tag(theme)
                                }
                            }
                            .pickerStyle(SegmentedPickerStyle())
                            .onChange(of: settings.currentTheme) { _ in
                                settings.save()
                            }
                        }
                        
                        HStack {
                            Text("Sensitivity:")
                

/// View for creating a new preset
@available(macOS 15.0, *)
struct NewPresetView: Sendable: View {
    @Binding var presetName: String
    @Binding var presetDescription: String
    
    let onSave: () -> Void
    let onCancel: () -> Void
    
    @Environment(\.presentationMode) private var presentationMode
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Preset Details")) {
                    TextField("Preset Name", text: $presetName)
                        .disableAutocorrection(true)
                    
                    TextField("Description (optional)", text: $presetDescription)
                        .disableAutocorrection(true)
                }
                
                Section {
                    Text("This preset will save your current visualization settings, audio configuration and neural parameters.")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            }
            .navigationTitle("New Preset")
            .navigationBarItems(
                leading: Button("Cancel") {
                    onCancel()
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Save") {
                    onSave()
                    presentationMode.wrappedValue.dismiss()
                }
                .disabled(presetName.isEmpty)
            )
        }
    }
}

/// View for editing an existing preset
@available(macOS 15.0, *)
struct EditPresetView: Sendable: View {
    let preset: Preset
    let onSave: (Preset) -> Void
    let onCancel: () -> Void
    
    @State private var editedName: String
    @State private var editedDescription: String
    
    @Environment(\.presentationMode) private var presentationMode
    
    init(preset: Preset, onSave: @escaping (Preset) -> Void, onCancel: @escaping () -> Void) {
        self.preset = preset
        self.onSave = onSave
        self.onCancel = onCancel
        
        // Initialize state variables
        _editedName = State(initialValue: preset.name)
        _editedDescription = State(initialValue: preset.description)
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Preset Details")) {
                    TextField("Preset Name", text: $editedName)
                        .disableAutocorrection(true)
                    
                    TextField("Description", text: $editedDescription)
                        .disableAutocorrection(true)
                }
                
                Section(header: Text("Preset Information")) {
                    HStack {
                        Text("Created")
                        Spacer()
                        Text(formattedDate(preset.createdDate))
                            .foregroundColor(.gray)
                    }
                    
                    HStack {
                        Text("Last Modified")
                        Spacer()
                        Text(formattedDate(preset.lastModifiedDate))
                            .foregroundColor(.gray)
                    }
                }
            }
            .navigationTitle("Edit Preset")
            .navigationBarItems(
                leading: Button("Cancel") {
                    onCancel()
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Save") {
                    // Create updated preset maintaining all the original settings
                    var updatedPreset = preset
                    updatedPreset.name = editedName
                    updatedPreset.description = editedDescription
                    
                    onSave(updatedPreset)
                    presentationMode.wrappedValue.dismiss()
                }
                .disabled(editedName.isEmpty)
            )
        }
    }
    
    /// Helper function to format dates
    private func formattedDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

/// View for importing presets
@available(macOS 15.0, *)
struct ImportPresetsView: Sendable: View {
    let presetManager: PresetManager
    
    @State private var selectedURL: URL?
    @State private var isFilePickerPresented = false
    @State private var importMessage: String?
    @State private var showingImportResult = false
    
    @Environment(\.presentationMode) private var presentationMode
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Information at the top
                VStack(alignment: .leading, spacing: 10) {
                    Text("Import Presets")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text("Select a preset file (.abpreset) or a preset package (.zip) to import visualization presets.")
                        .font(.body)
                        .foregroundColor(.secondary)
                    
                    Text("Imported presets will be added to your collection.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)
                
                // Import buttons
                VStack(spacing: 16) {
                    Button(action: {
                        isFilePickerPresented = true
                    }) {
                        HStack {
                            Image(systemName: "doc.badge.plus")
                                .font(.title2)
                            Text("Select Preset File")
                                .font(.headline)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                    
                    Button(action: {
                        // Import sample presets for demo purposes
                        importSamplePresets()
                    }) {
                        HStack {
                            Image(systemName: "wand.and.stars")
                                .font(.title2)
                            Text("Import Sample Presets")
                                .font(.headline)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.purple)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                }
                .padding(.horizontal)
                
                Spacer()
            }
            .padding(.vertical)
            .navigationBarTitle("Import Presets", displayMode: .inline)
            .navigationBarItems(
                trailing: Button("Done") {
                    presentationMode.wrappedValue.dismiss()
                }
            )
            .alert(isPresented: $showingImportResult) {
                Alert(
                    title: Text("Import Result"),
                    message: Text(importMessage ?? ""),
                    dismissButton: .default(Text("OK"))
                )
            }
            // This would be implemented using a document picker in a real app
            // For now, we'll simulate file selection
            .sheet(isPresented: $isFilePickerPresented) {
                // Simulated file picker - in a real app would use UIDocumentPickerViewController on iOS
                // or NSOpenPanel on macOS
                VStack {
                    Text("Simulated File Picker")
                        .font(.headline)
                    
                    Button("Select Sample Preset") {
                        // Simulate selecting a file
                        simulateFileSelection()
                        isFilePickerPresented = false
                    }
                    .padding()
                    
                    Button("Cancel") {
                        isFilePickerPresented = false
                    }
                    .padding()
                }
                .padding()
            }
        }
    }
    
    /// Simulates file selection (in a real app would use actual file picker)
    private func simulateFileSelection() {
        // Create a temporary file URL
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("sample.abpreset")
        
        // Create a sample preset
        let samplePreset = Preset(
            name: "Imported Sample",
            description: "This is a sample imported preset",
            visualSettings: VisualizationSettings(
                theme: .neon,
                sensitivity: 0.8,
                motionIntensity: 0.7,
                showFPS: true,
                showBeatIndicator: true
            ),
            audioSettings: AudioSettings(
                inputDevice: nil,
                outputDevice: nil,
                audioSource: "System Audio",
                micVolume: 0.5,
                systemAudioVolume: 1.0,
                mixInputs: false
            ),
            neuralSettings: NeuralSettings(
                enabled: true,
                beatSensitivity: 0.8,
                patternSensitivity: 0.7,
                emotionalSensitivity: 0.6
            )
        )
        
        // Encode the preset to JSON
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(samplePreset)
            try data.write(to: tempURL)
            
            // Import the preset
            if let importedPreset = presetManager.importPreset(from: tempURL) {
                importMessage = "Successfully imported preset: \(importedPreset.name)"
            } else {
                importMessage = "Failed to import preset."
            }
            showingImportResult = true
            
            // Clean up
            try? FileManager.default.removeItem(at: tempURL)
        } catch {
            importMessage = "Error: \(error.localizedDescription)"
            showingImportResult = true
        }
    }
    
    /// Imports sample presets for demo purposes
    private func importSamplePresets() {
        // Create some sample presets in a temporary directory
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("SamplePresets")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        
        // Create sample presets
        let presets = [
            Preset(
                name: "Dance Music",
                description: "Optimized for electronic dance music with beat emphasis",
                visualSettings: VisualizationSettings(
                    theme: .neon,
                    sensitivity: 0.9,
                    motionIntensity: 0.95,
                    showFPS: false,
                    showBeatIndicator: true
                ),
                audioSettings: AudioSettings(
                    inputDevice: nil,
                    outputDevice: nil,
                    audioSource: "System Audio",
                    micVolume: 0.0,
                    systemAudioVolume: 1.0,
                    mixInputs: false
                ),
                neuralSettings: NeuralSettings(
                    enabled: true,
                    beatSensitivity: 0.9,
                    patternSensitivity: 0.8,
                    emotionalSensitivity: 0.7
                )
            ),
            Preset(
                name: "Classical Ambient",
                description: "Subtle visualization for classical music",
                visualSettings: VisualizationSettings(
                    theme: .monochrome,
                    sensitivity: 0.7,
                    motionIntensity: 0.5,
                    showFPS: false,
                    showBeatIndicator: false
                ),
                audioSettings: AudioSettings(
                    inputDevice: nil,
                    outputDevice: nil,
                    audioSource: "System Audio",
                    micVolume: 0.0,
                    systemAudioVolume: 0.8,
                    mixInputs: false
                ),
                neuralSettings: NeuralSettings(
                    enabled: true,
                    beatSensitivity: 0.4,
                    patternSensitivity: 0.9,
                    emotionalSensitivity: 0.9
                )
            ),
            Preset(
                name: "Party Mode",
                description: "Maximum visual impact for parties",
                visualSettings: VisualizationSettings(
                    theme: .cosmic,
                    sensitivity: 1.0,
                    motionIntensity: 1.0,
                    showFPS: true,
                    showBeatIndicator: true
                ),
                audioSettings: AudioSettings(
                    inputDevice: nil,
                    outputDevice: nil,
                    audioSource: "Mixed (Mic + System)",
                    micVolume: 0.5,
                    systemAudioVolume: 0.9,
                    mixInputs: true
                ),
                neuralSettings: NeuralSettings(
                    enabled: true,
                    beatSensitivity: 1.0,
                    patternSensitivity: 0.8,
                    emotionalSensitivity: 0.8
                )
            )
        ]
        
        // Save each preset to the temp directory
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        for preset in presets {
            let fileURL = tempDir.appendingPathComponent("\(preset.name).abpreset")
            do {
                let data = try encoder.encode(preset)
                try data.write(to: fileURL)
            } catch {
                print("Error saving sample preset: \(error)")
            }
        }
        
        // Import all presets from the directory
        let importCount = presetManager.importPresetsFromDirectory(tempDir)
        
        // Update message
        importMessage = "Successfully imported \(importCount) sample presets."
        showingImportResult = true
        
        // Clean up
        try? FileManager.default.removeItem(at: tempDir)
    }
}
                    .fontWeight(.bold)
                
                Spacer()
                
                Button(action: { isCreatingNewPreset = true }) {
                    Label("New Preset", systemImage: "plus")
                }
                
                Button(action: { isImporting = true }) {
                    Label("Import", systemImage: "square.and.arrow.down")
                }
                
                Button(action: { isExporting = true }) {
                    Label("Export", systemImage: "square.and.arrow.up")
                }
            }
            .padding(.bottom, 8)
            
            // Search and filter
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                
                TextField("Search presets", text: $searchText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                Picker("Filter", selection: $categoryFilter) {
                    Text("All Presets").tag(nil as String?)
                    
                    Divider()
                    
                    ForEach(Array(presetManager.getPresetsByCategory().keys.sorted()), id: \.self) { category in
                        Text(category).tag(category as String?)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .frame(width: 140)
            }
            
            // Presets list
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(filteredPresets) { preset in
                        PresetItemView(
                            preset: preset,
                            isSelected: presetManager.currentPreset?.id == preset.id,
                            onSelect: { presetManager.applyPreset(preset) },
                            onEdit: {
                                editedPreset = preset
                                isEditingPreset = true
                            },
                            onDelete: { presetManager.deletePreset(preset) },
                            isExportMode: isExporting,
                            isSelected_export: selectedPresetsForExport.contains(preset.id),
                            onToggleExport: {
                                if selectedPresetsForExport.contains(preset.id) {
                                    selectedPresetsForExport.remove(preset.id)
                                } else {
                                    selectedPresetsForExport.insert(preset.id)
                                }
                            }
                        )
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(height: 300)
            .background(Color(.secondarySystemBackground))
            .cornerRadius(8)
            
            // Current preset parameters section
            if let currentPreset = presetManager.currentPreset {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Adjusting: \(currentPreset.name)")
                        .font(.headline)
                    
                    // Visualization adjustments
                    Group {
                        HStack {
                            Text("Theme:")
                                .frame(width: 100, alignment: .leading)
                            
                            Picker("Theme", selection: $settings.currentTheme) {
                                ForEach(VisualTheme.allCases) { theme in
                                    Text(theme.displayName).tag(theme)
                                }
                            }
                            .pickerStyle(SegmentedPickerStyle())
                            .onChange(of: settings.currentTheme) { _ in
                                settings.save()
                            }
                        }
                        
                        HStack {
                            Text("Sensitivity:")
                                .frame(width: 100, alignment: .leading)
                            
                            Slider(value: $settings.audioSensitivity, in: 0.1...1.0, step: 0.05)
                                .onChange(of: settings.audioSensitivity) { _ in
                                    settings.save()
                                }
                            
                            Text(String(format: "%.2f", settings.audioSensitivity))
                                .frame(width: 40)
                        }
                        
                        HStack {
                            Text("Motion:")
                                .frame(width: 100, alignment: .leading)
                            
                            Slider(value: $settings.motionIntensity, in: 0.1...1.0, step: 0.05)
                                .onChange(of: settings.motionIntensity) { _ in
                                    settings.save()
                                }
                            
                            Text(String(format: "%.2f", settings.motionIntensity))
                                .frame(width: 40)
                        }
                    }
                    
                    // Neural engine adjustments
                    Group {
                        Toggle("Neural enhancement", isOn: $settings.neuralEngineEnabled)
                            .onChange(of: settings.neuralEngineEnabled) { _ in
                                settings.save()
                            }
                        
                        if settings.neuralEngineEnabled {
                            HStack {
                                Text("Beat detection:")
                                    .frame(width: 100, alignment: .leading)
                                
                                Slider(value: Binding(
                                    get: { Double(settings.beatSensitivity) },
                                    set: { settings.beatSensitivity = Float($0) }
                                ), in: 0.1...1.0, step: 0.05)
                                .onChange(of: settings.beatSensitivity) { _ in
                                    settings.save()
                                }
                                
                                Text(String(format: "%.2f", settings.beatSensitivity))
                                    .frame(width: 40)
                            }
                            
                            HStack {
                                Text("Emotion:")
                                    .frame(width: 100, alignment: .leading)
                                
                                Slider(value: Binding(
                                    get: { Double(settings.emotionalSensitivity) },
                                    set: { settings.emotionalSensitivity = Float($0) }
                                ), in: 0.1...1.0, step: 0.05)
                                .onChange(of: settings.emotionalSensitivity) { _ in
                                    settings.save()
                                }
                                
                                Text(String(format: "%.2f", settings.emotionalSensitivity))
                                    .frame(width: 40)
                            }
                        }
                    }
                    
                    // Save changes button
                    HStack {
                        Spacer()
                        
                        Button("Update Preset") {
                            // Update the current preset with new settings
                            let updatedPreset = Preset.fromCurrentSettings(
                                settings: settings,
                                name: currentPreset.name,
                                description: currentPreset.description
                            )
                            
                            presetManager.updatePreset(presetId: currentPreset.id, updatedPreset: updatedPreset)
                        }
                        
                        Button("Save As New...") {
                            isCreatingNewPreset = true
                            newPresetName = "\(currentPreset.name) Copy"
                        }
                    }
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)
            }
        }
        .padding()
        .sheet(isPresented: $isCreatingNewPreset) {
            NewPresetView(
                presetName: $newPresetName,
                presetDescription: $newPresetDescription,
                onSave: {
                    presetManager.createPresetFromCurrentSettings(
                        name: newPresetName,
                        description: newPresetDescription
                    )
                    newPresetName = ""
                    newPresetDescription = ""
                },
                onCancel: {
                    newPresetName = ""
                    newPresetDescription = ""
                }
            )
        }
        .sheet(isPresented: $isEditingPreset) {
            if let editedPreset = editedPreset {
                EditPresetView(
                    preset: editedPreset,
                    onSave: { updatedPreset in
                        presetManager.updatePreset(presetId: editedPreset.id, updatedPreset: updatedPreset)
                        self.editedPreset = nil
                    },
                    onCancel: {
                        self.editedPreset = nil
                    }
                )
            }
        }
        .sheet(isPresented: $isImporting) {
            ImportPresetsView(presetManager: presetManager)
        }
        .actionSheet(isPresented: $isExporting) {
            ActionSheet(
                title: Text("Export Presets"),
                message: Text("Choose export options"),
                buttons: [
                    .default(Text("Export Selected (\(selectedPresetsForExport.count))")) {
                        exportSelectedPresets()
                    },
                    .default(Text("Export All")) {
                        exportAllPresets()
                    },
                    .cancel {
                        selectedPresetsForExport.removeAll()
                        isExporting = false
                    }
                ]
            )
        }
    }
    
    /// Returns filtered presets based on search and category
    private var filteredPresets: [Preset] {
        var result = presetManager.presets
        
        // Apply category filter if selected
        if let categoryFilter = categoryFilter {
            result = presetManager.getPresetsByCategory()[categoryFilter] ?? []
        }
        
        // Apply search filter if text entered
        if !searchText.isEmpty {
            result = result.filter { preset in
                preset.name.localizedCaseInsensitiveContains(searchText) ||
                preset.description.localizedCaseInsensitiveContains(searchText)
            }
        }
        
        return result
    }
    
    /// Exports selected presets
    private func exportSelectedPresets() {
        let selectedPresets = presetManager.presets.filter { selectedPresetsForExport.contains($0.id) }
        if !selectedPresets.isEmpty {
            _ = presetManager.createPresetPackage(selectedPresets)
        }
        
        selectedPresetsForExport.removeAll()
        isExporting = false
    }
    
    /// Exports all presets
    private func exportAllPresets() {
        _ = presetManager.createPresetPackage(presetManager.presets)
        isExporting = false
    }
}

/// Individual preset item in the list
@available(macOS 15.0, *)
private struct PresetItemView: Sendable: View {
    let preset: Preset
    let isSelected: Bool
    let onSelect: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void
    
    let isExportMode: Bool
    let isSelected_export: Bool
    let onToggleExport: () -> Void
    
    @State private var showDeleteConfirmation = false
    
    var body: some View {
        HStack {
            // Selection checkbox for export mode
            if isExportMode {
                Button(action: onToggleExport) {
                    Image(systemName: isSelected_export ? "checkmark.square.fill" : "square")
                        .foregroundColor(isSelected_export ? .blue : .gray)
                }
                .buttonStyle(PlainButtonStyle())
                .padding(.trailing, 8)
            }
            
            // Preset content
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(preset.name)
                        .font(.headline)
                    
                    if preset.neuralSettings.enabled {
                        Image(systemName: "brain")
                            .foregroundColor(.purple)
                            .font(.caption)
                    }
                    
                    Spacer()
                    
                    // Audio source icon
                    switch preset.audioSettings.audioSource {
                    case "Microphone":
                        Image(systemName: "mic.fill")
                            .foregroundColor(.blue)
                    case "System Audio":
                        Image(systemName: "speaker.wave.2.fill")
                            .foregroundColor(.green)
                    case "Mixed (Mic + System)":
                        Image(systemName: "dial.max.fill")
                            .foregroundColor(.orange)
                    default:
                        EmptyView()
                    }
                    
                    // Theme indicator
                    switch preset.visualSettings.theme {
                    case .classic:
                        Circle().fill(Color.blue).frame(width: 10, height: 10)
                    case .neon:
                        Circle().fill(Color.green).frame(width: 10, height: 10)
                    case .monochrome:
                        Circle().fill(Color.gray).frame(width: 10, height: 10)
                    case .cosmic:
                        Circle().fill(Color.purple).frame(width: 10, height: 10)
                    }
                }
                
                if !preset.description.isEmpty {
                    Text(preset.description)
                        .font(.caption)
                        .foregroundColor(.gray)
                        .lineLimit(1)
                }
                
                // Date information
                Text("Created: \(formattedDate(preset.createdDate))")
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(isSelected ? Color.blue.opacity(0.2) : Color.clear)
            .cornerRadius(8)
            .contentShape(Rectangle())
            .onTapGesture {
                onSelect()
            }
            
            // Action buttons (only if not in export mode)
            if !isExportMode {
                VStack(spacing: 8) {
                    Button(action: onEdit) {
                        Image(systemName: "pencil")
                            .foregroundColor(.blue)
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                    Button(action: { showDeleteConfirmation = true }) {
                        Image(systemName: "trash")
                            .foregroundColor(.red)
                    }
                    .buttonStyle(PlainButtonStyle())
                    .alert(isPresented: $showDeleteConfirmation) {
                        Alert(
                            title: Text("Delete Preset"),
                            message: Text("Are you sure you want to delete '\(preset.name)'? This cannot be undone."),
                            primaryButton: .destructive(Text("Delete")) {
                                onDelete()
                            },
                            secondaryButton: .cancel()
                        )
                    }
                }
                .padding(.leading, 8)
            }
        }
        .padding(.horizontal, 2)
        .background(Color(.systemBackground))
        .cornerRadius(8)
        .shadow(color: Color.black.opacity(0.1), radius: 1, x: 0, y: 1)
    }
    
    /// Helper function to format dates
    private func formattedDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .none
        return formatter.string(from: date)
    }
}

/// Preset tag view for displaying categories and attributes
@available(macOS 15.0, *)
private struct PresetTagView: Sendable: View {
    let text: String
    let color: Color
    
    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundColor(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color)
            .cornerRadius(12)
    }
}

/// Empty state view for when no presets are available
@available(macOS 15.0, *)
private struct EmptyPresetsView: Sendable: View {
    let onCreateNew: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "waveform")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("No Presets Available")
                .font(.headline)
            
            Text("Create your first preset to save your visualization settings")
                .font(.subheadline)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            Button(action: onCreateNew) {
                HStack {
                    Image(systemName: "plus.circle.fill")
                    Text("Create First Preset")
                }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
            }
        }
        .padding()
    }
}

