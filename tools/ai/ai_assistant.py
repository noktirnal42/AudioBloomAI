#!/usr/bin/env python3
# AudioBloomAI - Simple AI Assistant Script
# This script provides an interface to Ollama for AI-assisted development

import os
import sys
import json
import logging
import argparse
import requests
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser("~/Developer/AudioBloomAI/tools/ai/ai_assistant.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AudioBloomAI-AI-Assistant")

class OllamaAssistant:
    """Simple AI Assistant for AudioBloomAI development using Ollama."""
    
    def __init__(self, model: str = "deepseek-r1:8b"):
        """
        Initialize the AI Assistant with Ollama configuration.
        
        Args:
            model: Name of the Ollama model to use
        """
        self.repo_path = os.path.expanduser("~/Developer/AudioBloomAI")
        self.api_base = "http://localhost:11434"
        self.model = model
        
        # Test Ollama connection
        try:
            response = requests.get(f"{self.api_base}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                if self.model in model_names:
                    logger.info(f"Ollama model {self.model} is available")
                else:
                    logger.warning(f"Ollama model {self.model} not found in available models: {model_names}")
            else:
                logger.warning(f"Failed to get Ollama models: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama API: {str(e)}")
    
    def query_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Send a query to Ollama and return the response.
        
        Args:
            prompt: The prompt to send to Ollama
            temperature: Temperature setting for generation
            
        Returns:
            The response text from Ollama
        """
        try:
            api_url = f"{self.api_base}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "max_tokens": 2048
            }
            
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            return f"Error: Failed to query Ollama - {str(e)}"
    
    def analyze_code(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Swift code file and provide feedback.
        
        Args:
            file_path: Path to the Swift file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Check if file exists
            full_path = file_path
            if not os.path.isabs(file_path):
                full_path = os.path.join(self.repo_path, file_path)
                
            if not os.path.exists(full_path):
                logger.error(f"File not found: {full_path}")
                return {
                    "success": False,
                    "error": f"File not found: {full_path}"
                }
                
            # Check if it's a Swift file
            if not full_path.endswith(".swift"):
                logger.warning(f"Not a Swift file: {full_path}")
                return {
                    "success": False,
                    "error": f"Not a Swift file: {full_path}"
                }
                
            # Read the file content
            with open(full_path, 'r') as f:
                code_content = f.read()
            
            # Build the prompt for code analysis
            prompt = f"""
            Please analyze this Swift code and provide feedback on:
            1. Code quality and adherence to Swift 6.0 guidelines
            2. Potential performance issues
            3. Error handling completeness
            4. Thread safety concerns
            5. Documentation quality
            6. Suggestions for improvements

            Important context:
            - This is part of the AudioBloomAI project
            - The project targets macOS 15+ and uses Swift 6.0
            - Performance is critical, with real-time audio processing requirements
            
            Here is the code from {os.path.basename(full_path)}:
            
            ```swift
            {code_content}
            ```
            """
            
            # Get response from Ollama
            response = self.query_ollama(prompt)
            
            return {
                "success": True,
                "file": full_path,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to analyze code: {str(e)}"
            }
    
    def design_api(self, requirements: str, module: str) -> Dict[str, Any]:
        """
        Design an API based on given requirements.
        
        Args:
            requirements: Requirements for the API
            module: The module where the API will be implemented
            
        Returns:
            Dictionary containing API design
        """
        try:
            # Validate module
            valid_modules = [
                "AudioProcessor", "Visualizer", "MLEngine", 
                "AudioBloomUI", "AudioBloomCore", "AudioBloomApp"
            ]
            
            if module not in valid_modules:
                logger.warning(f"Invalid module: {module}")
                return {
                    "success": False,
                    "error": f"Invalid module: {module}. Valid modules are: {', '.join(valid_modules)}"
                }
                
            # Module-specific guidelines
            module_guidelines = {
                "AudioProcessor": [
                    "Real-time processing requirements with max latency of 10ms", 
                    "Buffer management considerations", 
                    "Thread safety requirements", 
                    "Error recovery procedures"
                ],
                "Visualizer": [
                    "Metal shader optimization", 
                    "Frame rate requirement: 60 FPS", 
                    "Memory management", 
                    "State handling"
                ],
                "MLEngine": [
                    "Model integration guidelines", 
                    "Feature extraction requirements", 
                    "Performance optimization", 
                    "Error handling"
                ],
                "AudioBloomUI": [
                    "SwiftUI best practices", 
                    "State management with Composable Architecture", 
                    "Performance considerations", 
                    "Accessibility requirements"
                ],
                "AudioBloomCore": [
                    "Shared functionality", 
                    "Core data structures", 
                    "Common utilities",
                    "Thread safety"
                ],
                "AudioBloomApp": [
                    "Main application logic", 
                    "Module integration", 
                    "Resources management",
                    "Configuration handling"
                ]
            }
            
            guidelines = module_guidelines.get(module, [])
            guidelines_text = "\n".join([f"- {g}" for g in guidelines])
                
            prompt = f"""
            Please design an API based on these requirements:
            
            {requirements}
            
            Module: {module}
            
            Module-specific guidelines:
            {guidelines_text}
            
            Project context:
            - This is part of the AudioBloomAI project
            - The project targets macOS 15+ and uses Swift 6.0
            - Performance is critical, with real-time constraints
            
            Provide:
            1. Interface/protocol definitions
            2. Function signatures with parameter and return types
            3. Public vs. private considerations
            4. Error handling strategy
            5. Thread safety recommendations
            6. Documentation in DocC format
            """
            
            # Get response from Ollama
            response = self.query_ollama(prompt)
            
            return {
                "success": True,
                "module": module,
                "requirements": requirements,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error designing API: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to design API: {str(e)}"
            }
    
    def suggest_implementation(self, description: str, module: str) -> Dict[str, Any]:
        """
        Suggest an implementation for a new feature.
        
        Args:
            description: Description of the feature to implement
            module: The module where the feature should be implemented
            
        Returns:
            Dictionary containing implementation suggestion
        """
        try:
            # Validate module
            valid_modules = [
                "AudioProcessor", "Visualizer", "MLEngine", 
                "AudioBloomUI", "AudioBloomCore", "AudioBloomApp"
            ]
            
            if module not in valid_modules:
                logger.warning(f"Invalid module: {module}")
                return {
                    "success": False,
                    "error": f"Invalid module: {module}. Valid modules are: {', '.join(valid_modules)}"
                }
                
            # Module-specific guidelines (same as in design_api)
            module_guidelines = {
                "AudioProcessor": [
                    "Real-time processing requirements with max latency of 10ms", 
                    "Buffer management considerations", 
                    "Thread safety requirements", 
                    "Error recovery procedures"
                ],
                "Visualizer": [
                    "Metal shader optimization", 
                    "Frame rate requirement: 60 FPS", 
                    "Memory management", 
                    "State handling"
                ],
                "MLEngine": [
                    "Model integration guidelines", 
                    "Feature extraction requirements", 
                    "Performance optimization", 
                    "Error handling"
                ],
                "AudioBloomUI": [
                    "SwiftUI best practices", 
                    "State management with Composable Architecture", 
                    "Performance considerations", 
                    "Accessibility requirements"
                ],
                "AudioBloomCore": [
                    "Shared functionality", 
                    "Core data structures", 
                    "Common utilities",
                    "Thread safety"
                ],
                "AudioBloomApp": [
                    "Main application logic", 
                    "Module integration", 
                    "Resources management",
                    "Configuration handling"
                ]
            }
            
            guidelines = module_guidelines.get(module, [])
            guidelines_text = "\n".join([f"- {g}" for g in guidelines])
                
            prompt = f"""
            Please suggest an implementation for the following feature:
            
            {description}
            
            Module: {module}
            
            Module-specific guidelines:
            {guidelines_text}
            
            Project context:
            - This is part of the AudioBloomAI project
            - The project targets macOS 15+ and uses Swift 6.0
            - Performance is critical, with real-time constraints
            
            Provide:
            1. A high-level approach
            2. Key data structures and functions
            3. Swift 6.0 code implementation
            4. Considerations for performance, thread safety, and error handling
            5. Integration points with existing code
            """
            
            # Get response from Ollama
            response = self.query_ollama(prompt)
            
            return {
                "success": True,
                "module": module,
                "description": description,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error suggesting implementation: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to suggest implementation: {str(e)}"
            }


def main():
    """Main entry point for the AI Assistant CLI"""
    parser = argparse.ArgumentParser(description='AudioBloomAI AI Assistant - Interfaces with Ollama')
    
    # Global arguments
    parser.add_argument('--model', type=str, default="deepseek-r1:8b", help='Ollama model to use')
    
    # Create subparsers for different command types
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Analyze code command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze Swift code')
    analyze_parser.add_argument('file_path', type=str, help='Path to the Swift file to analyze')
    
    # Implement feature command
    implement_parser = subparsers.add_parser('implement', help='Get implementation suggestions for a feature')
    implement_parser.add_argument('--module', type=str, required=True, 
                                  choices=["AudioProcessor", "Visualizer", "MLEngine", 
                                           "AudioBloomUI", "AudioBloomCore", "AudioBloomApp"], 
                                  help='Module where the feature will be implemented')
    implement_parser.add_argument('--description', type=str, required=True, help='Feature description')
    
    # Design API command
    api_parser = subparsers.add_parser('api', help='Design an API based on requirements')
    api_parser.add_argument('--module', type=str, required=True, 
                            choices=["AudioProcessor", "Visualizer", "MLEngine", 
                                     "AudioBloomUI", "AudioBloomCore", "AudioBloomApp"], 
                            help='Module where the API will be implemented')
    api_parser.add_argument('--requirements', type=str, required=True, help='API requirements')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize AI Assistant
    assistant = OllamaAssistant(model=args.model)
    
    # Handle commands
    if args.command == 'analyze':
        result = assistant.analyze_code(args.file_path)
        if not result.get("success", False):
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        print(f"\nAnalysis Results for {os.path.basename(result['file'])}:\n")
        print(result['response'])
        
    elif args.command == 'implement':
        result = assistant.suggest_implementation(args.description, args.module)
        if not result.get("success", False):
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        print(f"\nImplementation Suggestions for {args.module}:\n")
        print(result['response'])
        
    elif args.command == 'api':
        result = assistant.design_api(args.requirements, args.module)
        if not result.get("success", False):
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        print(f"\nAPI Design for {args.module}:\n")
        print(result['response'])
        
    else:
        parser.print_help()
        print("\nError: No command specified. Please use one of the available commands.")
        sys.exit(1)


if __name__ == "__main__":
    main()

