#!/usr/bin/env python3
# AudioBloomAI - AI Assistant Script
# This script provides interfaces for both Ollama and Google Gemini
# and coordinates their usage for various development tasks.

import os
import sys
import json
import logging
import argparse
import requests
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Try importing Google Generative AI
try:
    import google.generativeai as genai
except ImportError:
    print("Google Generative AI package not found. Please install with:")
    print("pip install google-generativeai")
    sys.exit(1)

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

# Define task complexity levels (as per AI_WORKFLOW.md)
class ComplexityLevel(Enum):
    SIMPLE = 1
    STRAIGHTFORWARD = 2
    MODERATE = 3
    COMPLEX = 4
    HIGHLY_COMPLEX = 5

# Define task types for routing
class TaskType(Enum):
    CODE_REVIEW = "code_review"
    CODE_IMPLEMENTATION = "code_implementation"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    ML_INTEGRATION = "ml_integration"
    API_DESIGN = "api_design"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BUG_FIX = "bug_fix"

# Define project modules
class ProjectModule(Enum):
    AUDIO_PROCESSOR = "AudioProcessor"
    VISUALIZER = "Visualizer"
    ML_ENGINE = "MLEngine"
    AUDIO_BLOOM_UI = "AudioBloomUI"
    AUDIO_BLOOM_CORE = "AudioBloomCore"
    AUDIO_BLOOM_APP = "AudioBloomApp"

class AIAssistant:
    """
    AI Assistant for AudioBloomAI development that coordinates between
    Ollama (deepseek-r1:8b) and Google Gemini.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI Assistant with configurations for both Ollama and Gemini.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.repo_path = os.path.expanduser("~/Developer/AudioBloomAI")
        
        # Default configuration
        self.config = {
            "ollama": {
                "api_base": "http://localhost:11434",
                "model": "deepseek-r1:8b",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            },
            "gemini": {
                "model": "gemini-pro",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_output_tokens": 2048
                }
            },
            "task_routing": {
                "code_review": "ollama",
                "code_implementation": "ollama",
                "documentation": "ollama", 
                "architecture": "gemini",
                "ml_integration": "gemini",
                "api_design": "gemini",
                "performance_optimization": "ollama",
                "bug_fix": "ollama"
            }
        }
        
        # Load custom configuration if provided
        if config_path:
            self._load_config(config_path)
            
        # Initialize Gemini API if API key is available
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel(
                    self.config["gemini"]["model"],
                    generation_config=self.config["gemini"]["parameters"]
                )
                logger.info(f"Initialized Gemini with model {self.config['gemini']['model']}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {str(e)}")
                self.gemini_model = None
        else:
            logger.warning("No Gemini API key found in environment variables (GEMINI_API_KEY)")
            self.gemini_model = None
        
        # Test Ollama connection
        try:
            response = requests.get(f"{self.config['ollama']['api_base']}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                if self.config['ollama']['model'] in model_names:
                    logger.info(f"Ollama model {self.config['ollama']['model']} is available")
                else:
                    logger.warning(f"Ollama model {self.config['ollama']['model']} not found in available models: {model_names}")
            else:
                logger.warning(f"Failed to get Ollama models: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama API: {str(e)}")
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from a JSON file"""
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                
            # Update config with custom values
            for key, value in custom_config.items():
                if key in self.config:
                    if isinstance(self.config[key], dict) and isinstance(value, dict):
                        self.config[key].update(value)
                    else:
                        self.config[key] = value
                        
            logger.info(f"Loaded custom configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
    
    def _route_task(self, task_type: TaskType, complexity: ComplexityLevel) -> str:
        """
        Determine which AI service should handle a given task.
        
        Args:
            task_type: Type of task
            complexity: Complexity level of task
            
        Returns:
            "ollama" or "gemini" based on routing rules
        """
        # Default routing based on task type
        service = self.config["task_routing"].get(task_type.value, "ollama")
        
        # Override based on complexity
        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.HIGHLY_COMPLEX]:
            # Use Gemini for more complex tasks
            if task_type not in [TaskType.CODE_IMPLEMENTATION, TaskType.BUG_FIX]:
                service = "gemini"
                
        return service
    
    def query_ollama(self, prompt: str) -> str:
        """
        Send a query to Ollama and return the response.
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            The response text from Ollama
        """
        try:
            api_url = f"{self.config['ollama']['api_base']}/api/generate"
            payload = {
                "model": self.config['ollama']['model'],
                "prompt": prompt,
                "stream": False,
                **self.config['ollama']['parameters']
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
    
    def query_gemini(self, prompt: str) -> str:
        """
        Send a query to Google Gemini and return the response.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            The response text from Gemini
        """
        if not self.gemini_model:
            return "Error: Gemini is not configured properly. Please set GEMINI_API_KEY environment variable."
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error querying Gemini: {str(e)}")
            return f"Error: Failed to query Gemini - {str(e)}"
    
    def submit_task(self, 
                   task_type: Union[TaskType, str], 
                   prompt: str, 
                   complexity: Union[ComplexityLevel, int] = ComplexityLevel.MODERATE,
                   module: Optional[Union[ProjectModule, str]] = None,
                   force_service: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit a task to the appropriate AI service based on task type and complexity.
        
        Args:
            task_type: Type of task (use TaskType enum or string name)
            prompt: The prompt describing the task
            complexity: Complexity level (use ComplexityLevel enum or int 1-5)
            module: Project module this task relates to (optional)
            force_service: Force using a specific service ("ollama" or "gemini")
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Convert string task_type to enum if needed
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                logger.warning(f"Invalid task type: {task_type}. Using CODE_IMPLEMENTATION as default.")
                task_type = TaskType.CODE_IMPLEMENTATION
        
        # Convert integer complexity to enum if needed
        if isinstance(complexity, int):
            try:
                complexity = ComplexityLevel(complexity)
            except ValueError:
                logger.warning(f"Invalid complexity level: {complexity}. Using MODERATE as default.")
                complexity = ComplexityLevel.MODERATE
        
        # Convert string module to enum if needed
        if isinstance(module, str) and module:
            try:
                module = ProjectModule(module)
            except ValueError:
                logger.warning(f"Invalid module: {module}")
                module = None
        
        # Determine which service to use
        service = force_service if force_service else self._route_task(task_type, complexity)
        
        # Build the full prompt with context
        full_prompt = self._build_prompt(prompt, task_type, complexity, module)
        
        # Log the task submission
        logger.info(f"Submitting {task_type.value} task (complexity: {complexity.value}) to {service}")
        
        # Query the appropriate service
        if service == "gemini":
            response_text = self.query_gemini(full_prompt)
        else:  # default to ollama
            response_text = self.query_ollama(full_prompt)
        
        # Return results with metadata
        return {
            "service": service,
            "task_type": task_type.value,
            "complexity": complexity.value,
            "module": module.value if module else None,
            "prompt": prompt,
            "full_prompt": full_prompt,
            "response": response_text
        }
    
    def _build_prompt(self, prompt: str, task_type: TaskType, 
                     complexity: ComplexityLevel, 
                     module: Optional[ProjectModule]) -> str:
        """
        Build a full prompt with context based on the task parameters.
        
        Args:
            prompt: The base prompt
            task_type: Type of task
            complexity: Complexity level
            module: Project module this task relates to (optional)
            
        Returns:
            A full prompt with appropriate context
        """
        # Add AudioBloomAI project context
        context_parts = [
            "You are assisting with the AudioBloomAI project, a next-generation audio visualizer",
            "optimized for Apple Silicon M3 Pro, featuring Neural Engine integration and Metal-based graphics.",
            f"Project requirements: macOS 15+, Swift 6.0, Xcode 16+",
            f"Task type: {task_type.value}",
            f"Complexity level: {complexity.value}/5"
        ]
        
        # Add module-specific guidelines if applicable
        if module:
            module_guidelines = self._get_module_guidelines(module)
            if module_guidelines:
                context_parts.append(f"Module guidelines for {module.value}:")
                context_parts.extend([f"- {guideline}" for guideline in module_guidelines])
        
        # Add general code standards
        if task_type in [TaskType.CODE_IMPLEMENTATION, TaskType.CODE_REVIEW, TaskType.BUG_FIX]:
            context_parts.extend([
                "Code standards:",
                "- Follow Swift 6.0 guidelines",
                "- Implement comprehensive error handling",
                "- Ensure thread safety where appropriate",
                "- Use DocC comment format for documentation",
                "- Include performance considerations"
            ])
        
        # Combine context with the original prompt
        full_context = "\n".join(context_parts)
        full_prompt = f"{full_context}\n\n{prompt}"
        
        return full_prompt
    
    def _get_module_guidelines(self, module: ProjectModule) -> List[str]:
        """Get module-specific guidelines based on AI_WORKFLOW.md"""
        guidelines = {
            ProjectModule.AUDIO_PROCESSOR: [
                "Real-time processing requirements with max latency of 10ms", 
                "Buffer management considerations", 
                "Thread safety requirements", 
                "Error recovery procedures"
            ],
            ProjectModule.VISUALIZER: [
                "Metal shader optimization", 
                "Frame rate requirement: 60 FPS", 
                "Memory management", 
                "State handling"
            ],
            ProjectModule.ML_ENGINE: [
                "Model integration guidelines", 
                "Feature extraction requirements", 
                "Performance optimization", 
                "Error handling"
            ],
            ProjectModule.AUDIO_BLOOM_UI: [
                "SwiftUI best practices", 
                "State management with Composable Architecture", 
                "Performance considerations", 
                "Accessibility requirements"
            ],
            ProjectModule.AUDIO_BLOOM_CORE: [
                "Shared functionality", 
                "Core data structures", 
                "Common utilities",
                "Thread safety"
            ],
            ProjectModule.AUDIO_BLOOM_APP: [
                "Main application logic", 
                "Module integration", 
                "Resources management",
                "Configuration handling"
            ]
        }
        
        return guidelines.get(module, [])
    
    def analyze_code(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Swift code file and provide feedback.
        
        Args:
            file_path: Path to the Swift file to analyze
            
        Returns:
            Dictionary containing analysis results and metadata
        """

