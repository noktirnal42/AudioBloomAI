#!/usr/bin/env python3
"""
AI Collaborator - A local development tool that combines Ollama and Gemini
for GitHub issue and PR management in AudioBloomAI project.

This tool is not part of the AudioBloomAI codebase, but a personal developer utility.
"""

import os
import sys
import json
import subprocess
import argparse
import requests
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_collaborator")

class AICollaborator:
    def __init__(self):
        # Check for required environment variables
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Gemini features disabled.")
        
        self.ollama_model = "llama3.2:latest"
        self.gemini_model = "gemini-1.5-pro"
        
        # Context for Swift and macOS requirements
        self.swift_version = "6.0"
        self.macos_version = "15.0"
        
    def run_command(self, cmd: List[str]) -> Tuple[int, str]:
        """Run a shell command and return exit code and output."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return result.returncode, result.stdout
        except Exception as e:
            logger.error(f"Error running command {' '.join(cmd)}: {e}")
            return 1, str(e)
    
    def ollama_query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Query the local Ollama service with llama3.2:latest model."""
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return f"Error: {str(e)}"
    
    def gemini_query(self, prompt: str) -> str:
        """Query the Google Gemini API."""
        if not self.gemini_api_key:
            return "Gemini API not configured. Set GEMINI_API_KEY environment variable."
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 4096,
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Error querying Gemini API: {e}")
            return f"Error: {str(e)}"
    
    def analyze_pr(self, pr_number: int) -> str:
        """Analyze a PR using both Ollama and Gemini."""
        logger.info(f"Analyzing PR #{pr_number}")
        
        # Get PR details using GitHub CLI
        exit_code, pr_json = self.run_command([
            "gh", "pr", "view", str(pr_number), 
            "--json", "title,body,files,commits,reviewDecision"
        ])
        
        if exit_code != 0:
            return f"Failed to get PR data: {pr_json}"
        
        try:
            pr_data = json.loads(pr_json)
            
            # Get the diff for more detailed analysis
            exit_code, pr_diff = self.run_command(["gh", "pr", "diff", str(pr_number)])
            
            # Analyze with Ollama first
            ollama_prompt = f"""
            Review this Pull Request for AudioBloomAI (Swift 6, macOS 15+):
            
            Title: {pr_data.get('title')}
            Description: {pr_data.get('body')}
            
            Files changed: {', '.join(f.get('path', '') for f in pr_data.get('files', []))}
            
            Diff:
            {pr_diff[:4000]}  # Truncate if too large
            
            Analyze for:
            1. Swift 6 compatibility
            2. macOS 15+ compatibility
            3. Code quality and best practices
            4. Potential bugs or issues
            
            Provide specific, actionable feedback.
            """
            
            ollama_review = self.ollama_query(ollama_prompt)
            
            # Only use Gemini if configured
            if self.gemini_api_key:
                gemini_prompt = f"""
                You're reviewing a Pull Request for an advanced audio processing app built with Swift.
                
                PR Details:
                Title: {pr_data.get('title')}
                
                A preliminary review found these issues:
                
                {ollama_review[:3000]}
                
                Please evaluate this review and add any additional insights focusing on:
                - Swift 6 and macOS 15+ specific concerns
                - Audio processing best practices
                - Performance implications
                - Architectural considerations
                
                Be specific and actionable in your recommendations.
                """
                
                gemini_review = self.gemini_query(gemini_prompt)
                
                combined = f"""
                # Collaborative PR Review: #{pr_number} - {pr_data.get('title')}
                
                ## Technical Review (Ollama)
                
                {ollama_review}
                
                ## Advanced Analysis (Gemini)
                
                {gemini_review}
                
                ## Next Steps
                
                1. Review the feedback above
                2. Address critical issues first
                3. Consider performance implications
                4. Ensure Swift 6 and macOS 15+ compatibility
                """
                
                return combined
            else:
                # Return just the Ollama review if Gemini isn't available
                return f"# PR Review: #{pr_number} - {pr_data.get('title')}\n\n{ollama_review}"
                
        except Exception as e:
            logger.error(f"Error analyzing PR: {e}")
            return f"Error analyzing PR: {str(e)}"
    
    def fix_issue(self, issue_number: int) -> str:
        """Generate a solution for an issue using collaborative AI."""
        logger.info(f"Generating solution for issue #{issue_number}")
        
        # Get issue details using GitHub CLI
        exit_code, issue_json = self.run_command([
            "gh", "issue", "view", str(issue_number), 
            "--json", "title,body,labels"
        ])
        
        if exit_code != 0:
            return f"Failed to get issue data: {issue_json}"
        
        try:
            issue_data = json.loads(issue_json)
            
            # Analyze with Ollama first
            ollama_prompt = f"""
            Analyze this issue for AudioBloomAI (Swift 6, macOS 15+):
            
            Title: {issue_data.get('title')}
            Description: {issue_data.get('body')}
            Labels: {', '.join(l.get('name', '') for l in issue_data.get('labels', []))}
            
            Provide a solution that:
            1. Addresses the core problem
            2. Is compatible with Swift 6 and macOS 15+
            3. Follows best practices
            4. Includes code examples where appropriate
            
            Be specific and actionable.
            """
            
            ollama_solution = self.ollama_query(ollama_prompt)
            
            # Only use Gemini if configured
            if self.gemini_api_key:
                gemini_prompt = f"""
                You're helping solve an issue for an advanced audio processing app built with Swift.
                
                Issue Details:
                Title: {issue_data.get('title')}
                
                A proposed solution:
                
                {ollama_solution[:3000]}
                
                Please refine this solution focusing on:
                - Swift 6 and macOS 15+ specific implementations
                - Audio processing best practices
                - Performance optimization
                - Code clarity and maintainability
                
                Include specific code examples if possible.
                """
                
                gemini_solution = self.gemini_query(gemini_prompt)
                
                combined = f"""
                # Collaborative Solution: Issue #{issue_number} - {issue_data.get('title')}
                
                ## Initial Solution (Ollama)
                
                {ollama_solution}
                
                ## Refined Approach (Gemini)
                
                {gemini_solution}
                
                ## Implementation Plan
                
                1. Review both solutions above
                2. Create a new branch: `git checkout -b fix/issue-{issue_number}`
                3. Implement the solution, prioritizing critical components
                4. Add tests to verify the fix
                5. Create PR with: `gh pr create --title "Fix #{issue_number}: {issue_data.get('title')}" --body "Fixes #{issue_number}"`
                """
                
                return combined
            else:
                # Return just the Ollama solution if Gemini isn't available
                return f"# Issue Solution: #{issue_number} - {issue_data.get('title')}\n\n{ollama_solution}"
                
        except Exception as e:
            logger.error(f"Error fixing issue: {e}")
            return f"Error fixing issue: {str(e)}"

    def create_pr(self, title: str, body: str, base_branch: str = "main") -> str:
        """Create a PR with AI-enhanced description."""
        logger.info(f"Creating PR: {title}")
        
        # Get current branch
        exit_code, current_branch = self.run_command(["git", "branch", "--show-current"])
        if exit_code != 0:
            return f"Failed to get current branch: {current_branch}"
        
        current_branch = current_branch.strip()
        
        # Enhance PR description with Ollama
        ollama_prompt = f"""
        Enhance this Pull Request description for AudioBloomAI (Swift 6, macOS 15+):
        
        Title: {title}
        Description: {body}
        
        Create a structured PR description that includes:
        1. Summary of changes
        2. Technical implementation details
        3. Testing performed
        4. Swift 6 and macOS 15+ compatibility notes
        
        Format with markdown and keep it concise but informative.
        """
        
        enhanced_body = self.ollama_query(ollama_prompt)
        
        # Create the PR using GitHub CLI
        exit_code, result = self.run_command([
            "gh", "pr", "create",
            "--title", title,
            "--body", enhanced_body,
            "--base", base_branch
        ])
        
        if exit_code != 0:
            return f"Failed to create PR: {result}"
        
        return f"PR created successfully:\n{result}"

def main():
    parser = argparse.ArgumentParser(description="AI Collaborator for GitHub issues and PRs")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # PR analysis command
    pr_parser = subparsers.add_parser("pr", help="Analyze a PR")
    pr_parser.add_argument("number", type=int, help="PR number to analyze")
    
    # Issue solution command
    issue_parser = subparsers.add_parser("issue", help="Generate a solution for an issue")
    issue_parser.add_argument("number", type=int, help="Issue number to fix")
    
    # Create PR command
    create_pr_parser = subparsers.add_parser("create-pr", help="Create a PR with AI-enhanced description")
    create_pr_parser.add_argument("title", help="PR title")
    create_pr_parser.add_argument("--body", default="", help="Initial PR description")
    create_pr_parser.add_argument("--base", default="main", help="Base branch for the PR")
    
    args = parser.parse_args()
    
    # Initialize the collaborator
    collaborator = AICollaborator()
    
    if args.command == "pr":
        result = collaborator.analyze_pr(args.number)
        print(result)
    elif args.command == "issue":
        result = collaborator.fix_issue(args.number)
        print(result)
    elif args.command == "create-pr":
        result = collaborator.create_pr(args.title, args.body, args.base)
        print(result)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

