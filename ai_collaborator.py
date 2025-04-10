#!/usr/bin/env python3
"""
AI Collaborator - A local development tool that uses multiple Ollama models 
for GitHub issue and PR management in AudioBloomAI project.

This tool is not part of the AudioBloomAI codebase, but a personal developer utility.
It uses llama3.2:latest and deepseek-r1:8b models for collaborative AI analysis.
"""

import os
import sys
import json
import subprocess
import argparse
import requests
import time
from typing import List, Dict, Optional, Tuple, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_collaborator")

class AICollaborator:
    def __init__(self):
        # Configure Ollama models
        self.primary_model = "llama3.2:latest"
        self.secondary_model = "deepseek-r1:8b"
        
        # Repository configuration
        self.repo = "noktirnal42/AudioBloomAI"
        
        # Context for Swift and macOS requirements
        self.swift_version = "6.0"
        self.macos_version = "15.0"
        
        # Tracking for repository monitoring
        self.processed_prs = set()
        self.processed_issues = set()
        self.check_interval = 300  # 5 minutes by default
        
        logger.info(f"AI Collaborator initialized with models: {self.primary_model} and {self.secondary_model}")
        logger.info(f"Monitoring repository: {self.repo}")
        
    def run_command(self, cmd: List[str]) -> Tuple[int, str]:
        """Run a shell command and return exit code and output."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return result.returncode, result.stdout
        except Exception as e:
            logger.error(f"Error running command {' '.join(cmd)}: {e}")
            return 1, str(e)
    
    def ollama_query(self, prompt: str, model: Optional[str] = None, system_prompt: Optional[str] = None, 
                temperature: float = 0.2) -> str:
        """Query the local Ollama service with specified model."""
        url = "http://localhost:11434/api/generate"
        
        # Use specified model or default to primary model
        model_to_use = model if model else self.primary_model
        
        logger.debug(f"Querying {model_to_use} model")
        
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Error querying Ollama with {model_to_use}: {e}")
            return f"Error: {str(e)}"
            
    def collaborative_query(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """Query both models sequentially and return both responses."""
        # Primary model analysis
        primary_response = self.ollama_query(prompt, self.primary_model, system_prompt)
        
        # Secondary model refinement
        secondary_prompt = f"""
        You are reviewing and refining an analysis or solution.
        
        Original prompt:
        {prompt}
        
        Primary analysis from {self.primary_model}:
        {primary_response}
        
        Please improve, refine, and expand on this analysis. Focus on:
        1. Technical accuracy and completeness
        2. Swift 6 and macOS 15+ compatibility
        3. Audio processing best practices
        4. Providing concrete examples where helpful
        
        Your task is to complement the primary analysis, not just repeat it.
        """
        
        secondary_response = self.ollama_query(secondary_prompt, self.secondary_model)
        
        return primary_response, secondary_response
    
    def analyze_pr(self, pr_number: int) -> str:
        """Analyze a PR using both Ollama models collaboratively."""
        logger.info(f"Analyzing PR #{pr_number}")
        
        # Get PR details using GitHub CLI
        exit_code, pr_json = self.run_command([
            "gh", "pr", "view", str(pr_number), 
            "--json", "title,body,files,commits,reviewDecision",
            "--repo", self.repo
        ])
        
        if exit_code != 0:
            return f"Failed to get PR data: {pr_json}"
        
        try:
            pr_data = json.loads(pr_json)
            
            # Get the diff for more detailed analysis
            exit_code, pr_diff = self.run_command(["gh", "pr", "diff", str(pr_number), "--repo", self.repo])
            
            # Create analysis prompt
            pr_prompt = f"""
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
            
            # System prompt with context
            system_prompt = f"""
            You are a code reviewer for an advanced audio app built with Swift {self.swift_version} 
            for macOS {self.macos_version}+. 
            Focus on technical accuracy, performance, and platform compatibility.
            """
            
            # Get both model analyses
            primary_review, secondary_review = self.collaborative_query(pr_prompt, system_prompt)
            
            # Combine the reviews
            combined = f"""
            # Collaborative PR Review: #{pr_number} - {pr_data.get('title')}
            
            ## Primary Analysis ({self.primary_model})
            
            {primary_review}
            
            ## Enhanced Review ({self.secondary_model})
            
            {secondary_review}
            
            ## Next Steps
            
            1. Review the feedback above
            2. Address critical issues first
            3. Consider performance implications
            4. Ensure Swift 6 and macOS 15+ compatibility
            """
            
            return combined
                
        except Exception as e:
            logger.error(f"Error analyzing PR: {e}")
            return f"Error analyzing PR: {str(e)}"
    
    def fix_issue(self, issue_number: int) -> str:
        """Generate a solution for an issue using dual Ollama models."""
        logger.info(f"Generating solution for issue #{issue_number}")
        
        # Get issue details using GitHub CLI
        exit_code, issue_json = self.run_command([
            "gh", "issue", "view", str(issue_number), 
            "--json", "title,body,labels",
            "--repo", self.repo
        ])
        
        if exit_code != 0:
            return f"Failed to get issue data: {issue_json}"
        
        try:
            issue_data = json.loads(issue_json)
            
            # Create analysis prompt
            issue_prompt = f"""
            Analyze this issue for AudioBloomAI (Swift 6, macOS 15+):
            
            Title: {issue_data.get('title')}
            Description: {issue_data.get('body')}
            Labels: {', '.join(l.get('name', '') for l in issue_data.get('labels', []))}
            
            Provide a solution that:
            1. Addresses the core problem
            2. Is compatible with Swift 6 and macOS 15+
            3. Follows best practices for audio processing
            4. Includes code examples where appropriate
            
            Be specific and actionable.
            """
            
            # System prompt with context
            system_prompt = f"""
            You are a Swift {self.swift_version} developer solving issues for an advanced audio app 
            targeting macOS {self.macos_version}+. 
            Focus on audio processing best practices, performance, and platform compatibility.
            """
            
            # Get both model analyses
            primary_solution, secondary_solution = self.collaborative_query(issue_prompt, system_prompt)
            
            # Combine the solutions
            combined = f"""
            # Collaborative Solution: Issue #{issue_number} - {issue_data.get('title')}
            
            ## Initial Solution ({self.primary_model})
            
            {primary_solution}
            
            ## Enhanced Solution ({self.secondary_model})
            
            {secondary_solution}
            
            ## Implementation Plan
            
            1. Review both solutions above
            2. Create a new branch: `git checkout -b fix/issue-{issue_number}`
            3. Implement the solution, prioritizing critical components
            4. Add tests to verify the fix
            5. Create PR with: `gh pr create --title "Fix #{issue_number}: {issue_data.get('title')}" --body "Fixes #{issue_number}"`
            """
            
            return combined
                
        except Exception as e:
            logger.error(f"Error fixing issue: {e}")
            return f"Error fixing issue: {str(e)}"

    def create_pr(self, title: str, body: str, base_branch: str = "main") -> str:
        """Create a PR with AI-enhanced description using both models."""
        logger.info(f"Creating PR: {title}")
        
        # Get current branch
        exit_code, current_branch = self.run_command(["git", "branch", "--show-current"])
        if exit_code != 0:
            return f"Failed to get current branch: {current_branch}"
        
        current_branch = current_branch.strip()
        
        # Enhance PR description with dual Ollama models
        pr_prompt = f"""
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
        
        system_prompt = f"""
        You are creating documentation for a PR in an audio processing application 
        built with Swift {self.swift_version} for macOS {self.macos_version}+. 
        Focus on clarity, completeness, and technical accuracy.
        """
        
        # Get enhanced descriptions from both models
        primary_desc, secondary_desc = self.collaborative_query(pr_prompt, system_prompt)
        
        # Create a combined description that takes the best from both
        final_prompt = f"""
        You are combining two PR descriptions into a final version. Take the best parts 
        of both descriptions, ensuring the final version is concise but complete.
        
        Original PR information:
        Title: {title}
        Description: {body}
        
        Description 1:
        {primary_desc}
        
        Description 2:
        {secondary_desc}
        
        Create a single cohesive PR description that combines the strengths of both.
        The final description should be well-formatted markdown and include all important details.
        """
        
        enhanced_body = self.ollama_query(final_prompt)
        
        # Create the PR using GitHub CLI
        exit_code, result = self.run_command([
            "gh", "pr", "create",
            "--title", title,
            "--body", enhanced_body,
            "--base", base_branch,
            "--repo", self.repo
        ])
        
        if exit_code != 0:
            return f"Failed to create PR: {result}"
        
        return f"PR created successfully:\n{result}"
    
    def fetch_open_prs(self) -> List[Dict]:
        """Fetch all open PRs from the repository."""
        logger.info(f"Fetching open PRs from {self.repo}")
        
        exit_code, pr_json = self.run_command([
            "gh", "pr", "list",
            "--json", "number,title,updatedAt",
            "--state", "open",
            "--repo", self.repo
        ])
        
        if exit_code != 0:
            logger.error(f"Failed to fetch PRs: {pr_json}")
            return []
            
        try:
            return json.loads(pr_json)
        except json.JSONDecodeError:
            logger.error("Failed to parse PR JSON")
            return []
            
    def fetch_open_issues(self) -> List[Dict]:
        """Fetch all open issues from the repository."""
        logger.info(f"Fetching open issues from {self.repo}")
        
        exit_code, issue_json = self.run_command([
            "gh", "issue", "list",
            "--json", "number,title,updatedAt",
            "--state", "open",
            "--repo", self.repo
        ])
        
        if exit_code != 0:
            logger.error(f"Failed to fetch issues: {issue_json}")
            return []
            
        try:
            return json.loads(issue_json)
        except json.JSONDecodeError:
            logger.error("Failed to parse issue JSON")
            return []
    
    def monitor_repository(self, interval: int = None) -> None:
        """Continuously monitor the repository for new PRs and issues."""
        if interval:
            self.check_interval = interval
            
        logger.info(f"Starting repository monitoring with {self.check_interval}s interval")
        
        try:
            while True:
                # Check for new PRs
                prs = self.fetch_open_prs()
                for pr in prs:
                    pr_number = pr.get('number')
                    if pr_number and pr_number not in self.processed_prs:
                        logger.info(f"Found new PR #{pr_number}: {pr.get('title')}")
                        review = self.analyze_pr(pr_number)
                        self.processed_prs.add(pr_number)
                        
                        # Write review to a file
                        with open(f"pr_review_{pr_number}.md", "w") as f:
                            f.write(review)
                        logger.info(f"Saved PR review to pr_review_{pr_number}.md")
                
                # Check for new issues
                issues = self.fetch_open_issues()
                for issue in issues:
                    issue_number = issue.get('number')
                    if issue_number and issue_number not in self.processed_issues:
                        logger.info(f"Found new issue #{issue_number}: {issue.get('title')}")
                        solution = self.fix_issue(issue_number)
                        self.processed_issues.add(issue_number)
                        
                        # Write solution to a file
                        with open(f"issue_solution_{issue_number}.md", "w") as f:
                            f.write(solution)
                        logger.info(f"Saved issue solution to issue_solution_{issue_number}.md")
                
                logger.info(f"Monitoring cycle complete. Sleeping for {self.check_interval}s")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")


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
    
    # Monitor repository command
    monitor_parser = subparsers.add_parser("monitor", help="Continuously monitor repository for new PRs and issues")
    monitor_parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds (default: 300)")
    
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
    elif args.command == "monitor":
        collaborator.monitor_repository(args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

