#! /usr/bin/env python3

from typing import List, Dict, Optional
import subprocess
import logging
from pathlib import Path
from xml.sax.saxutils import escape
from dataclasses import dataclass
import time

import boto3

AWS_REGION = 'us-east-1'

MODEL_ID = 'us.anthropic.claude-3-sonnet-20240229-v1:0'

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION
)
        

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add this constant near the top of the file, after other imports and constants
MAX_PROMPT_SIZE = 14000  # Adjust this value as needed

@dataclass
class FileInfo:
    path: str
    content: str
    history: str
    diff: str

@dataclass
class CommitOptions:
    include_content: bool
    include_history: bool
    context_lines: int
    debug: bool

def setup_logging(debug: bool = False) -> None:
    """Configure logging level and format."""
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(level)

def escape_xml_content(content: str) -> str:
    """Escape special characters for XML content."""
    return escape(content, {'"': '&quot;', "'": '&apos;'})

def create_xml_element(tag: str, content: str, attributes: Dict[str, str] = None) -> str:
    """Create an XML element with optional attributes."""
    attrs = '' if attributes is None else ' ' + ' '.join(f'{k}="{v}"' for k, v in attributes.items())
    return f"<{tag}{attrs}>{escape_xml_content(content)}</{tag}>"

def get_staged_files() -> List[str]:
    """Get list of staged files."""
    try:
        # First, get the list of all changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        # Filter for staged files (those starting with 'A', 'M', 'R', or 'D')
        files = list(set([line.split()[-1] for line in result.stdout.splitlines() 
                          if line.startswith(('A', 'M', 'R', 'D'))]))
        logger.info(f"Found {len(files)} staged files")
        return files
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get staged files: {e.stderr}")
        return []

def get_file_content(file_path: str) -> str:
    """Get the current content of a file."""
    try:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return ""
        
        # Should handle different encodings and binary files
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Could not read {file_path} as UTF-8, skipping content")
            return ""
        
        logger.debug(f"Read {len(content)} characters from {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return ""

def get_staged_diff(file_path: Optional[str] = None, context_lines: int = 3) -> str:
    """Get the diff of staged changes for a specific file or all files."""
    try:
        cmd = ['git', 'diff', '--staged', f'-U{context_lines}']
        if file_path:
            cmd.append(file_path)
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting git diff: {e.stderr}")
        return f"Error: Failed to get git diff: {e.stderr}"

def get_file_history(file_path: str, max_entries: int = 5) -> str:
    """Get the git history for a file."""
    try:
        result = subprocess.run(
            ['git', 'log', f'-{max_entries}', '-p', file_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        if "does not have any commits yet" in e.stderr or "no such path" in e.stderr:
            logger.info(f"No commit history for {file_path} (new file or repository)")
            return "New file"
        logger.error(f"Error getting file history for {file_path}: {e.stderr}")
        return ""

def get_file_info(file_path: str) -> FileInfo:
    """Collect all information about a file."""
    return FileInfo(
        path=file_path,
        content=get_file_content(file_path),
        history=get_file_history(file_path),
        diff=get_staged_diff(file_path)
    )

def format_file_info_xml(file_info: FileInfo, options: CommitOptions) -> str:
    """Format file information as XML."""
    status = "new" if file_info.history == "New file" else "modified"
    xml = f"""<file path="{file_info.path}" status="{status}">
    <diff>{escape_xml_content(file_info.diff)}</diff>"""
    
    if options.include_history:
        xml += f"\n    <history>{escape_xml_content(file_info.history[:500])}</history>"
    if options.include_content:
        xml += f"\n    <content>{escape_xml_content(file_info.content[:1000])}</content>"
    
    xml += "\n</file>"
    return xml

def invoke_text_model(prompt: str, max_retries: int = 3) -> Optional[str]:
    """Invoke a text model via Amazon Bedrock Converse API."""
    for attempt in range(max_retries):
        try:
            response = bedrock_runtime.converse(
                modelId=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [ { "text": prompt } ]
                    }
                ]
            )
            
            return response['output']['message']['content'][0]['text']
        
        except boto3.client('bedrock-runtime').exceptions.ThrottlingException:
            if attempt == max_retries - 1:
                logger.error("Max retries reached for rate limit")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Error invoking model: {str(e)}")
            return None

def generate_commit_prompt(files_info: List[FileInfo], options: CommitOptions) -> str:
    """Generate a prompt to create a commit message."""
    files_xml = "\n".join(format_file_info_xml(info, options) for info in files_info)
    
    is_initial_commit = all(not info.history for info in files_info)
    
    if is_initial_commit:
        context = "This is the initial commit in the repository."
    else:
        context = "This is a subsequent commit in the repository."
    
    prompt = f"""<commit_request>
    <instructions>
    You are an expert developer reviewing code changes for a commit message. 
    {context}
    Based on the following file changes, generate a clear and informative git commit message following these guidelines:
    - Start with a concise summary line (max 50 chars)
    - Add detailed explanation after a blank line
    - Use bullet points for multiple changes
    - Mention the purpose of each file being added or modified
    - Highlight any important setup, configuration, or architectural changes
    - For subsequent commits, focus on what has changed since the last commit
    </instructions>
    
    <changed_files>
    {files_xml}
    </changed_files>
    </commit_request>
    
    Please generate a commit message that best describes these changes."""
    
    if len(prompt) > MAX_PROMPT_SIZE:
        truncation_message = f"\n\nNote: The prompt has been truncated due to size limitations. Some file information may have been omitted."
        prompt = prompt[:MAX_PROMPT_SIZE - len(truncation_message)] + truncation_message
        logger.warning(f"Prompt exceeded maximum size and was truncated to {MAX_PROMPT_SIZE} characters.")
    
    return prompt

def commit_changes(message: str) -> bool:
    """Commit changes with the given message."""
    if not message or message.isspace():
        logger.error("Empty commit message provided")
        return False
        
    try:
        result = subprocess.run(
            ['git', 'commit', '-m', message],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully committed changes: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to commit changes: {e.stderr}")
        return False

def is_git_repository() -> bool:
    """Check if the current directory is a Git repository."""
    try:
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], 
                       capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def get_non_staged_changes() -> List[str]:
    """Get list of non-staged changes."""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        # Filter for non-staged files (those starting with ' M', '??', ' D', etc.)
        files = [line.split()[-1] for line in result.stdout.splitlines() 
                 if not line.startswith(('A', 'M', 'R', 'D'))]
        logger.info(f"Found {len(files)} non-staged changes")
        return files
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get non-staged changes: {e.stderr}")
        return []

def prompt_for_override() -> bool:
    """Prompt the user to override the non-staged changes warning."""
    response = input("There are non-staged changes. Do you want to continue anyway? (y/n): ").lower()
    return response == 'y'

def main(options: CommitOptions) -> None:
    """Main function to generate git commit message."""
    setup_logging(options.debug)
    
    # Check if the current directory is a Git repository
    if not is_git_repository():
        logger.error("This script must be run in a Git repository.")
        return

    # Get staged files
    staged_files = get_staged_files()
    
    # Check for non-staged changes
    non_staged_changes = get_non_staged_changes()

    if not staged_files and not non_staged_changes:
        logger.info("No changes found in the repository.")
        return
    
    if not staged_files:
        logger.error("No staged changes found.")
        if non_staged_changes:
            print("There are non-staged changes in the repository:")
            for file in non_staged_changes:
                print(f"  - {file}")
            print("Please stage your changes before running this script.")
        return

    if non_staged_changes:
        logger.warning("There are non-staged changes in the repository.")
        print("Non-staged files:")
        for file in non_staged_changes:
            print(f"  - {file}")
        
        if not prompt_for_override():
            logger.info("Operation cancelled by user.")
            return

    try:
        # Collect information about each file
        files_info = [get_file_info(file_path) for file_path in staged_files]
        
        # Generate prompt
        prompt = generate_commit_prompt(files_info, options)
        
        # Get commit message
        commit_message = invoke_text_model(prompt)
        
        if commit_message:
            logger.info("\nSuggested commit message:")
            print("-" * 50)
            print(commit_message)
            print("-" * 50)
            
            if input("\nWould you like to use this commit message? (y/n): ").lower() == 'y':
                if commit_changes(commit_message):
                    print("Changes committed successfully!")
                else:
                    print("Failed to commit changes. Please check the logs for details.")
        else:
            logger.error("Failed to generate commit message.")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if options.debug:
            raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate git commit messages using an AI model.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--include-content', action='store_true', 
                       help='Include file content in the prompt (disabled by default)')
    parser.add_argument('--include-history', action='store_true',
                       help='Include file history in the prompt (disabled by default)')
    parser.add_argument('--context-lines', type=int, default=3,
                       help='Number of context lines to show in git diff (default: 3)')
    args = parser.parse_args()
    
    options = CommitOptions(
        include_content=args.include_content,
        include_history=args.include_history,
        context_lines=args.context_lines,
        debug=args.debug
    )
    
    main(options)
