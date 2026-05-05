from agents import Agent, Runner
from agents.tool import function_tool
import subprocess
import ast


@function_tool
def read_file(path: str) -> str:
    """Read a file from the repository."""
    with open(path, 'r') as f:
        return f.read()


@function_tool
def write_file(path: str, content: str) -> str:
    """Write code to a file."""
    with open(path, 'w') as f:
        f.write(content)
    return f'Written {len(content)} bytes to {path}'


@function_tool
def run_tests(test_path: str = 'tests/') -> str:
    """Run pytest and return results."""
    result = subprocess.run(
        ['python', '-m', 'pytest', test_path, '-v', '--tb=short'],
        capture_output=True, text=True, timeout=120,
    )
    return f'Exit code: {result.returncode}\n{result.stdout}\n{result.stderr}'


@function_tool
def search_code(pattern: str, directory: str = '.') -> str:
    """Search codebase for a pattern using ripgrep."""
    result = subprocess.run(
        ['rg', pattern, directory, '--type', 'py', '-n', '-C', '2'],
        capture_output=True, text=True,
    )
    return result.stdout[:5000]


@function_tool
def git_diff() -> str:
    """Get the current git diff of changes."""
    result = subprocess.run(
        ['git', 'diff', '--stat'], capture_output=True, text=True
    )
    return result.stdout


@function_tool
def submit_pr(title: str, body: str, branch: str) -> str:
    """Create a pull request on GitHub."""
    subprocess.run(['git', 'checkout', '-b', branch])
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', title])
    subprocess.run(['git', 'push', 'origin', branch])
    # Use GitHub CLI to create PR
    result = subprocess.run(
        ['gh', 'pr', 'create', '--title', title, '--body', body],
        capture_output=True, text=True,
    )
    return result.stdout


def create_coding_agent():
    return Agent(
        name='SWE-Agent',
        instructions="""You are an expert software engineer.
        Follow this workflow for every issue:
        1. Analyze the issue and plan your approach
        2. Search the codebase to understand relevant context
        3. Implement the fix with proper error handling
        4. Write or update tests to cover the change
        5. Run tests to verify everything passes
        6. Review your diff before submitting
        7. Submit a PR with a clear description""",
        tools=[read_file, write_file, run_tests, search_code, git_diff, submit_pr],
        model='gpt-4o',
    )


def solve_issue(issue_text: str):
    agent = create_coding_agent()
    result = Runner.run_sync(agent, issue_text)
    return result


if __name__ == '__main__':
    issue = 'Fix: API returns 500 when user email contains unicode characters'
    result = solve_issue(issue)
    print(result)
