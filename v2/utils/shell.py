"""Shell command execution utilities for V2 API.

Provides async subprocess execution with timeout support.
"""

import asyncio
from typing import List


async def run_command(cmd: List[str], timeout: int = 30) -> tuple[str, str, int]:
    """Run shell command and return stdout, stderr, returncode.

    Args:
        cmd: Command and arguments as list
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (stdout, stderr, returncode)

    Raises:
        Exception: If command times out or fails to execute
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )
        return stdout.decode(), stderr.decode(), process.returncode
    except asyncio.TimeoutError:
        raise Exception(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    except Exception as e:
        raise Exception(f"Command failed: {e}")
