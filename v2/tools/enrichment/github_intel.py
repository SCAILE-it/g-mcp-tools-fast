"""GitHub profile intelligence tool for V2 API.

Analyzes GitHub user profiles and repositories.
"""

from typing import Any, Dict

from v2.utils.decorators import enrichment_tool


@enrichment_tool("github-intel")
async def analyze_github_profile(username: str) -> Dict[str, Any]:
    """Analyze a GitHub user profile.

    Args:
        username: GitHub username to analyze

    Returns:
        Dictionary with username, profile info, repo stats, and languages

    Raises:
        Exception: If GitHub API returns error
    """
    import requests

    # Get user data
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")

    user_data = response.json()

    # Get repositories
    repos_url = f"https://api.github.com/users/{username}/repos?per_page=100"
    repos_response = requests.get(repos_url, timeout=10)
    repos = repos_response.json() if repos_response.status_code == 200 else []

    # Analyze languages used
    languages = {}
    for repo in repos[:20]:
        if repo.get("language"):
            lang = repo["language"]
            languages[lang] = languages.get(lang, 0) + 1

    return {
        "username": username,
        "name": user_data.get("name"),
        "bio": user_data.get("bio"),
        "company": user_data.get("company"),
        "location": user_data.get("location"),
        "publicRepos": user_data.get("public_repos"),
        "followers": user_data.get("followers"),
        "following": user_data.get("following"),
        "languages": languages,
        "profileUrl": user_data.get("html_url")
    }
