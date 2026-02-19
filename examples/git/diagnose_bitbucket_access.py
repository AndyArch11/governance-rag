#!/usr/bin/env python3
"""
Diagnostic script to check Bitbucket Cloud access and identify workspace names.
"""

import os
import sys
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    username = os.getenv("BITBUCKET_API_USERNAME") or os.getenv("BITBUCKET_USERNAME") or input("BitBucket account email (for API): ")
    password = os.getenv("BITBUCKET_PASSWORD") or input("BitBucket app password: ")
    host = os.getenv("BITBUCKET_HOST", "https://api.bitbucket.org/2.0")
    
    if not username or not password:
        print("[ERROR] Username and password are required")
        sys.exit(1)
    
    auth = HTTPBasicAuth(username, password)
    
    print(f"\n{'='*80}")
    print("BitBucket Cloud Access Diagnostic")
    print(f"{'='*80}")
    print(f"\nUsing API username: {username}")
    print(f"Note: For Bitbucket Cloud API, use your account EMAIL, not Bitbucket username")
    print()
    
    # Test 1: Check authentication
    print("[1/4] Testing authentication...")
    try:
        resp = requests.get(f"{host}/user", auth=auth)
        resp.raise_for_status()
        user_data = resp.json()
        print(f"  ✓ Authentication successful")
        print(f"  - Username: {user_data.get('username')}")
        print(f"  - Display name: {user_data.get('display_name')}")
        print(f"  - UUID: {user_data.get('uuid')}")
    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        print("\n[ERROR] Cannot authenticate. Check your credentials.")
        sys.exit(1)
    
    # Test 2: List workspaces
    print("\n[2/4] Listing workspaces you have access to...")
    try:
        resp = requests.get(f"{host}/workspaces", auth=auth, params={"pagelen": 100})
        resp.raise_for_status()
        workspaces = resp.json().get("values", [])
        
        if workspaces:
            print(f"  ✓ Found {len(workspaces)} workspace(s):")
            for ws in workspaces:
                print(f"    - {ws['slug']} ({ws.get('name', 'N/A')})")
        else:
            print("  ! No workspaces found")
    except Exception as e:
        print(f"  ✗ Failed to list workspaces: {e}")
    
    # Test 3: List repositories (user has access to)
    print("\n[3/4] Listing repositories you have access to...")
    try:
        resp = requests.get(
            f"{host}/user/permissions/repositories",
            auth=auth,
            params={"pagelen": 10, "q": 'permission="write" OR permission="admin"'},
        )
        resp.raise_for_status()
        repos = resp.json().get("values", [])
        
        if repos:
            print(f"  ✓ Found {len(repos)} repository/repositories (showing first 10):")
            workspaces_found = set()
            for item in repos:
                repo = item.get("repository", {})
                full_name = repo.get("full_name", "")
                workspace = full_name.split("/")[0] if "/" in full_name else "unknown"
                workspaces_found.add(workspace)
                print(f"    - {full_name}")
            
            print(f"\n  Workspaces with repositories:")
            for ws in sorted(workspaces_found):
                print(f"    - {ws}")
        else:
            print("  ! No repositories found")
    except Exception as e:
        print(f"  ✗ Failed to list repositories: {e}")
    
    # Test 4: Try specific workspace (if provided)
    workspace = input("\n[4/4] Enter workspace name to test (or press Enter to skip): ").strip()
    if workspace:
        print(f"\nTesting access to workspace: {workspace}")
        try:
            resp = requests.get(
                f"{host}/repositories/{workspace}",
                auth=auth,
                params={"pagelen": 10},
            )
            resp.raise_for_status()
            repos = resp.json().get("values", [])
            print(f"  ✓ Can list repositories in '{workspace}' workspace")
            print(f"  - Found {len(repos)} repository/repositories (first page)")
            for repo in repos:
                print(f"    - {repo['slug']}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"  ✗ 401 Unauthorised - No access to workspace '{workspace}'")
                print("     The workspace name might be incorrect, or you don't have permission")
            elif e.response.status_code == 404:
                print(f"  ✗ 404 Not Found - Workspace '{workspace}' does not exist")
            else:
                print(f"  ✗ Error: {e}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")
    print("\n1. For Bitbucket Cloud, use DIFFERENT usernames for git vs API:")
    print(f"   - Git operations: Use your Bitbucket username")
    print(f"   - API operations: Use your account email ({username})")
    print("\n2. Set environment variables:")
    print(f"   export BITBUCKET_USERNAME=<bitbucket-username>  # For git")
    print(f"   export BITBUCKET_API_USERNAME={username}        # For API")
    print("   export BITBUCKET_PASSWORD=<app-password>")
    print("\n3. Or use command-line arguments:")
    print("   --username <bitbucket-username> --api-username <email>")
    print("\n4. Use the workspace names listed above with --project")
    print("\n5. If workspace listing fails, the script will automatically fall back")
    print("   to listing repositories you have explicit access to")
    print("\nExample command:")
    if workspaces:
        ws_example = workspaces[0]['slug']
        print(f"  python scripts/ingest/ingest_bitbucket.py \\")
        print(f"    --username myuser --api-username {username} \\")
        print(f"    --project {ws_example} --repo-pattern <pattern> --is-cloud")
    print()

if __name__ == "__main__":
    main()
