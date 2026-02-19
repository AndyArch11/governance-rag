#!/usr/bin/env python3
"""List projects and repositories in a Bitbucket workspace.

This helps you find the correct repository slug when repos are organised in projects.
"""

import os
import sys
import requests
from getpass import getpass


def list_projects_and_repos():
    """List all projects and their repositories in the workspace."""
    
    # Get credentials
    username = os.getenv("BITBUCKET_USERNAME") or input("Bitbucket username: ")
    password = os.getenv("BITBUCKET_PASSWORD") or getpass("Bitbucket API token: ")
    
    workspace = "myworkspace"  # Update with your Bitbucket workspace ID
    
    print(f"\n🔍 Exploring Bitbucket workspace: {workspace}\n")
    
    # List all projects in the workspace
    print("=" * 70)
    print("PROJECTS IN WORKSPACE")
    print("=" * 70)
    
    url = f"https://api.bitbucket.org/2.0/workspaces/{workspace}/projects"
    
    try:
        response = requests.get(url, auth=(username, password), params={"pagelen": 100})
        response.raise_for_status()
        
        projects_data = response.json()
        projects = projects_data.get('values', [])
        
        if not projects:
            print("⚠️  No projects found in workspace")
            return False
        
        print(f"Found {len(projects)} projects:\n")
        
        for project in projects:
            key = project.get('key', 'N/A')
            name = project.get('name', 'N/A')
            description = project.get('description', '')
            
            print(f"📁 Project: {key}")
            print(f"   Name: {name}")
            if description:
                print(f"   Description: {description[:60]}...")
            
            # List repositories in this project
            print(f"   Repositories in {key}:")
            
            # Get repos for this project
            repos_url = f"https://api.bitbucket.org/2.0/repositories/{workspace}"
            repos_response = requests.get(
                repos_url, 
                auth=(username, password),
                params={"q": f'project.key="{key}"', "pagelen": 100}
            )
            
            if repos_response.status_code == 200:
                repos_data = repos_response.json()
                repos = repos_data.get('values', [])
                
                if repos:
                    for repo in repos:
                        slug = repo.get('slug', 'unknown')
                        updated = repo.get('updated_on', 'N/A')[:10]
                        print(f"      📦 {slug:45} (updated: {updated})")
                else:
                    print(f"      (no repositories)")
            else:
                print(f"      (could not list repositories)")
            
            print()  # Blank line between projects
        
        print("\n" + "=" * 70)
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ FAILED: HTTP {e.response.status_code}")
        print(f"   Error: {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED: Network error - {e}")
        return False


if __name__ == "__main__":
    success = list_projects_and_repos()
    sys.exit(0 if success else 1)
