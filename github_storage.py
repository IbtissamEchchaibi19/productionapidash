# github_storage.py

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

import os
import base64
import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime


class GitHubCSVStorage:
    def __init__(self, repo_owner: str, repo_name: str, token: str, csv_filename: str = "invoice_data.csv"):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token
        self.csv_filename = csv_filename
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }

    def _make_request(self, method: str, url: str, data: dict = None) -> requests.Response:
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            return response
        except Exception as e:
            print(f"Error making GitHub API request: {e}")
            raise

    def get_file_content(self) -> tuple[Optional[str], Optional[str]]:
        url = f"{self.base_url}/contents/{self.csv_filename}"
        try:
            response = self._make_request('GET', url)
            if response.status_code == 200:
                file_data = response.json()
                content = base64.b64decode(file_data['content']).decode('utf-8')
                return content, file_data['sha']
            elif response.status_code == 404:
                print(f"CSV file {self.csv_filename} not found in repository")
                return None, None
            else:
                print(f"Error fetching file: {response.status_code} - {response.text}")
                return None, None
        except Exception as e:
            print(f"Error getting file content: {e}")
            return None, None

    def upload_csv_content(self, content: str, sha: Optional[str] = None, commit_message: str = None) -> bool:
        url = f"{self.base_url}/contents/{self.csv_filename}"
        if not commit_message:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"Update invoice data - {timestamp}"
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        data = {
            'message': commit_message,
            'content': encoded_content
        }
        if sha:
            data['sha'] = sha
        try:
            response = self._make_request('PUT', url, data)
            if response.status_code in [200, 201]:
                print(f"Successfully {'updated' if sha else 'created'} {self.csv_filename}")
                return True
            else:
                print(f"Error uploading file: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error uploading CSV: {e}")
            return False

    def read_csv_as_dataframe(self) -> Optional[pd.DataFrame]:
        content, _ = self.get_file_content()
        if content:
            try:
                from io import StringIO
                return pd.read_csv(StringIO(content))
            except Exception as e:
                print(f"Error parsing CSV content: {e}")
                return None
        return None

    def append_data_to_csv(self, new_data: List[Dict]) -> bool:
        if not new_data:
            print("No new data to append")
            return True
        existing_content, sha = self.get_file_content()
        if existing_content:
            try:
                from io import StringIO
                df_existing = pd.read_csv(StringIO(existing_content))
                df_new = pd.DataFrame(new_data)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception as e:
                print(f"Error processing existing CSV: {e}")
                return False
        else:
            df_combined = pd.DataFrame(new_data)
            sha = None
        csv_content = df_combined.to_csv(index=False)
        commit_message = f"Add {len(new_data)} new invoice records"
        return self.upload_csv_content(csv_content, sha, commit_message)

    def update_entire_csv(self, dataframe: pd.DataFrame, commit_message: str = None) -> bool:
        _, sha = self.get_file_content()
        csv_content = dataframe.to_csv(index=False)
        if not commit_message:
            commit_message = f"Update complete CSV with {len(dataframe)} records"
        return self.upload_csv_content(csv_content, sha, commit_message)

    def delete_records_by_condition(self, condition_func) -> bool:
        df = self.read_csv_as_dataframe()
        if df is None:
            print("No CSV data found to delete from")
            return False
        original_count = len(df)
        df_filtered = df[~df.apply(condition_func, axis=1)]
        deleted_count = original_count - len(df_filtered)
        if deleted_count > 0:
            commit_message = f"Delete {deleted_count} invoice records"
            return self.update_entire_csv(df_filtered, commit_message)
        else:
            print("No records matched deletion condition")
            return True

    def get_raw_csv_url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repo_owner}/{self.repo_name}/main/{self.csv_filename}"


class GitHubConfig:
    def __init__(self):
        self.repo_owner = os.getenv('GITHUB_REPO_OWNER')
        self.repo_name = os.getenv('GITHUB_REPO_NAME')
        self.token = os.getenv('GITHUB_TOKEN')
        self.csv_filename = os.getenv('GITHUB_CSV_FILENAME', 'honey_production_data.csv')
        if not all([self.repo_owner, self.repo_name, self.token]):
            missing = []
            if not self.repo_owner: missing.append('GITHUB_REPO_OWNER')
            if not self.repo_name: missing.append('GITHUB_REPO_NAME')
            if not self.token: missing.append('GITHUB_TOKEN')
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def get_storage_instance(self) -> GitHubCSVStorage:
        return GitHubCSVStorage(
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
            token=self.token,
            csv_filename=self.csv_filename
        )
