import requests
import re
import bs4
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Union
import sys
import os
import subprocess
import random


def _get_html_content(url: str) -> Union[bs4.BeautifulSoup, int]:
    """
    Returns HTML content represented by BeautifulSoup class
    """
    response = requests.get(url, headers=get_user_agent())
    if response.status_code == 200:
        html_source = response.text
        return BeautifulSoup(html_source, "html.parser")
    print(f"Warning: failed to fetch {url}: response code: {response.status_code}")
    return -1


def _build_table_string(table_data) -> str:
    """
    Transform the table_data into the string
        "{'column' : [<column name>, ...], 'row': [<entry> ...]}<SEP> ..."

    <SEP> character will be used for splitting criteria when making passages from document
    """
    final_str = ""
    columns: list = table_data["columns"]
    rows: List[list] = table_data["rows"]
    for row in rows:
        final_str += f"{{'column': {str(columns)}, 'row': {str(row)}}}<SEP>"

    return final_str


def _process_table(table: bs4.element.Tag) -> str:
    """
    :param table - whole html table element represented by bs4.element.Tag class

    Returns json formatted string of table (prediction model consumes json better for structured data).
    One row will have the following format:
        "{'column' : [<column name>, ...], 'row': [<entry> ...]}<SEP> ..."
    """
    # Find and extract tables
    # table_data's 'rows' value will be list of list
    table_data = {"columns": [], "rows": []}
    # Extract rows
    rows = table.find_all("tr")
    column_row_found = False
    for row in rows:
        cells = row.find_all(["th", "td"])
        # skip until we find the column
        if (not column_row_found) and all(
            cell.get_text().strip() == "" for cell in cells
        ):
            continue
        # Use the first non-empty row as column names
        if not table_data["columns"]:
            for cell in cells:
                cell_text = cell.get_text().strip()
                if cell_text:
                    table_data["columns"].append(cell_text)
                    column_row_found = True
        else:
            # Data rows
            row_data = []
            for cell in cells:
                row_data.append(cell.get_text().strip())
            table_data["rows"].append(row_data)

    return _build_table_string(table_data)


def get_document(url: str, process_table=True) -> Tuple[str, str]:
    """
    Return a document given url.
    If process_table is set to True,
        process (one or more) tables in the document separately
    Else return empty string in the second argument
    """
    soup = _get_html_content(url=url)
    if soup == -1:
        raise Exception
    tables_string = ""
    if process_table:
        tables = soup.find_all("table")
        if len(tables) != 0:
            for table in tables:
                table_string: str = _process_table(table)
                tables_string += table_string

        # After finishing processing tables, remove table tags from soup as well as others
        for script in soup(["table", "script", "style", "meta"]):
            script.extract()
    else:
        # Remove unnecessary tags
        for script in soup(["script", "style", "meta"]):
            script.extract()

    # Get text from the remaining HTML
    text = soup.get_text(separator="\n", strip=True)
    # Replace multiple spaces or lines with one space
    text = re.sub(r"\s+", " ", text)

    return text, tables_string


def get_user_agent():
    """
    Get random user agent from the list
    """
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    ]
    return {"User-Agent": random.choice(user_agents)}


def get_unique_pdfs(json_dict: dict) -> list:
    """
    Get json dict loaded from google_scholar_pubs.json file
    and return a unique pdf urls
    """
    result = set()
    for faculty in json_dict:
        unique_urls = set([paper["url"] for paper in json_dict[faculty]])
        result.update(unique_urls)
    return list(result)


def wget_file_to_dir(url, download_path, custom_file_name):
    try:
        subprocess.run(
            [
                "wget",
                "-P",
                download_path,
                "-O",
                os.path.join(download_path, custom_file_name),
                url,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to download file: {e}")
    except Exception as e:
        print(f"Error: {e}")


def remove_downloaded_file(fp):
    try:
        os.remove(fp)
    except FileNotFoundError:
        print("Error: File not found.")
    except PermissionError:
        print("Error: Permission denied to remove the file.")
    except Exception as e:
        print(f"Error: {e}")


# a, b = get_document(url="https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_summer_1.htm")
