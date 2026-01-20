"""
Recursively gathers URLs of pages to be stored in a JSON file for Vector DB processing.
Creates 'collected_links.json' for all links including PDF links.
Crawled on Feb 20th, 2024.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import json
import argparse
import sys


def is_valid_url(url):
    """
    Checks if the URL is valid based on its file extension or lack thereof.
    Valid extensions include .htm, .html, .pdf, and directories (ending with /).
    Also considers URLs without a period in the last segment as valid (assuming it leads to a webpage).
    """
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    if path.endswith((".htm", ".html", ".pdf", "/")):
        return True
    if "/" in path:
        last_segment = path.split("/")[-1]
        return "." not in last_segment
    return False


def get_links(
    url,
    base_url,
    visited,
    max_depth,
    current_depth=0,
    links_collected=None,
    pdf_links=None,
):
    """
    Recursively gathers links from a specified URL up to a maximum depth.
    Filters out invalid URLs and stores collected links in a dictionary.
    Separately tracks PDF links.
    """
    if current_depth > max_depth or url in visited:
        return

    visited.add(url)

    if links_collected is None:
        links_collected = {}

    if base_url not in links_collected:
        links_collected[base_url] = []

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(url)
            if base_url != url:
                links_collected[base_url].append(url)

            soup = BeautifulSoup(response.text, "html.parser")

            if url == "https://lti.cs.cmu.edu/people/faculty/index.html":
                links = []
                faculty_divs = soup.find_all("div", class_="filterable")
                for div in faculty_divs:
                    a_tag = div.find("a", href=True)
                    if "categories-1:Core Faculty" in a_tag["data-categories"]:
                        links.append(a_tag)

            else:
                links = soup.find_all("a", href=True)

            for link in links:
                href = link.get("href")
                full_url = urljoin(url, href)
                full_url, _ = urldefrag(full_url)

                if urlparse(full_url).netloc == urlparse(
                    base_url
                ).netloc and is_valid_url(full_url):
                    if (
                        full_url not in links_collected[base_url]
                        and full_url not in visited
                    ):

                        get_links(
                            full_url,
                            base_url,
                            visited,
                            max_depth,
                            current_depth + 1,
                            links_collected,
                            pdf_links,
                        )
    except Exception as e:
        print(f"Error accessing {url}: {e}")

    if current_depth == 0:
        with open("./dataset/collected_links_depth_1.json", "w") as file:
            json.dump(links_collected, file, indent=4)
        # with open("./collected_pdfs.json", "w") as file:
        #     json.dump(pdf_links, file, indent=4)


def main():
    """
    Parses arguments from the command line, reads base URLs from a file,
    and initiates the link collection process.
    """
    parser = argparse.ArgumentParser(description="Collecting links.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="File containing base urls",
        default="./dataset/base_urls.txt",
    )
    parser.add_argument(
        "-d", "--depth", type=int, help="Maximum depth to crawl", default=1
    )
    args = parser.parse_args()

    base_urls = []
    with open(args.file, "r") as file:
        base_urls = [line.strip() for line in file if line.strip()]

    links_collected = {}
    pdf_links = []
    visited = set()

    for base_url in base_urls:
        get_links(
            base_url, base_url, visited, args.depth, 0, links_collected, pdf_links
        )

    print("Finished collecting links")


if __name__ == "__main__":
    main()
