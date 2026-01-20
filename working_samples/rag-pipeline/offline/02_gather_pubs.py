"""
Gather publications of faculty to be stored in Vector DB
Creates 'faculty.json' file and 'faculty_pubs.json' in the designated directory
Crawled on Feb 23rd 2024
"""

import requests
import json

API_KEY = "G8lNHi9zlU50oAwiHt5x66KmY9zlzA2P27YcNynN"
AUTHOR_DETAILS_URL = "https://api.semanticscholar.org/graph/v1/author/{author_id}"
PAPER_DETAILS_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"


def get_author_details_by_id(author_id):
    """
    Fetch author details from Semantic Scholar API using the author's ID.
    Returns the author details as a JSON object if successful, or None upon failure.
    """
    url = AUTHOR_DETAILS_URL.format(author_id=author_id)
    params = {
        "fields": "authorId,externalIds,url,name,aliases,affiliations,homepage,paperCount,citationCount,hIndex,papers.paperId,papers.year"
    }
    headers = {"x-api-key": API_KEY}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        print(f"Fetched author details for {author_id}")
        return response.json()
    else:
        print(f"Error fetching author details for {author_id}")
        return None


def get_paper_details_by_id(paper_id, seen_titles):
    """
    Fetch paper details from Semantic Scholar API using the paper's ID.
    Filters out duplicates and non-open access papers.
    Returns the paper details as a JSON object if successful, or None upon failure.
    """
    url = PAPER_DETAILS_URL.format(paper_id=paper_id)
    params = {
        "fields": "title,abstract,authors,publicationVenue,year,tldr,url,isOpenAccess,openAccessPdf,citationCount"
    }
    headers = {"x-api-key": API_KEY}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        paper_details = response.json()
        title = paper_details.get("title")

        is_open_access = paper_details.get("isOpenAccess", False)

        if title not in seen_titles and is_open_access:
            print(f"Fetched details for paper ID {paper_id}")
            seen_titles.add(title)
            del paper_details["isOpenAccess"]
            return paper_details
    else:
        print(f"Error fetching details for paper ID {paper_id}")
        return None


def read_author_ids():
    """
    Reads author IDs from a text file named 'author_ids.txt'.
    Returns a list of author IDs.
    """
    with open("./dataset/author_ids.txt", "r") as file:
        author_ids = file.read().splitlines()
    return author_ids


def main():
    """
    Main function to fetch and store publication details of faculty members.
    It creates 'faculty.json' for storing faculty publications and
    'faculty_pubs.json' for storing open access PDF links.
    """
    author_ids = read_author_ids()
    print("Fetched author ids from file")

    faculty_details_dict = {}
    open_access_pdfs = {}
    seen_titles = set()

    for author_id in author_ids:
        author_details = get_author_details_by_id(author_id)

        if author_details:
            papers_in_2023 = [
                paper
                for paper in author_details.get("papers", [])
                if paper["year"] == 2023
            ]
            paper_details_list = []

            for paper in papers_in_2023:
                paper_detail = get_paper_details_by_id(paper["paperId"], seen_titles)
                if paper_detail:
                    paper_details_list.append(paper_detail)
                    open_access_pdf = paper_detail.get("openAccessPdf")
                    if open_access_pdf:
                        faculty_name = author_details["name"]
                        if faculty_name not in open_access_pdfs:
                            open_access_pdfs[faculty_name] = []
                        open_access_pdfs[faculty_name].append(open_access_pdf)

            author_details["papers"] = paper_details_list
            faculty_details_dict[author_details["name"]] = author_details

    with open("./dataset/faculty.json", "w") as outfile:
        json.dump(faculty_details_dict, outfile, indent=4)

    with open("./dataset/faculty_pubss.json", "w") as pdf_outfile:
        json.dump(open_access_pdfs, pdf_outfile, indent=4)


if __name__ == "__main__":
    main()
