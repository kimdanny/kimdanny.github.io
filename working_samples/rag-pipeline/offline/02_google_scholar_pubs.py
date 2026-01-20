"""
Scrapes Google Scholar for paper details for Core Faculty.
"""

import requests
import time
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_driver():
    """
    Returns a Chrome WebDriver with the Selenium Stealth plugin enabled.
    """

    options = webdriver.ChromeOptions()

    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)

    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )

    return driver


def no_profile_paper_details(base_url, driver):
    """
    Scrapes paper details from Google Scholar for a given author (Rita Singh) without profile ID.
    """
    papers = []
    start = 0

    while True:
        url = base_url.format(start=start)

        driver.get(url)
        time.sleep(30)

        page_source = driver.page_source

        soup = BeautifulSoup(page_source, "html.parser")

        for paper in soup.find_all("div", class_="gs_ri"):
            title = paper.find("h3", class_="gs_rt").find("a").get_text()
            link_div = paper.find("h3", class_="gs_rt")
            link = link_div.find("a")["href"] if link_div else None
            citation_count = 0

            authors, abstract = None, None

            print("Scraping paper:", title)

            if title == "Leveraging Heterogeneity in Time-to-Event Predictions":
                return papers

            bot_info = paper.find_all("div", class_="gs_fl gs_flb")

            if bot_info:
                for info in bot_info:
                    if "Cited by" in info.text:
                        citation_count = int(info.text.split("</a>")[0].split()[4])
                        break
            # if the link is an arxiv, go to thie link and get authors and asbtract
            if "arxiv" in link:
                soup = BeautifulSoup(requests.get(link).content, "html.parser")

                authors = soup.find("div", class_="authors").find_all("a")
                authors = ", ".join([author.get_text() for author in authors])

                abstract = soup.find("blockquote", class_="abstract mathjax").get_text()

            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "url": link,
                    "citation_count": citation_count,
                }
            )

        start += 10


def get_paper_details(intermediate_url, driver):
    """
    Scrapes paper details from an intermediate URL (the google scholar view url).
    """
    driver.get(intermediate_url)
    time.sleep(5)

    page_source = driver.page_source

    intermediate_soup = BeautifulSoup(page_source, "html.parser")

    author_info, journal_info = None, None

    link_div = intermediate_soup.find("div", class_="gsc_oci_title_ggi")
    link = link_div.find("a")["href"] if link_div else None

    for gs_scl_div in intermediate_soup.find_all("div", class_="gs_scl"):
        field_div = gs_scl_div.find("div", class_="gsc_oci_field")
        value_div = gs_scl_div.find("div", class_="gsc_oci_value")

        if field_div and value_div:
            field = field_div.get_text(strip=True)
            value = value_div.get_text(strip=True)
            if field == "Authors":
                author_info = value
            elif field == "Journal" or field == "Conference" or field == "Book":
                journal_info = value

    description_div = intermediate_soup.find("div", class_="gsh_small")
    abstract = description_div.get_text(strip=True) if description_div else None

    return {
        "authors": author_info,
        "publicationVenue": journal_info,
        "url": link,
        "abstract": abstract,
    }


def scrape_google_scholar(author_profile_id, target_year, driver):
    """
    Scrapes Google Scholar for a given author's profile ID.
    """
    base_url = f"https://scholar.google.com/citations?hl=en&user={author_profile_id}&view_op=list_works&sortby=pubdate"
    papers = []
    start = 0

    while True:
        url = f"{base_url}&cstart={start}"
        driver.get(url)
        time.sleep(30)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".gsc_a_tr"))
            )
        except:
            print("Timed out waiting for page to load")

        page_source = driver.page_source

        soup = BeautifulSoup(page_source, "html.parser")

        author_name = soup.find("div", id="gsc_prf_in").text

        for paper_html in soup.find_all("tr", class_="gsc_a_tr"):
            year_span = paper_html.select_one(".gsc_a_y span")
            year = year_span.text if year_span else None

            if year and int(year) <= (target_year - 1):
                print(f"Found year {year}, stopping scrape")
                return papers, author_name

            if year and year == str(target_year):
                title = paper_html.find("a", class_="gsc_a_at").text
                link_suffix = paper_html.find("a", class_="gsc_a_at")["href"]
                intermediate_link = f"https://scholar.google.com{link_suffix}"

                print(f"Scraping details for {title}")

                paper_details = get_paper_details(intermediate_link, driver)

                papers.append(
                    {
                        "title": title,
                        "authors": paper_details["authors"],
                        "year": int(year),
                        "abstract": paper_details["abstract"],
                        "url": paper_details["url"],
                        "publicationVenue": paper_details["publicationVenue"],
                        "citation_count": paper_html.find(
                            "a", class_="gsc_a_ac"
                        ).text.strip()
                        or "0",
                    }
                )

        show_more_button = soup.find("button", id="gsc_bpf_more")

        is_disabled = (
            show_more_button.has_attr("disabled") if show_more_button else True
        )

        if is_disabled:
            break

        start += 20

    return papers, author_name


def read_scholar_ids():
    """
    Reads scholar IDs from a text file named 'google_scholar_ids.txt'.
    Returns a list of scholar IDs.
    """
    with open("./dataset/google_scholar_ids.txt", "r") as file:
        scholar_ids = file.read().splitlines()
    return list(scholar_ids)


if __name__ == "__main__":
    author_profile_id = read_scholar_ids()
    target_year = 2023
    pub_data = []

    for author in author_profile_id:
        driver = get_driver()
        print(f"Scraping Google Scholar for author {author}")
        papers_from_target_year, author_name = scrape_google_scholar(
            author, target_year, driver
        )
        pub_data.append({author_name: papers_from_target_year})
        print(f"Scraped {len(papers_from_target_year)} papers for {author_name}")
        driver.quit()

    # special case fo Rita Singh
    driver = get_driver()
    papers = no_profile_paper_details(
        "https://scholar.google.com/scholar?start={start}&q=rita+singh+carnegie+mellon+university&hl=en&as_sdt=0,39&as_ylo=2023&as_yhi=2023",
        driver,
    )
    pub_data.append({"Rita Singh": papers})
    driver.quit()

    with open(f"./dataset/google_scholar_pub_ritas.json", "w") as file:
        json.dump(pub_data, file, indent=4)
