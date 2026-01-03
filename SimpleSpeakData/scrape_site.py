from lxml import etree          # pip install lxml
from io import BytesIO
import os
from selenium import webdriver  # pip install selenium
from selenium.webdriver.common.by import By
import re
import mmls2csv
import requests                 # pip install requests
Response = requests.models.Response

# for debugging
import sys
sys.stdout.reconfigure(encoding='utf-8')


def do_redirect(url: str) -> Response:
    '''Check for a redirect and return the response"'''
    response = requests.get(url, verify=True)   # change to false if SSL error
    if response.status_code != 200:
        raise Exception(f"Bad status_code {response.status_code} for {url}")
    page_text = response.text
    for _, meta in etree.iterparse(BytesIO(page_text.encode('utf-8')),
                                tag='meta',
                                remove_blank_text=True,
                                remove_comments=True,
                                encoding='UTF-8',
                                html=True):
        # pick apart the string meta http-equiv="refresh" content="0; URL=\'frontmatter-1.html\'
        if meta.get("http-equiv") == "refresh":
            url = f"{url}/{meta.get('content').lower().split("url='", 1)[1][0:-1]}"
            response = requests.get(url, verify=True)  # change to false if SSL error
            if response.status_code == 200:
                return do_redirect(url)
            else:
                raise Exception(f"Bad status_code {response.status_code} for {url}")
    return response


def scrape_domain(domain: str, url: str, visited_links: set[str], depth=1, max_depth=100) -> set[str]:
    '''Scrape the `url` and its links that remain within `domain`.
    '''
    if url in visited_links:
        return visited_links
    if depth > max_depth:
        print(f"max depth={max_depth} reached")
        return visited_links
    visited_links.add(url)
    print(f"requesting url='{url}'")
    try:
        response = do_redirect(url)
    except Exception as e:
        print(f"URL request caused an error:\n{e}\n  Original URL {url}")
        return visited_links
    if response.status_code != 200:
        print(f"...response ({response}) was bad.")
        return visited_links
    html_text = response.text
    # find all the anchor tags with "href"
    for _, link in etree.iterparse(BytesIO(html_text.encode('utf-8')),
                                   tag='a',
                                   remove_blank_text=True,
                                   remove_comments=True,
                                   encoding='UTF-8',
                                   html=True
                                   ):
        link = link.get('href')
        if link is None:
            continue
        # print(f"\nLooking at link '{link}' in page '{url}'")
        if link.find(':') >= 0:
            # if protocol is present, make sure protocol is http or https
            if not link.split(':', 1)[0].startswith('http'):
                continue
        else:
            # link is relative to current page -- remove 'xxx.html' and add in link
            link = f"{url.rsplit('/', 1)[0]}/{link}"
        # print(f"  ??? link={link}")
        link = link.split('#', 1)[0].rstrip('/')
        if (
            link not in visited_links and           # haven't already visited
            link.startswith(domain) and
            not link.endswith('.pdf')
           ):

            if link not in visited_links:
                print(f"n={len(visited_links)}, link: {link}")

            visited_links.update(scrape_domain(domain, link, visited_links, depth=depth+1,
                                               max_depth=max_depth))
    return visited_links


def extract_mathml(url: str) -> list[str]:
    '''Given a URL, find all the MathML expression in it and return a list of strings for the MathML.
       We need to open a browser because the math needs to get turned into MathML by MathJaX.
    '''
    print(f"EXTRACTING mathml from {url}")
    opts = []
    try:
        opts = webdriver.ChromeOptions()
        opts.add_argument(argument=r'binary=C:\Users\neils\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe')
        opts.add_argument('headless')
        opts.add_experimental_option('excludeSwitches', ['enable-logging'])
    except Exception as e:
        print(f"extract_mathml: exception in setting up options for Chrome in {url}:\n{e}")
        return []
    try:
        browser = webdriver.Chrome(options=opts)
        browser.implicitly_wait(5)
        browser.get(url)
    except Exception as e:
        print(f"extract_mathml: exception for browser getting url {url}:\n{e}")
        return []
    try:
        results = browser.find_elements(By.TAG_NAME, 'math')
        answer = []
        for result in results:
            answer.append(result.get_attribute('outerHTML'))
        return answer
    except Exception as e:
        print(f"extract_mathml: exception while navigating DOM in {url}:\n{e}")
        return []


def get_visited_urls(url_file_or_dir: str, links: set[str], names: dict[str, str]) :
    '''Return the visited URL set (.urls) in the given file or directory.
       This avoids revisting links we've already scraped'''
    if os.path.isfile(url_file_or_dir):
        get_visited_urls_in_file(url_file_or_dir, links, names)
        return
    for entry in url_file_or_dir:
        get_visited_urls_in_file(f"{url_file_or_dir}/{entry}", links, names)


def get_visited_urls_in_file(file: str, links: set[str], names: dict[str, str]):
    if file.endswith('.urls'):
        with open(file, 'r', encoding='utf8') as in_stream:
            lines = in_stream.readlines()
            for url in lines:
                url = url.strip()
                if url == '' or url.startswith('#'):   # empty lines and commented out lines
                    continue
                if url.startswith('('):
                    url_and_name = url.split(',', 1)
                    url = url_and_name[0].strip('(').strip()
                    names[url] = url_and_name[1].rstrip(')').strip().replace(':', "_50_")
                links.add(url)

def write_urls(file: str, urls: set[str]):
    with open(file, 'w', encoding='utf8') as out_stream:
        for url in sorted(urls):
            out_stream.write(url)
            out_stream.write('\n')


def write_mmls_file_for_domain(urls: set[str], mmls_file: str):
    print(f"write_mmls_file_for_domain: mmls_file='{mmls_file}'")
    with open(mmls_file, 'w', encoding='utf8') as out_stream:
        n_mathml = 0
        for url in urls:
            try: 
                site_mathml = extract_mathml(url)
                n_mathml += len(site_mathml)
                for mathml in site_mathml:
                    try:
                        mathml = mathml.replace('&nbsp;', '&#x00A0;')   # probably should handle all entity names
                        # some alttext attr has unescaped "<" which causes errors in "fromstring"
                        try:
                            cleaned_mathml = mmls2csv.clean_mathml(etree.fromstring(mathml))
                        except Exception:
                            # some alttext attr has unescaped "<" which causes errors in "fromstring"
                            # this cheapo fix seems to catch all the instances which involve "...y < ..." and "{ < }"
                            mathml = mathml.replace("y <", "y &lt;").replace("{ < }", "{ &lt; }")
                            cleaned_mathml = mmls2csv.clean_mathml(etree.fromstring(mathml))
                        if cleaned_mathml is not None:
                            out_stream.write(etree.tostring(cleaned_mathml, encoding="unicode"))
                            out_stream.write('\n')
                    except Exception as e:
                        print(f"\nFailure in extracting in url='{url}'\nmathml={mathml}")
                        print(f'Message is {e}')
                        print('Skipping expression\n')
            except Exception as e:
                    print(f"\nFailure in extracting math in url='{url}'")
                    print(f'Message is {e}')
                    print('Skipping url\n')
    print(f"#of exprs={n_mathml}")



ILLEGAL_FILE_NAME_CHARS_PATTERN = (
    re.compile('|'.join(sorted(re.escape(ch) for ch in r'@$%&\/:*?"\'<>|~`#^+={}[];!')))
    )


def get_name_from_url(url: str) -> str:
    '''Get a useful name for the URL -- title isn't good for some books, so I'm going with the text in an <h1>'''
    try:
        name = url.split(':', 1)[1][2:]
    except Exception as e:
        print(f'get_name_from_url: url={url}')
        print(f'Exception: {e}')
        return "Error"
    if name.endswith('.html'):
        name = name[0:-5]
    elif name.endswith('.php'):
        name = name[0:-4]
    elif name.endswith('/'):
        name = name[0:-1]
    try:
        response = do_redirect(url)
        if response.status_code == 200:
            html_text = response.text
            # find all the anchor tags with "href"
            for _, h1 in etree.iterparse(BytesIO(html_text.encode('utf-8')),
                                         tag='h1',
                                         remove_blank_text=True,
                                         remove_comments=True,
                                         encoding='UTF-8',
                                         html=True):
                real_name = ''.join(h1.itertext()).strip()
                if len(real_name) > 2:
                    name = real_name
                    break
            if name in url:
                print("   DID NOT FIND <h1>")
    except Exception as e:
        print(f"URL request caused an error:\n{e}")
    finally:
        return ILLEGAL_FILE_NAME_CHARS_PATTERN.sub(
            lambda m: f"__{ord(m.group(0))}__", name)


# all_web_pages = scrape_domain("https://yoshiwarabooks.org",
#                               "https://yoshiwarabooks.org",
#                               set())
# all_web_pages = scrape_domain("http://linear.pugetsound.edu/fcla",
#                             "http://linear.pugetsound.edu/fcla/front-matter.html",
#                             set())

# the following was used to generate the pretext_catalog (which was then filtered by hand)
def initial_setup():
    already_visited_links = set()
    get_visited_urls(r"C:\Dev\SimpleSpeakData\ebooks", already_visited_links, {})
    all_web_pages = scrape_domain("https://math.dartmouth.edu/~trs/PreTeXtProjects/abstract-algebra-refresher-CLI",
                                  "https://math.dartmouth.edu/~trs/PreTeXtProjects/abstract-algebra-refresher-CLI/output/web/aar.html",
                                  already_visited_links.copy(),
                                  max_depth=3)
    write_urls(r"C:\Dev\SimpleSpeakData\ebooks\nordstrom-game-theory.urls", all_web_pages - already_visited_links)

# print(f"\n\n#all_web_pages={len(all_web_pages)}\n")
# write_mmls_file_for_domain(all_web_pages, r'C:\Dev\SimpleSpeakData\ebooks\fcla.mmls')


def generate_urls_from_catalog(file_name: str, out_dir: str):
    urls = []
    names = {}

    if file_name.endswith('.urls'):
        # e.g, r"C:\Dev\SimpleSpeakData\pretext-catalog.urls"
        urls_as_set = set()
        get_visited_urls(file_name, urls_as_set, names)
        urls = sorted(urls_as_set)
    else:
        # generating a urls for a single file
        urls = [file_name]
    already_visited_links = set()
    get_visited_urls(out_dir, already_visited_links, {})
    print(f'# already visited links: {len(already_visited_links)}')
    for url in list(urls):
        file_name = out_dir + "\\" + (names[url] if url in names else get_name_from_url(url)) + ".urls"
        print(f"\n\nurl '{url}' maps to \n    '{file_name}'")
        if not os.path.exists(file_name):
            print(f"...scraping domain...")
            web_pages = scrape_domain(url.rsplit('/', 1)[0],
                                      url, already_visited_links.copy(), max_depth=2)
            write_urls(file_name, web_pages - already_visited_links)


def generate_mathml_files_from_urls(url_file_or_dir: str):
    if os.path.isfile(url_file_or_dir):
        generate_mathml_files_from_url_file(url_file_or_dir)
    else:
        for entry in os.listdir(url_file_or_dir):
            generate_mathml_files_from_url_file(f"{url_file_or_dir}\\{entry}")


def generate_mathml_files_from_url_file(url_file: str):
    if url_file.endswith('.urls'):
        mmls_file = url_file.replace('urls', 'mmls')
        if not os.path.isfile(mmls_file):
            entry_urls = set()
            get_visited_urls(url_file, entry_urls, {})
            write_mmls_file_for_domain(entry_urls, mmls_file)


def fixfiles():
    '''forgot to clean the ebook files -- this repairs that'''
    dir = r"C:\Dev\SimpleSpeakData\open-stax"
    for entry in os.listdir(dir):
        if entry.endswith(".raw-mmls"):
            with open(dir+'/' + entry, 'r', encoding='utf8') as in_stream:
                with open(dir+'/'+entry.replace(".raw-mmls", ".mmls"), 'w', encoding='utf8') as out_stream:
                    lines = in_stream.readlines()
                    for mathml in lines:
                        html = etree.HTML(mathml) # returns <html><body><math>...</math></body></html>
                        cleaned_mathml = mmls2csv.clean_mathml(html[0][0])
                        if cleaned_mathml is not None:
                            cleaned_mathml = etree.tostring(cleaned_mathml, encoding="unicode")
                            out_stream.write(cleaned_mathml)
                            out_stream.write('\n')


# fixfiles()
# initial_setup()
# use the line below (ends in '.urls') to give a large number of links
# reads catalog data
# generate_urls_from_catalog(r"C:\Dev\SimpleSpeakData\yoshiwarabooks-catalog.urls", r"C:\Dev\SimpleSpeakData\highschool-yoshi")
# use on a single file or an entire directory (if .mmls file exists, it will NOT regenerate it)
generate_mathml_files_from_urls(r"C:\Dev\SimpleSpeakData\highschool-yoshi\Algebra Toolkit.urls")
