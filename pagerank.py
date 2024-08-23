import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability = dict()
    for page_name in corpus:
        probability[page_name] = 0

    if len(corpus[page]) == 0:
        for page_name in probability:
            probability[page_name] += 1 / len(corpus)
        return probability

    prob_link = damping_factor / len(corpus[page])
    for page_name in corpus[page]:
        probability[page_name] += prob_link

    prob_random = (1 - damping_factor) / len(probability)
    for page_name in probability:
        probability[page_name] += prob_random

    return probability

    #raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = dict()
    for page_name in corpus:
        pagerank[page_name] = 0

    curr_page = random.choice(list(pagerank))
    pagerank[curr_page] += 1

    for i in range(n-1):
        weight = transition_model(corpus, curr_page, damping_factor)
        
        rand_choice = random.random()
        prob = 0

        for page_name, probability in weight.items():
            prob += probability
            if prob >= rand_choice:
                curr_page = page_name
                break
        
        pagerank[curr_page] += 1

    for page_name, count in pagerank.items():
        pagerank[page_name] = count / n

    return pagerank

    #raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = dict()
    new_pagerank = dict()
    prob_random = (1 - damping_factor) / len(corpus)
    prob_all = 1 / len(corpus)
    for page_name in corpus:
        pagerank[page_name] = prob_random
        new_pagerank[page_name] = None

    max_change = 1

    while max_change > 0.001:
        max_change = 0

        for page in corpus:
            prob_choice = 0
            for other_page in corpus:
                if len(corpus[other_page]) == 0:
                    prob_choice += pagerank[other_page] * prob_all
                elif page in corpus[other_page]:
                    prob_choice += pagerank[other_page] / len(corpus[other_page])
            
            new_pagerank[page] = prob_random + (damping_factor * prob_choice)

        total_prob = sum(new_pagerank.values())
        for page, rank in new_pagerank.items():
            new_pagerank[page] = rank / total_prob
        
        for page in corpus:
            rank_change = abs(pagerank[page] - new_pagerank[page])
            max_change = max(max_change, rank_change)

        pagerank = new_pagerank.copy()

    return pagerank
    #raise NotImplementedError


if __name__ == "__main__":
    main()
