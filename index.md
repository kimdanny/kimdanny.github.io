---
layout: default
title: "Home"
---

<div class="profile">
  <img class="profile__photo" src="{{ '/images/profilepic.jpeg' | relative_url }}"
       alt="Portrait of To Eun Kim">
  <div class="profile__body">
    <h1 class="profile__name">{{ site.title }}</h1>
    <p class="profile__meta">
      {{ site.korean_name }}<br>
      {{ site.tagline }}<br>
      toeunkim<span aria-hidden="true">{at}</span>cmu<span aria-hidden="true">{dot}</span>edu
    </p>
    <p class="profile__links">
      <a href="{{ site.cv | relative_url }}">CV</a>
      <a href="{{ site.scholar }}" rel="noopener">Google Scholar</a>
      <a class="icon-link" href="{{ site.github }}" rel="noopener" title="GitHub" aria-label="GitHub">{% include icon.html name="github" %}</a>
      <a class="icon-link" href="{{ site.linkedin }}" rel="noopener" title="LinkedIn" aria-label="LinkedIn">{% include icon.html name="linkedin" %}</a>
      <a class="icon-link" href="{{ site.twitter }}" rel="noopener" title="X" aria-label="X">{% include icon.html name="x" %}</a>
    </p>
  </div>
</div>

I'm a PhD student at CMU's [Language Technologies Institute (LTI)](https://lti.cs.cmu.edu),
advised by Prof. [Fernando Diaz](https://841.io). 
I’m grateful to be supported by the CMU LTI ASA Presidential Fellowship during 2026–2027.

My research focuses on emerging AI ecosystems formed by distributed AI systems. 
I study these systems through an economic and information retrieval lens, asking how we can build platforms and infrastructure that help agents search, share knowledge, and co-evolve with a broader community of agents. 
I’m also interested in how we evaluate these ecosystems as a whole, beyond isolated task performance, using outcomes that ultimately matter after deployment, such as sustained utility and market health.
Alongside this work, I study agentic search (e.g., deep research) and complex query retrieval (e.g., tip-of-the-tongue queries).

<details class="bio-more" markdown="1">
<summary>Read full bio</summary>

Before CMU I was a graduate researcher in the [Web Intelligence Group](https://wi.cs.ucl.ac.uk)
at [University College London](https://www.ucl.ac.uk), where I completed my M.Eng. in Computer
Science in 2022 under Prof. [Emine Yilmaz](https://sites.google.com/site/emineyilmaz/) and Prof.
[Aldo Lipani](https://aldolipani.com). My work there centred on conversational AI and user
simulation, and I was a lead developer of
[Condita](https://www.amazon.science/alexa-prize/proceedings/condita-a-state-machine-like-architecture-for-multi-modal-task-bots)
in the first Alexa Prize TaskBot Challenge. In summer 2022 I interned at [Raft](https://www.raft.ai)
as a machine learning engineer, automating freight-forwarding paperwork with OCR and NLP. During my
undergraduate degree I worked with Prof. [Marianna Obrist](https://uclic.ucl.ac.uk/people/marianna-obrist)
at the [UCL Interaction Centre](https://uclic.ucl.ac.uk) on clustering text stories by their authors'
smell experiences.

I take pride in being one of the early members of the
[UCL Artificial Intelligence Society](https://uclaisociety.co.uk), where I founded the first
[Machine Learning tutorial series](https://github.com/UCLAIS/Machine-Learning-Tutorials) — now an
annual tradition.

</details>

## Research

**Distributed Agents & Agent Economies**. 
Agents no longer act alone. They produce information and tools, consume what
other agents produce, and meet on platforms that mediate between them. I study how to serve a
whole population of agents, and what we learn by viewing their interactions through an
economic and information retrieval lens.

<p class="paper-links" markdown="span">
[Multi-Agent Transactive Memory](https://arxiv.org/abs/2606.19911)
[Marketplace-Eval](https://dl.acm.org/doi/10.1145/3805712.3808542)
[AgentSearch Workshop](https://agent-search.github.io/agentsearch-sigir26/)
[Agent and Tool Search Survey](https://www.preprints.org/manuscript/202609.0402)
</p>

**Distributed Retrievers**. 
A standard RAG system assumes a single retriever. However, an agent may have many options to
choose from, each strong on different queries. I work on selecting and combining retrievers so
that an agent gets the right evidence for the question.

<p class="paper-links" markdown="span">
[Mixture of Retrievers](https://aclanthology.org/2025.emnlp-main.601/)
[Learning To Rank Retrievers](https://dl.acm.org/doi/10.1145/3805712.3809954)
[REML](https://arxiv.org/abs/2407.12982)
</p>

**Monetization in conversational AI systems**.
Deployed systems face business realities that benchmarks leave out. How do we
give data providers fair exposure for the content they supply? How can advertising be woven
into a conversation without interrupting the flow? 
What does it take to actually build and operate one for thousands of real users?

<p class="paper-links" markdown="span">
[Fair RAG](https://doi.org/10.1145/3731120.3744599)
[Ad Integration and Detection](https://ceur-ws.org/Vol-4038/paper_385.pdf)
[Alexa Prize TaskBot '21](https://www.amazon.science/alexa-prize/proceedings/condita-a-state-machine-like-architecture-for-multi-modal-task-bots)
</p>

**Complex/ToT Query Retrieval**.
A surprising share of web search queries are actually tip-of-the-tongue (ToT)
queries. In ToT known-item retrieval, the searcher cannot recall the name of what they are after
and can only describe it — at length, vaguely, and often inaccurately. I build the queries,
benchmarks, and shared evaluation tracks for retrieval when there is nothing precise to lean on.
([Exa](https://exa.ai) has adopted ToT as one of its benchmarks.)

<p class="paper-links" markdown="span">
[LLM and Human Elicited ToT](https://dl.acm.org/doi/10.1145/3726302.3730335)
[Multilingual ToT](https://dl.acm.org/doi/10.1145/3805712.3808626)
[NTCIR-19](https://ntcir-tot.github.io)
[TREC'25](https://trec.nist.gov/pubs/trec34/papers/Overview_tot.pdf)
[TREC'24](https://trec.nist.gov/pubs/trec33/papers/Overview_tot.pdf)
</p>



{% assign recent = site.data.news | slice: 0, 4 %}
{% assign rest_size = site.data.news.size | minus: 4 %}

## News

<ul class="news">
{% for item in recent %}{% include news.html item=item %}{% endfor %}
</ul>

{% if rest_size > 0 %}
{% assign rest = site.data.news | slice: 4, rest_size %}
<details class="news-more">
<summary>Older news</summary>
<ul class="news">
{% for item in rest %}{% include news.html item=item %}{% endfor %}
</ul>
</details>
{% endif %}

## Selected publications <span class="heading-note">[full list on Google Scholar]({{ site.scholar }})</span>
{: #selected-publications}

{% for p in site.data.publications %}{% if p.selected %}{% include pub.html pub=p tldr=true %}{% endif %}{% endfor %}

<p><a href="{{ '/publications/' | relative_url }}">All publications →</a></p>

## Theses

Master's (UCL): [Multi-Task Neural User Simulator for Task Oriented Dialogue System]({{ '/assets/ucl-master-thesis.pdf' | relative_url }})

Bachelor's (UCL): [Exploring the Potential of Automating the Process of Clustering Smell Stories](https://drive.google.com/file/d/1lhXtb0c5mApa0n78qHJL6rdkxGpHI1e9/view)
