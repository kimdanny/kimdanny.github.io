---
layout: page
title: "Publications"
permalink: /publications/
---

{%- assign groups = site.data.publications | group_by: "year" -%}

<p>
  {{ site.data.publications.size }} papers.
  Full list also on <a href="{{ site.scholar }}" rel="noopener">Google Scholar</a>.
</p>

<div class="filters" id="filters">
  <button type="button" class="chip" data-topic="all" aria-pressed="true">All topics</button>
  {%- for t in site.topics %}
  <button type="button" class="chip" data-topic="{{ t.key }}" aria-pressed="false">{{ t.label }}</button>
  {%- endfor %}
  <label class="visually-hidden" for="year-filter">Year</label>
  <select id="year-filter">
    <option value="all">All years</option>
    {%- for g in groups %}
    <option value="{{ g.name }}">{{ g.name }}</option>
    {%- endfor %}
  </select>
</div>

<p class="filter-count" id="filter-count" hidden></p>

{% for g in groups %}
<section class="year-group" data-year="{{ g.name }}">
  <h2 class="year-head">{{ g.name }}</h2>
  {% for p in g.items %}{% include pub.html pub=p tldr=true %}{% endfor %}
</section>
{% endfor %}

<script src="{{ '/assets/js/filter.js' | relative_url }}" defer></script>
