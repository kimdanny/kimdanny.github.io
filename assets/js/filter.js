/* Topic + year filtering for /publications/.
   Progressive enhancement: with JS off, every entry stays visible. */
(function () {
  var filters = document.getElementById('filters');
  if (!filters) return;

  var chips   = Array.prototype.slice.call(filters.querySelectorAll('.chip'));
  var yearSel = document.getElementById('year-filter');
  var count   = document.getElementById('filter-count');
  var pubs    = Array.prototype.slice.call(document.querySelectorAll('.pub'));
  var groups  = Array.prototype.slice.call(document.querySelectorAll('.year-group'));

  var topic = 'all';
  var year  = 'all';

  function apply() {
    var shown = 0;

    pubs.forEach(function (el) {
      var topics  = (el.getAttribute('data-topics') || '').split(' ');
      var okTopic = topic === 'all' || topics.indexOf(topic) !== -1;
      var okYear  = year === 'all' || el.getAttribute('data-year') === year;
      var visible = okTopic && okYear;
      el.hidden = !visible;
      if (visible) shown++;
    });

    // Hide a year heading once nothing under it survives the filter.
    groups.forEach(function (g) {
      var any = Array.prototype.slice.call(g.querySelectorAll('.pub'))
        .some(function (el) { return !el.hidden; });
      g.hidden = !any;
    });

    chips.forEach(function (c) {
      c.setAttribute('aria-pressed', c.getAttribute('data-topic') === topic ? 'true' : 'false');
    });

    if (topic === 'all' && year === 'all') {
      count.hidden = true;
    } else {
      count.hidden = false;
      count.textContent = shown === 0
        ? 'No publications match this filter.'
        : 'Showing ' + shown + ' of ' + pubs.length + ' publications.';
    }
  }

  function syncHash() {
    var parts = [];
    if (topic !== 'all') parts.push('topic=' + topic);
    if (year !== 'all') parts.push('year=' + year);
    var hash = parts.length ? '#' + parts.join('&') : ' ';
    history.replaceState(null, '', location.pathname + (parts.length ? hash : ''));
  }

  function readHash() {
    var h = location.hash.replace(/^#/, '');
    if (!h) return;
    h.split('&').forEach(function (pair) {
      var kv = pair.split('=');
      if (kv[0] === 'topic' && kv[1]) topic = kv[1];
      if (kv[0] === 'year' && kv[1]) year = kv[1];
    });
    if (yearSel) yearSel.value = year;
  }

  chips.forEach(function (c) {
    c.addEventListener('click', function () {
      var next = c.getAttribute('data-topic');
      // Clicking the active chip clears it.
      topic = (next === topic && next !== 'all') ? 'all' : next;
      apply();
      syncHash();
    });
  });

  if (yearSel) {
    yearSel.addEventListener('change', function () {
      year = yearSel.value;
      apply();
      syncHash();
    });
  }

  readHash();
  apply();
})();
