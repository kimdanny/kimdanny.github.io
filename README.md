# kimdanny.github.io

Source for Danny To Eun Kim's academic website. Built with [Jekyll](https://jekyllrb.com/);
content lives in YAML data files so each publication is written once and rendered everywhere.

Layout adapted from [Jon Barron's website](https://jonbarron.info/) — please keep that
attribution if you reuse this.

---

## Running it locally

This repo pins **Ruby 3.0.0** via `.ruby-version`. The `jekyll` on your `PATH` at
`/usr/local/bin/jekyll` runs on macOS system Ruby 2.6 and **crashes** — always go through
`bundle exec`, never call `jekyll` directly.

### One-time setup

```bash
cd <current-dir>

# Make sure rbenv's Ruby is the one in use (should print 3.0.0, not 2.6.x)
ruby -v

# eventmachine needs to be pointed at Homebrew's OpenSSL or it fails to link on macOS
bundle config set --local build.eventmachine "--with-openssl-dir=$(brew --prefix openssl@3)"

bundle install
```

If `ruby -v` prints 2.6.x, rbenv isn't initialised in your shell. Either add
`eval "$(rbenv init - zsh)"` to `~/.zshrc`, or prefix commands with
`export PATH="$HOME/.rbenv/shims:$PATH"`.

### Preview

```bash
bash launch.sh      # build + serve at http://127.0.0.1:4000
bash takedown.sh    # stop it
```

`launch.sh` runs the server in the background with `--livereload`, so saving any file
rebuilds the site and refreshes the open browser tab — no need to re-run anything. It waits
for the first build and prints the build log if the server fails to come up; the full log is
at `/tmp/jekyll-preview.log`.

To stop the server: `bash takedown.sh` (or <kbd>Ctrl-C</kbd> if you ran `jekyll serve` by hand).

### Build without serving

```bash
bundle exec jekyll build     # output lands in _site/
```

---

## Updating content

Almost everything is data, not markup. You rarely need to touch HTML.

### Add a publication

Edit **`_data/publications.yml`** and add an entry at the top (the file is ordered
newest-first; `year` drives the grouping on the publications page):

```yaml
- title: "Your Paper Title"
  authors: ["To Eun Kim", "Co Author"]     # your name is bolded automatically
  equal_note: true                         # optional: prints "* denotes equal contribution."
  year: 2027
  venue: "SIGIR 2027"
  venue_note: "Oral"                       # optional: appended after an em dash
  also: "SIGIR 2026 Workshop — Spotlight"  # optional: second venue line
  award: "🏆 Best Paper Award"             # optional
  topics: [retrievers, deployment]         # see _config.yml for valid keys
  selected: true                           # optional: also show on the homepage
  image: /images/your-teaser.png           # optional: 180px-wide thumbnail
  tldr: >
    One paragraph describing the work.
  links:
    paper: "https://..."
    code: "https://..."
```

Notes:
- Mark equal contribution by putting `*` in the author string itself: `"To Eun Kim*"`.
- `links` keys become the visible link labels, in the order you write them. Multi-word keys
  work — `workshop page:`, `LLM elicitation code:`.
- Put teaser images in `images/`. Roughly square, around 400×400, works best.
- `topics` keys must match `topics:` in `_config.yml`; add a new topic there and a new
  filter chip appears automatically.

### Add a news item

Edit **`_data/news.yml`**, newest first. `date` only needs to be right to the month; `text`
is Markdown, so inline links work:

```yaml
- date: 2027-01-01
  text: >
    Something happened — [with a link](https://...).
```

The homepage shows the newest four and folds the rest into "Older news".
