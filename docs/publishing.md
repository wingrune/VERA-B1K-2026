# Publish the GitHub Pages blog

The project website is the static `docs/` directory. It needs no package install or site build.

After pushing this repository, a repository administrator, maintainer, or user with permission to manage Pages should select:

1. Repository **Settings → Pages**.
2. **Deploy from a branch**.
3. Branch **main**, folder **/docs**, then **Save**.

The expected project URL is https://wingrune.github.io/VERA-B1K-2026/ . GitHub may take a few minutes to publish the first build. Check the Pages deployment before sharing the URL. The file `.nojekyll` keeps the HTML and data files unchanged.

The `.github/workflows/release-checks.yml` workflow verifies the imported source, CPU tests and static links on pushes and pull requests. Pages itself is deployed by GitHub's branch publishing service. Choosing branch publishing avoids requiring an Actions workflow to create or enable a Pages site.

For a local preview:

```bash
python3 -m http.server 8080 --directory docs
```

Open http://localhost:8080/ . All asset references are relative, so the same files work at the GitHub project subpath.

The results explorer loads `data/public-results.json`; its source dataset and revision are displayed beside the results. Regenerate that file using `tools/summarize_public_results.py` only when intentionally selecting a new evidence snapshot. Keep the report and README's numbers consistent with it.
