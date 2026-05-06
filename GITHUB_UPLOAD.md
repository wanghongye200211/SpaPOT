# GitHub Upload

From this directory:

```bash
git init
git add .
git commit -m "Initial SpaPOT release"
git branch -M main
git remote add origin git@github.com:<USER>/<REPO>.git
git push -u origin main
```

Before pushing, check:

```bash
git status --short
git ls-files
```

Large data and generated runs should not appear in `git ls-files`.

Expected lightweight tracked content:

```text
README.md
METHOD.md
GITHUB_UPLOAD.md
pyproject.toml
requirements.txt
src/
scripts/
examples/
results/r14_classifier_nomass_hybrid/
```
