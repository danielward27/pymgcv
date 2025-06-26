

### Serving docs locally
```bash
mkdocs serve
```

### Adding code examples e.g. in docstrings
In order for the doctest to run properly, and documentation to be generated properly:
- Use the admonition `!!! example`, containing a python fenced code block.
- Do not use a prompt `>>> `
- We generally do not test the output, just that each example runs (although you could use an assert)
- Use a comment at the bottom if you want to show an output example.
