def pytest_markdown_docs_markdown_it():
    """Support admonitions."""
    import markdown_it
    from mdit_py_plugins.admon import admon_plugin

    mi = markdown_it.MarkdownIt(config="commonmark")
    mi.use(admon_plugin)
    return mi
