# Terms

- Terms are the components of GAM models (e.g. linear, smooths, intercepts etc).
- If you are familiar with ``mgcv``, then the mgcv representation of the term can be inspected for any term using ``str(term)``.
- We support adding of terms as syntactic sugar for creating a list of terms, i.e.

```python
from pymgcv.terms import S, L
assert L("x0") + S("x1") == [L("x0"), S("x1")]
```

::: pymgcv.terms.L
    options:
      members:
        - "__init__"
::: pymgcv.terms.S
    options:
      members:
        - "__init__"
::: pymgcv.terms.T
    options:
      members:
        - "__init__"
::: pymgcv.terms.Interaction
    options:
      members:
        - "__init__"
::: pymgcv.terms.Offset
    options:
      members:
        - "__init__"
::: pymgcv.terms.Intercept
    options:
      members:
        - "__init__"
::: pymgcv.terms.AbstractTerm
    options:
        members:
          - label
          - mgcv_identifier
          - __str__
