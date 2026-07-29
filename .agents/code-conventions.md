# Code conventions

- Do not add spacing solely to vertically align assignments, arguments, comments, or similar syntax.
- Put comments on the line above the code they describe. Do not use trailing comments.
- Keep wrapped calls non-hanging: put the opening parenthesis on its own line and indent continuation lines once. Do not align later arguments under an opening parenthesis on the preceding line.

  ```python
  function(x, y, z)

  func(
      arg1, arg2,
      arg3, ...,
  )

  func(
      arg1,
      arg2,
      arg3,
  )
  ```

- Within this layout, choose one or multiple arguments per line to balance clarity and compactness. Do not default to one argument per line.

- Avoid type hints by default. Add them only when they materially clarify a non-obvious interface or prevent a real mistake.
