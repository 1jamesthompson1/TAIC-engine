# SavedDataFrames

Saved Dataframes all inherit from `SavedDataFrame` and provide a convenient way to persist and load Pandas DataFrames to/from disk. Each subclass defines its own file path and can be used to save or load the corresponding DataFrame. The subclasses also include a Pydantic model that shows what each DataFrame row should look like.

::: engine.SavedDataFrames
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      heading_level: 2
      members_order: source
      show_if_no_docstring: true
