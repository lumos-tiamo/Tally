"""Source of the in-sandbox tool modules for the diligence scenario.

These are real, importable, testable modules. They are copied verbatim into the
generated ``tools`` package (see ``ToolRegistry.attach_source_file``) rather than
being generated from ``local_source`` snippets, because they contain enough logic
to deserve tests of their own.

They must not import anything outside the sandbox's frozen dependency set:
stdlib, pandas, numpy, pyarrow, lxml, beautifulsoup4.
"""
