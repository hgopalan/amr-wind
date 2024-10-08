.. _dev-documenting:

Documentation - user manual and source code docs
================================================

AMR-Wind comes with two different types of documentation:

- The manual, i.e., the document you are reading now, that is
  written using `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_, and

- Inline documentation within C++ source code that are written in a format that can be
  processed automatically by `Doxygen <http://www.doxygen.nl/manual/index.html>`_

User documentation
------------------

AMR-Wind user documentation is written using a special format called
ReStructured Text (ReST) and is converted into HTML and PDF formats using a
python package Sphinx. Since the manuals are written in simple text files, they
can be version controlled alongside the source code. Documentation is
automatically generated with new updates to the GitHub repository and deployed
at `AMR-Wind documentation site <https://exawind.github.io/amr-wind>`_.

Writing user documentation
``````````````````````````

As mentioned previously, documentation is written using a special text format
called reStructuredText. Sphinx user manual provides a `reST Primer
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ that
provides an overview of this format and how to write documentation using this format.


Documenting source code
-------------------------

Source code (C++ files) are commented using a special format that
allows Doxygen to extract the annotated comments and create source
code documentation as well as inheritance diagrams. The
:doc:`../doxygen/html/index` documentation for the latest snapshot of
the codebase can be browsed in this manual. The `Doxygen manual
<http://www.doxygen.nl/manual/index.html>`_ provides an overview of
the syntax that must be used. Please follow the Doxygen style of
commenting code when commenting AMR-Wind sources.

When commenting code, try to use self-documenting code, i.e., descriptive names
for variables and functions that eliminate the need to describe what is going on
in comments. In general, comments should address *why* something is being coded
in a particular way, rather than how the code does things. Try to write the code
in a clear manner so that it is obvious from reading the code instead of having
to rely on comments to follow the code structure.

Building documentation
----------------------

To generate this documentation on a local machine, or to rebuild docs
during code development, `doxygen`, `graphviz`, `doxysphinx`, and
`sphinx` are required, as well as turning on the
`AMR_WIND_ENABLE_DOCUMENTATION` cmake option.
```{shell}
$ cd build && cmake -DAMR_WIND_ENABLE_DOCUMENTATION:BOOL=ON .. && cmake --build . -t docs
```
The resulting documentation is in `docs/spinx/html` directory.Documentation can
also be generated in other formats, consult `Sphinx docs
<https://www.sphinx-doc.org/en/master/usage/builders/index.html>`_ for available
formats and their usage.
