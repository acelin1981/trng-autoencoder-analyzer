Technical Article:  
AI-Assisted TRNG Entropy Analysis Using SP800-90B and Autoencoder Residual Mapping  
https://medium.com/@ace.lin0121/ai-assisted-trng-entropy-analysis-using-sp800-90b-and-autoencoder-residual-mapping-cdf2ca6e3cb1

TRNG Autoencoder Analysis – Documentation
-----------------------------------------

This repository provides a public technical presentation (PPTX file)
describing a conceptual framework for evaluating True Random Number
Generator (TRNG) entropy quality using NIST SP800-90B principles and
an Autoencoder-based residual inspection approach.

Repository Scope
----------------
In addition to the technical presentation, this repository includes a reference
implementation demonstrating how autoencoder-based residual analysis can be applied
to TRNG output data. The implementation is intended to complement NIST SP800-90B
statistical tests by providing a structural and visualization-oriented inspection
method.

## Motivation
Traditional entropy validation methods defined in NIST SP800-90B focus on statistical
properties of raw output. While effective, they may not directly reveal spatial or
structural patterns such as stuck bits, bias, or repetition.

This work explores the use of an autoencoder trained on healthy TRNG output to learn
normal bit-level behavior. Reconstruction residuals are then used as an auxiliary
signal to highlight potential structural anomalies that are difficult to detect
using statistics alone.

Purpose
-------

The purpose of this repository is to make the methodology explanation
and related research notes publicly accessible.  
This allows external verification, timestamped publication, and a
permanent public record of the work.

Repository Contents
-------------------

AE_SP80090B_Safe_v2.pptx  
A technical slide deck covering:
- Motivation for applying machine learning techniques to TRNG analysis
- Summary of SP800-90B entropy-related concepts  
  (collision tests, symbol frequency, estimator considerations)
- A conceptual Autoencoder residual analysis idea for identifying
  potential patterns or anomalies in TRNG output data
- Discussion of how statistical and AI-based inspection can complement
  each other in TRNG quality evaluation

README.md  
This file.

Current Status
--------------

This repository currently provides documentation only.
No scripts, datasets, or executable code are included at this time.

Planned Additions
-----------------

Additional materials may be published in the future, such as:
- Extended documentation or notes
- Supporting figures or examples

Versioning
----------

v1.0.0 – Initial public release (documentation only)

Contact
-------

For questions or clarifications, please use GitHub Issues.
