# Third-Party Notices

KDMSi includes third-party source code under the licenses described below.
These notices cover the third-party components only; they do not grant a
license for the remainder of the KDMSi repository.

## RNA-FM

- Vendored component: `liubin_pipeline/fm/`
- Upstream project: [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM)
- Provenance reference: upstream commit
  [`348951516e0963d22bbb33b3c9fc18c89081d38e`](https://github.com/ml4bio/RNA-FM/commit/348951516e0963d22bbb33b3c9fc18c89081d38e)
- License: MIT; see [`licenses/RNA-FM-LICENSE.txt`](licenses/RNA-FM-LICENSE.txt)
- Copyright: Copyright (c) 2022 Jiayang Chen

At the provenance revision above, the Python files vendored in
`liubin_pipeline/fm/` match the upstream RNA-FM files except for local changes
to `fm/__init__.py` and `fm/pretrained.py`. The local changes avoid importing
unused downstream modules, support an explicit/local RNA-FM weight path, and
improve weight-loading behavior and download error messages.

RNA-FM incorporates code derived from Meta's ESM implementation. The original
Facebook/Meta copyright headers have been retained in the affected files.
ESM is also distributed under the MIT License; see
[`licenses/ESM-LICENSE.txt`](licenses/ESM-LICENSE.txt) and the
[official ESM repository](https://github.com/facebookresearch/esm).

The RNA-FM pretrained weight is distributed separately and is not contained in
this repository. Consult the RNA-FM project and the KDMSi asset documentation
for the applicable source, citation, and distribution information.
