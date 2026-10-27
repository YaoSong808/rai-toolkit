# Policy packs

A policy pack is a directory of YAML files loaded by `--policies-dir`. Every
YAML file in that directory must be a complete `PolicySet` mapping with
`name` and `policies` fields. A bare list of policies is not a valid file.

## Layout

The bundled example keeps both policies in one policy set:

```
my-policy-pack/
  pack.yaml         # name, version, description, and policies
  README.md         # optional owner/contact and other pack-level metadata
```

The supported `PolicySet` schema has no pack-level `owner` or `frameworks`
fields. Keep that descriptive metadata in an adjacent README instead of adding
unsupported YAML keys. In the bundled example, the illustrative owner is
`governance-team@example.com`; its broader context is
`Internal-Gov-Standard`, `EU-AI-Act-Art-15`, and `NIST-AI-RMF-MANAGE`. Those
pack-level notes are documentation only. Framework references inside each
policy remain part of that policy and are loaded normally.

## Lint a pack

Run the same offline validation in a local check or CI job:

```bash
rai policies lint my-policy-pack/
```

## Run an assessment

```bash
rai assess my_pkg.MyModel \
  --preset financial_services \
  --demo-datasets \
  --policies-dir my-policy-pack/
```

`my_pkg.MyModel` is a placeholder dotted Python import path; replace it with
your own model adapter class. `--demo-datasets` selects the bundled sample
data. For a real assessment, replace it with an explicit `--dataset <slug>`.

When `--policies-dir` is supplied, the policy engine loads the YAML policy sets
from that directory. It does not automatically merge them with the toolkit's
starter YAML policies. Include every policy set you want the policy engine to
evaluate in the specified directory.

## Working example

See `rai_toolkit/policies/packs/example_enterprise_pack/`. It contains a valid
`pack.yaml` with the enabled `high-risk-disclaimer-required` and
`no-external-data-references` policies.
