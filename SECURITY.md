# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting or a private security advisory
for this repository. Do not include credentials or exploit details in a public issue.

## Credential exposure response

A committed credential must be treated as compromised, even after the file is deleted.
Respond by:

1. Revoking or rotating the credential at its issuing service.
2. Removing the credential from the current repository tree.
3. Purging the credential from Git history in a coordinated history rewrite.
4. Reviewing authentication and access logs for unexpected use.
5. Running make security-check before committing and confirming the GitHub secret-scan workflow passes.

The repository scanner reports file paths and credential types only; it does not print
the detected secret value.
