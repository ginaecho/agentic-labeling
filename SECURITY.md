# Security Policy

> **Note:** This is a personal research project and is **not** an official
> Microsoft product. Vulnerability reports are **not** handled by the Microsoft
> Security Response Center (MSRC). Please use the channels below instead. If this
> project is ever adopted into an official Microsoft organization, this policy
> should be replaced with the standard Microsoft `SECURITY.md` and MSRC process.

## Supported Versions

Only the latest version on the `main` branch is actively maintained and
receives security fixes.

| Version | Supported |
| ------- | --------- |
| `main`  | ✅        |
| older   | ❌        |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them privately using GitHub's built-in
[Private Vulnerability Reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability):

1. Go to the **Security** tab of this repository.
2. Click **Report a vulnerability**.
3. Provide a clear description and reproduction steps.

Please include as much of the following as possible:

- Type of issue (e.g., injection, credential exposure, supply-chain).
- Affected file(s) and code path.
- Steps to reproduce.
- Potential impact.

We aim to acknowledge reports within a reasonable time frame and will keep you
updated on remediation progress.

## Scope & Hardening Notes

- **API keys:** This project requires an Anthropic API key supplied via the
  `LLM_API_KEY` environment variable or a local `.env` file. Never commit keys.
  The `.env` file is git-ignored.
- **LLM-generated content:** Cluster names, descriptions, and explanations are
  produced by an LLM and should be reviewed before being used in any
  downstream or production decision.
- **Input data:** Datasets you provide are processed locally and may be sent to
  the configured LLM provider for interpretation. Do not feed confidential or
  regulated data without verifying your provider's data-handling terms.
