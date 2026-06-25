# Security Policy

## Supported Versions

`EliminationSearchCV` is currently at an early pre-alpha stage. Only the latest released version receives security attention.

| Version | Supported |
|---|---|
| latest (`0.0.x`) | ✅ Yes |
| older | ❌ No |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

If you believe you have found a security vulnerability in this project, please disclose it responsibly by using [GitHub's private vulnerability reporting](https://github.com/thisal-d/elimination-search-cv/security/advisories/new).

You can also reach the maintainer directly via GitHub: [@thisal-d](https://github.com/thisal-d).

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any relevant version or environment information

You will receive a response as soon as possible. We ask that you give us reasonable time to address the issue before any public disclosure.

## Scope

This is a pure Python library with no network access, no authentication, and no persistent storage. The primary security concern would be **unsafe deserialization** (e.g. if pickle-based model serialization is ever introduced) or **arbitrary code execution through custom scorers**. Please report any findings related to these areas.
