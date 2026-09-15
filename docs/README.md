# Retrograd documentation

This directory contains the VitePress documentation for Retrograd. It includes
both user guides and the engineering notes and design records needed to keep
the implementation, tests, and operational procedures aligned.

Install the documentation dependencies with Bun:

```bash
bun install
```

Start the local documentation server:

```bash
bun run docs:dev
```

Build the static site:

```bash
bun run docs:build
```

Preview the built site:

```bash
bun run docs:preview
```

Use Bun for documentation commands. The training project itself is built with
Cargo; see the user documentation for the CLI workflow.
