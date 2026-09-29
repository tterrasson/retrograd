# Web interface

`retrograd-server` can serve a web interface under `/`. From it you can start,
follow, debug and compare fine-tunings without writing TOML or curl. The
interface is a client of the `/v1` API like any other. Every refusal, every
estimate and every derived value comes from the server.

## Building it

The interface is embedded in the binary at build time, behind the `ui`
feature. The feature is off by default: a build without it needs neither Bun
nor the interface's build.

```bash
(cd web && bun install --frozen-lockfile && bun run build)
cargo build --release -p retrograd-server --features ui
```

If `web/dist` is missing, the Cargo build stops and prints the command that
produces it. Cargo never calls Bun.

## Serving it

With the feature compiled in, the interface is on by default. Turn it off
without rebuilding in the server's TOML:

```toml
ui = false
```

The interface's own files (everything outside `/v1`) are public: they are
the same static files for every installation and contain no data. The data
stays behind `/v1`.

- **With `auth_token`**, the interface asks for the token on its login screen.
  It keeps the token for the browser tab, or in the browser's storage if you
  tick "remember". That token gives full control of the server.
- **Without a token** (loopback only), there is no login screen.

`GET /v1/capabilities` reports `features.ui`, `features.auth` and
`features.serving_enabled`. The interface uses them to decide what it shows.

## Downloads

A browser cannot attach a token to a download. So the interface asks for a
**signed link** (`POST /v1/runs/{id}/artifacts/{name}/link`), then navigates
to it. A link is valid for one artefact, for 60 seconds, and stops working when
the server restarts. Any other route ignores a signature.

## Trajectories of a CLI run

To view a run started with `retrograd train` (no server involved), serve its
`[observe]` directory:

```bash
retrograd-server view runs/42/observe --open
```

This process only binds loopback and has no token. It is read-only and serves
only the interface and that directory's trajectories. See
[Observing rollouts](../training/observe).

A run started from a recipe on the server exports its trajectories by default
when its objective generates rollouts. The server's `observe_target_updates`
setting (default `100`) sets how many updates keep their texts. `GET /v1/defaults`
publishes the rule.

## Behind a reverse proxy

The interface follows runs over server-sent events. A proxy that buffers
responses delays them until the run ends. With nginx, set
`proxy_buffering off;` on `/v1/`.

## Developing it

```bash
cargo run -p retrograd-server -- server.toml     # the API
cd web && bun install && bun run dev             # the interface, proxying /v1
```

Vite proxies `/v1` to `http://127.0.0.1:8471` by default, matching the
server's default bind address. Set `RETROGRAD_URL` to proxy to another server.

The interface's types are generated from `web/openapi.json`. After changing a
route or a DTO, regenerate both. The server's test suite fails while they
disagree.

```bash
cargo run -p retrograd-server --bin retrograd-server -- openapi > web/openapi.json
(cd web && bun run gen:api)
```

`scripts/test-web.sh` is the interface's lane.
