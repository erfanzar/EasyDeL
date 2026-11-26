# Run a team plane on one VM

`sudo eyvan team up` turns a plain Debian or Ubuntu machine into an Eyvan
control plane that serves team accounts. Run it **on the machine itself**: it
does not create VMs, call `gcloud`, or open firewalls for you.

## Requirements

- Debian 12+ or Ubuntu 24.04+ (`apt-get`, `systemd`, Python 3.11 or newer).
- Root (`sudo`). `--dry-run` works without it.
- An IPv4 address people can reach. On a GCE VM with an external IP it is read
  from the metadata server; anywhere else pass `--address`.
- The Eyvan CLI on the machine, so `sudo eyvan team up` can start. The plane
  itself is installed into its own virtual environment.

## Install

```bash
sudo eyvan team up --dry-run          # read the plan and installer first
sudo eyvan team up
```

Useful options:

| Option | Meaning |
|---|---|
| `--address IP` | Address in the certificate and the URL (default: GCE external IP). |
| `--dns-name NAME` | Also put this name in the certificate and use it in the URL. |
| `--port N` | HTTPS port, default 8443. |
| `--admin USER` | First admin's user name (default: the account that ran `sudo`). |
| `--wheel PATH` | Install this wheel instead of the running Eyvan version from the package index. |
| `--source-range CIDR` | Used in the printed firewall command; repeatable. |
| `--json` | Machine-readable plan or result. |

When it finishes it prints:

1. the certificate's SHA-256 fingerprint;
2. a **one-time invite link** for the first admin. It is shown only once and
   is not logged anywhere else. Open it privately and choose a password;
3. the login command for your own machine, for example
   `eyvan login https://203.0.113.7:8443 --username erfan --fingerprint SHA256:AB:CD:...`;
4. the `gcloud` commands that would open the port. They are **not** run: open
   the port only to the networks your team uses.

Running it again is safe. It keeps the generated secrets, the CA (so pinned
fingerprints stay valid until the server certificate is reissued) and every
account. It does not create a second invite once an account exists. Invite
more people from Settings -> Users or with `eyvan users invite`.

## What it changes

| Path | Contents |
|---|---|
| `/etc/eyvan/plane.env` | root-only (0600): database URL, repository backend, artifact signing key, metrics token, TLS paths and public URL. No mode setting. |
| `/etc/eyvan/platform.yaml` | `team: true`. Existing sections, such as `agents`, are kept. |
| `/etc/eyvan/control-plane-ca.pem`, `/etc/eyvan/tls/` | A private CA (10 years, created once) and a server certificate (825 days, reissued when the address or name changes or it nears expiry). |
| `/opt/eyvan/venv` | Eyvan and its dependencies; `/usr/local/bin/eyvan` links to it. |
| PostgreSQL | Local `eyvan` role and database; the password is never passed in argv. |
| `eyvan-plane.service` | Runs `eyvan-server --host 0.0.0.0 --port N` as the `eyvan` system user. |

Team mode comes from `team: true` in the platform file, not from an
environment variable.

The installer then runs the schema migration, starts the unit, waits for
`/healthz` over HTTPS verified against the new CA, and asks
`eyvan-server bootstrap-admin` for the first admin invite.

## Relationship to `eyvan head up`

A head created by `eyvan head up` uses `/etc/eyvan/head.env`. `eyvan team up`
refuses to run on such a host; turn team accounts on there with
`eyvan head up --team` instead.

## Troubleshooting

- Plane not healthy: `journalctl -u eyvan-plane.service -n 100`.
- Cannot connect from your laptop: check the firewall rule and that the VM
  carries its target tag.
- `eyvan login` reports a different fingerprint: the certificate was reissued
  (new address or name, or renewal). Read the current one with
  `openssl x509 -in /etc/eyvan/tls/server.pem -noout -fingerprint -sha256`.
