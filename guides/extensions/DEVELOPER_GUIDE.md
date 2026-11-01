# Asteria Extension Developer Guide

[日本語](DEVELOPER_GUIDE.ja.md) | [English](DEVELOPER_GUIDE.md)

Asteria V1.2 supports two execution levels:

- **Standard extension**: runs in a sandboxed browser frame and requests only the operations allowed by Asteria's broker. This is the default choice.
- **Full-access extension**: runs author-supplied Python and JavaScript with the same user-level access as Asteria. Use it only when the standard API cannot express the feature.

## Capability boundary

### What a standard extension can do

A standard extension can:

- add an independent left-navigation workspace;
- add declared panels and actions to Generate, Image Edit, Media Library, LoRA Training, Merge Studio, and Settings;
- navigate to an existing Asteria workspace;
- read the current positive and negative prompts when `prompt.read` is declared;
- connect through Asteria's network broker only to manifest-declared hosts;
- run JavaScript or WASM inside the sandboxed frame.

Practical examples include a prompt review panel, a prompt assistant that calls a declared external API, and a workspace that guides the user through existing Asteria screens.

### What a standard extension cannot do

A standard extension cannot directly:

- install ComfyUI custom nodes or Python packages;
- run `pip install`, `npm install`, `setup.py`, or package-defined install scripts;
- execute a Python backend, Node.js process, or arbitrary local code;
- launch arbitrary executables, PowerShell, or Command Prompt;
- import or patch Asteria internals, access the database, private stores, or private DOM;
- read or write local files and folders;
- read selected Media Library items, edit metadata, start generation, or register outputs;
- connect to undeclared URLs, scan the local network, or scan `localhost`;
- run custom CUDA, PyTorch, or native GPU code;
- leave a resident process behind after Asteria exits.

Choose a full-access extension if the feature requires one of these capabilities. Do not hide a local server or executable inside a standard extension to bypass the boundary.

### What a full-access extension can do

A full-access extension retains the freedom of Asteria's original code-execution format. It can integrate custom ComfyUI nodes, native libraries, CUDA workloads, external programs, local services, and workflows that cannot be represented by the public API.

It also runs with the same user-level file, network, and process access as Asteria. Asteria therefore presents an explicit warning and enables the extension only after the user approves that access. Safe Mode, disable, and rollback help recover Asteria startup; they cannot undo data theft or damage caused before the extension was stopped.

| Feature | Execution level |
| --- | --- |
| Read the current prompt and show suggestions | Standard extension |
| Call a declared HTTPS API or localhost endpoint through the broker | Standard extension |
| Add a workspace or panel that guides existing Asteria tasks | Standard extension |
| Process Media Library files, update metadata, or start generation | Full-access extension |
| Run Python, Node.js, or third-party local libraries | Full-access extension |
| Custom ComfyUI nodes, CUDA, native libraries | Full-access extension |
| Arbitrary executables or local servers | Full-access extension |
| Direct changes to Asteria internals | Consider proposing a core Asteria change |

Start with a standard extension. If a public API is missing, first consider whether Asteria should add a narrowly scoped brokered operation instead of requesting full access.

## Package shape

Every extension package contains a strict manifest, a prebuilt frontend ESM bundle, documentation, and a real license. The V1 package shape also requires a Python entrypoint path.

A Standard extension's packaged Python file is inert and is never imported or executed. Its frontend runs in an iframe with `sandbox="allow-scripts"`. In V1.2, the broker supports navigation, two safe core commands, `prompt.read`, and network requests to exact declared HTTPS origins or localhost host/port pairs.

A Full-access extension package additionally uses:

- a strict `asteria-extension.json` manifest;
- a Python backend entrypoint loaded in the Asteria process;
- a prebuilt Vue 3 + TypeScript Custom Element ESM bundle;
- package documentation and a real license.

Standard extensions declare their brokered operations and network hosts in advance. Asteria displays that scope during installation and enforces it at the broker boundary.

Full-access authors must declare `network`, `filesystem`, and `subprocess` capabilities honestly. A capability declaration informs the user; it is not an operating-system permission boundary for full-access code.

## Start from the scaffold

Install the Extension Kit dependencies once, then create a project:

```powershell
cd extension-kit
npm install
cd ..
node extension-kit/scripts/create-extension.mjs --output extensions/my-extension
```

The command creates a Standard extension by default. Add `--execution-level full_access` to create a Full-access extension. It asks for a globally unique extension ID, display name, and semantic version when they are not passed as options. IDs should use a reverse-domain form such as `com.example.my-extension`.

Replace both generated license placeholders before building. The builder rejects placeholder license metadata or text.

## Manifest contract

The canonical schema is `schemas/asteria-extension-v1.schema.json`. The root fields are:

- `schema_version`: `asteria.extension.v1`;
- `id`, `name`, and semantic `version`;
- an Asteria `minimum` and exclusive `maximum_exclusive` compatibility range;
- Python and frontend entrypoints plus the declared `frontend_element` Custom Element name;
- declared workspaces, panels, and actions;
- `execution_level` plus either Standard `permissions` or Full-access `capabilities`;
- license ID and URL.

An omitted `execution_level` is treated as `full_access` for existing V1 packages. Standard extensions declare only `prompt.read`, exact HTTPS origins, and localhost host/port pairs that they need. HTTPS origin hostnames must be lowercase, and the default port `443` must be omitted. They cannot declare `capabilities`, `backend.api_router`, or `backend.operation`. Full-access extensions keep `permissions` empty and declare any `network`, `filesystem`, or `subprocess` capabilities honestly.

Contribution IDs are unique within one extension. A Full-access backend entrypoint must register every contribution declared by the manifest; undeclared or duplicate registrations fail that extension without preventing the Asteria core from starting.

Supported V1.2 contribution surfaces are `independent_workspace`, `generate.inspector_panel`, `generate.inspector_action`, `image_edit.inspector_panel`, `image_edit.inspector_action`, `media_library.details`, `media_library.context_action`, `lora_training.tool`, `lora_training.action`, `merge_studio.tool`, `merge_studio.action`, `settings.integration_card`, `backend.api_router`, and `backend.operation`. The final two surfaces are Full-access only.

Backend contributions are an API router and an operation executor. Asteria mounts routers below `/api/extensions/<extension-id>/`; extensions declare only their relative router paths. Operation kinds are namespaced by Asteria. Raw database access, private Pinia stores, core DOM selectors, and runtime node IDs are not SDK contracts.

## Standard frontend and broker

A Standard frontend exports the `asteria.extension.standard.frontend.v1` contract. Render only inside the supplied `root` and request Asteria operations through `bridge.request`:

```ts
export const asteriaStandardExtension = {
  contractVersion: 'asteria.extension.standard.frontend.v1',
  async mount({ root, bridge }) {
    const { prompt, negativePrompt } = await bridge.request('prompt.read', null)
    const output = document.createElement('pre')
    output.textContent = `${prompt}\n\n${negativePrompt}`
    root.replaceChildren(output)
  },
}
```

The V1.2 broker supports `navigate`, `command`, `prompt.read`, and `network.fetch`. Network requests accept only `GET` and `POST`. A target must exactly match a declared HTTPS origin or localhost host/port pair. Redirects, credential-bearing URLs, undeclared targets, and external hosts that resolve to private addresses are rejected. The broker handles at most four requests at once; retry after an active request completes when it reports that it is busy.

Ports 3000, 3001, 5173, 8000, and 8188 are reserved for Asteria and ComfyUI and cannot be declared as Standard-extension localhost targets. Use another port for an existing local service. A feature that must call Asteria's internal API or ComfyUI directly belongs in a Full-access extension.

The packaged `backend/extension.py` only preserves the V1 package shape for a Standard extension. Asteria never imports or executes it. Choose Full access when Python execution is required.

## Backend entrypoint (Full access)

The manifest uses `module.path:function`, for example `backend.extension:create_extension`. The function receives the public registration context:

```python
from fastapi import APIRouter
from app.extensions.public_api import OperationExecutionContext, OperationExecutionResult


class MyExecutor:
    version = "1"

    def resource_claims(self, request_snapshot: dict) -> tuple[str, ...]:
        return ("workspace:my-extension",)

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        context.report_progress(1, 1, "item", phase="completed")
        return OperationExecutionResult()


def create_extension(context) -> None:
    router = APIRouter()

    @router.get("/status")
    def status() -> dict[str, str]:
        return {"status": "ready"}

    context.register_router("router.api", router)
    context.register_workspace("workspace.main")
    context.register_panel("panel.generate")
    context.register_action("action.generate")
    context.register_operation("operation.run", MyExecutor())
```

For author-side typing, copy or import the registration context and protocols from `extension-kit/src/backend/asteria_extension.py`. Runtime execution types are formally provided by `app.extensions.public_api`; do not import Asteria's internal service modules from an extension.

An operation executor exposes a `version` string, `resource_claims` derived from its immutable request snapshot, and an `execute` method. In V1.2, `execute` receives Asteria's `OperationExecutionContext` and returns `OperationExecutionResult`; do not return an arbitrary dictionary. Register it with `context.register_operation` for the manifest contribution ID; never access the shared coordinator directly.

`app.extensions.public_api` is the official entrypoint resolved only when the host loads an extension backend at runtime. Use the Extension Kit protocols for author-only type checks and this public entrypoint for Asteria-integrated execution.

## Custom Element frontend (Full access)

The frontend module exports `asteriaExtension` with the versioned contract:

```ts
import { defineCustomElement, h } from 'vue'

const MyCustomElement = defineCustomElement({
  render() {
    return h('div', 'My extension')
  },
})

export const asteriaExtension = {
  contractVersion: 'asteria.extension.frontend.v1',
  elementName: 'asteria-com-example-my-extension',
  register() {
    if (!customElements.get(this.elementName)) {
      customElements.define(this.elementName, MyCustomElement)
    }
  },
}
```

The host sets these four attributes on the Custom Element: `data-asteria-extension-id`, `data-asteria-contribution-id`, `data-asteria-contribution-kind`, and `data-asteria-contribution-surface`. To request navigation, dispatch a composed, bubbling `asteria:navigate` event with `{ target }`. To invoke a core command, dispatch `asteria:command` with `{ id, payload }`. The host validates both event shapes and decides what is allowed.

Do not ship raw `.vue` files as the declared entrypoint. The Extension Kit bundles Vue and TypeScript into the manifest's `.mjs` output before packaging.

## Test and build

Run author tests first, then build:

```powershell
npm --prefix extensions/my-extension test
node extension-kit/scripts/build-extension.mjs --project extensions/my-extension
```

The builder validates the manifest, required files, license, and contained entrypoints. It writes an inspectable `dist/package` tree and a deterministic `dist/<id>-<version>.zip`. Identical source bytes produce the same ZIP checksum.

The builder does not install, publish, run `setup.py`, run package scripts, or resolve Python dependencies. Asteria does not run `npm install` or `pip install` while installing an extension. Vendor any required runtime files inside the package during author-controlled build work. Asteria validates and installs the resulting package but does not run extension-defined setup commands.

## Install and lifecycle expectations

Install, update, enable/disable, rollback, and uninstall changes require an Asteria restart. Updates occur only after the user explicitly requests them. Managed installations keep one previous version for rollback; manual unmanaged folders do not receive managed update or rollback.

During development, use a manual folder or install the built ZIP through the Extensions workspace. A load failure is isolated to the extension. Users can disable it, restart in Safe Mode, reconnect a missing manual folder, or roll back a managed update.

The generated Standard scaffold is the smallest buildable Standard example. The complete Full-access example is `examples/extensions/sample-full`. It contains the manifest, Python backend, Custom Element, build configuration, and a real MIT license. Before distribution, replace the sample ID, display name, version, and other identity values with your own.
