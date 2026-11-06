import { p as publicAssetsURL } from '../routes/renderer.mjs';
import 'vue-bundle-renderer/runtime';
import '../_/nitro.mjs';
import 'node:http';
import 'node:https';
import 'node:events';
import 'node:buffer';
import 'node:fs';
import 'node:path';
import 'node:crypto';
import 'node:url';
import 'vue/server-renderer';
import 'unhead/server';
import 'devalue';
import 'unhead/utils';
import 'vue';

const StartupLoadingOverlay_vue_vue_type_style_index_0_scoped_9ebb4c7c_lang = ".startup-loading-overlay[data-v-9ebb4c7c]{position:fixed;inset:0;z-index:1000;display:grid;place-items:center;min-width:0;background:var(--asteria-bg);color:var(--asteria-text);cursor:wait}.startup-loading-overlay__status[data-v-9ebb4c7c]{display:grid;justify-items:center;gap:18px}.startup-loading-overlay__status strong[data-v-9ebb4c7c]{color:var(--asteria-text);font-size:13px;font-weight:650;letter-spacing:.08em}.startup-loading-overlay__mark[data-v-9ebb4c7c]{width:42px;height:44px;background:var(--asteria-text-strong);-webkit-mask:url(" + publicAssetsURL("/brand/asteria-mark-navy.svg") + ') center/contain no-repeat;mask:url(/brand/asteria-mark-navy.svg) center/contain no-repeat}.startup-loading-overlay__spinner[data-v-9ebb4c7c]{position:relative;width:46px;height:46px;border:2px solid color-mix(in srgb,var(--asteria-loading-indicator) 26%,transparent);border-top-color:var(--asteria-loading-indicator);border-radius:50%;color:var(--asteria-loading-indicator);animation:startup-loading-spin-9ebb4c7c .9s linear infinite}.startup-loading-overlay__spinner[data-v-9ebb4c7c]:after{position:absolute;inset:7px;border:1px solid color-mix(in srgb,var(--asteria-loading-indicator) 24%,transparent);border-right-color:var(--asteria-loading-indicator);border-radius:inherit;animation:startup-loading-spin-9ebb4c7c 1.35s linear infinite reverse;content:""}.startup-loading-enter-active[data-v-9ebb4c7c],.startup-loading-leave-active[data-v-9ebb4c7c]{transition:opacity .2s ease}.startup-loading-enter-from[data-v-9ebb4c7c],.startup-loading-leave-to[data-v-9ebb4c7c]{opacity:0}@keyframes startup-loading-spin-9ebb4c7c{to{transform:rotate(1turn)}}@media(prefers-reduced-motion:reduce){.startup-loading-overlay__spinner[data-v-9ebb4c7c],.startup-loading-overlay__spinner[data-v-9ebb4c7c]:after{animation:none}.startup-loading-enter-active[data-v-9ebb4c7c],.startup-loading-leave-active[data-v-9ebb4c7c]{transition-duration:.01ms}}';

const StartupLoadingOverlayStyles_B3GCivrK = [
  StartupLoadingOverlay_vue_vue_type_style_index_0_scoped_9ebb4c7c_lang
];

export { StartupLoadingOverlayStyles_B3GCivrK as default };
//# sourceMappingURL=StartupLoadingOverlay-styles.B3GCivrK.mjs.map
