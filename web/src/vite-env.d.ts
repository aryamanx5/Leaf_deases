/// <reference types="vite/client" />
interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  readonly VITE_DEPLOYED_APP_URL?: string;
  readonly VITE_GITHUB_URL?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
