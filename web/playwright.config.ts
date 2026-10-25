import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: /.*\.spec\.ts/,
  timeout: 120_000,
  use: { baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:5180', trace: 'retain-on-failure' },
  reporter: [['list']],
});
