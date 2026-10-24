import { cpSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
mkdirSync(resolve(root, 'public/wasm'), { recursive: true });
cpSync(
  resolve(root, 'node_modules/@litertjs/core/wasm'),
  resolve(root, 'public/wasm'),
  { recursive: true },
);
