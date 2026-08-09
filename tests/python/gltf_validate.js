#!/usr/bin/env node
/**
 * gltf_validate.js — Thin CLI wrapper around the Khronos gltf-validator npm library.
 *
 * Usage: node gltf_validate.js <file.glb|file.gltf>
 *
 * Exits 0 if validation reports 0 errors; exits 1 on errors.
 * Prints a JSON report to stdout.
 *
 * Dependencies (install next to this script or globally):
 *   npm install gltf-validator
 */

"use strict";

const fs = require("fs");
const path = require("path");

// Resolve gltf-validator relative to this script, then fall back to global.
let validator;
try {
  // Local install (next to this script or in node_modules alongside)
  const localPath = path.join(__dirname, "node_modules", "gltf-validator");
  validator = require(localPath);
} catch (_) {
  try {
    validator = require("gltf-validator");
  } catch (err) {
    process.stderr.write(
      `[gltf_validate.js] Cannot find gltf-validator: ${err.message}\n` +
      `  Install with: npm install gltf-validator (in tests/python/ or globally)\n`
    );
    process.exit(2); // exit 2 = tool not available (script treats this as SKIP)
  }
}

const filePath = process.argv[2];
if (!filePath) {
  process.stderr.write("Usage: node gltf_validate.js <file.glb>\n");
  process.exit(1);
}

let asset;
try {
  asset = fs.readFileSync(filePath);
} catch (err) {
  process.stderr.write(`[gltf_validate.js] Cannot read file: ${err.message}\n`);
  process.exit(1);
}

validator
  .validateBytes(new Uint8Array(asset), {
    uri: path.basename(filePath),
    maxIssues: 100,
    writeTimestamp: false,
  })
  .then((report) => {
    process.stdout.write(JSON.stringify(report, null, 2) + "\n");
    const numErrors = (report.issues && report.issues.numErrors) || 0;
    process.exit(numErrors === 0 ? 0 : 1);
  })
  .catch((err) => {
    process.stderr.write(`[gltf_validate.js] Validation threw: ${err}\n`);
    process.exit(1);
  });
