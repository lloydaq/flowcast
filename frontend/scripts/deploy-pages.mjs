// Publish the static demo to the repository's gh-pages branch (GitHub Pages serves it at /flowcast/).
//
// Usage, from frontend/ (needs the data and saved predictions locally):
//   python ../scripts/export_static.py
//   npm run deploy
//
// The branch holds only the latest build: each deploy replaces it with one fresh commit, so the
// repository does not grow by 27 MB per deploy. The source of every deploy is named in its message.
import { execFileSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const FRONTEND = join(dirname(fileURLToPath(import.meta.url)), "..");
const DIST = join(FRONTEND, "dist-static");
// npm is a .cmd file on Windows and needs a shell; git must not have one (it would split the message)
const run = (cmd, args, cwd = FRONTEND) =>
  execFileSync(cmd, args, { cwd, stdio: "inherit", shell: cmd === "npm" && process.platform === "win32" });
const out = (cmd, args, cwd = FRONTEND) => execFileSync(cmd, args, { cwd, encoding: "utf-8" }).trim();

if (!existsSync(join(FRONTEND, ".static", "data", "grid.json")))
  throw new Error("No exported data: run `python scripts/export_static.py` from the repository root first.");

const remote = out("git", ["remote", "get-url", "origin"]);
const source = out("git", ["rev-parse", "--short", "HEAD"]);
const dirty = out("git", ["status", "--porcelain"]) !== "";

run("npm", ["run", "build:static"]);
rmSync(join(DIST, ".git"), { recursive: true, force: true });
run("git", ["init", "-q", "-b", "gh-pages"], DIST);
run("git", ["add", "-A"], DIST);
run("git", ["commit", "-q", "-m", `Deploy the FlowCast demo from ${source}${dirty ? " (with uncommitted changes)" : ""}`], DIST);
run("git", ["push", "--force", "-q", remote, "gh-pages"], DIST);
rmSync(join(DIST, ".git"), { recursive: true, force: true });
console.log(`Deployed ${source} to gh-pages. Live at https://lloydaq.github.io/flowcast/ once Pages serves that branch.`);
