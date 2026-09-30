import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { createRequire } from "node:module"

// ponytail — OpenCode V2 plugin.
//
// Injects the ponytail ruleset into every agent-loop model request at the active
// intensity, persists `/ponytail` mode switches, and registers the slash
// commands and skills so they work from a local install. The instruction
// builder is shared with the other ponytail hosts (Claude Code, Codex, pi), so
// the ruleset text itself is not duplicated here.
//
// V2 plugin contract: the default export must be a definition object with an
// `id` and a `setup` function. `Plugin.define` from `@opencode/plugin` is only a
// typing helper and that package is not resolvable from a plugin file on disk,
// so the object is declared inline.
//
// Ported from the V1 plugin shipped in @dietrichgebert/ponytail 4.10.0, which is
// V1-only. V1 -> V2 mapping used here:
//   config                          -> ctx.command.transform + ctx.skill.transform
//   experimental.chat.system.transform -> ctx.session.hook("context")
//   command.execute.before          -> execute() on the plugin's own command
//
// Vendored under ./vendor at the layout ponytail's own modules expect:
// hooks/ponytail-instructions.js resolves ../skills/ponytail/SKILL.md.

const require = createRequire(import.meta.url)
const VENDOR = path.join(path.dirname(fileURLToPath(import.meta.url)), "vendor")
const { getPonytailInstructions } = require(path.join(VENDOR, "hooks", "ponytail-instructions"))
const { getDefaultMode, normalizePersistedMode } = require(path.join(VENDOR, "hooks", "ponytail-config"))

// OpenCode has no flag-file convention of its own; keep mode beside its config.
const statePath = path.join(
  process.env.XDG_CONFIG_HOME || path.join(process.env.HOME || "", ".config"),
  "opencode",
  ".ponytail-active",
)

function readMode(): string {
  try {
    return normalizePersistedMode(fs.readFileSync(statePath, "utf8").trim()) || getDefaultMode()
  } catch {
    return getDefaultMode()
  }
}

function writeMode(mode: string): void {
  fs.mkdirSync(path.dirname(statePath), { recursive: true })
  fs.writeFileSync(statePath, mode)
}

// ponytail's command files use YAML frontmatter with folded (`>`) descriptions.
// Only the two fields the skill domain needs are read here.
function parseFrontmatter(content: string): { description?: string; body: string } {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/)
  if (!match) return { body: content.trim() }

  const [, front, body] = match
  let description: string | undefined
  const lines = front.split(/\r?\n/)
  for (let i = 0; i < lines.length; i++) {
    // Folded or literal block first: `description: >` also matches the plain
    // form below, which would otherwise capture the bare `>` marker.
    if (/^description:\s*[>|]\s*$/.test(lines[i])) {
      const parts: string[] = []
      for (let j = i + 1; j < lines.length; j++) {
        if (!/^\s+\S/.test(lines[j])) break
        parts.push(lines[j].trim())
      }
      description = parts.join(" ")
      continue
    }
    const direct = lines[i].match(/^description:\s*(.+)$/)
    if (direct) description = direct[1].trim().replace(/^["']|["']$/g, "")
  }
  return { description, body: body.trim() }
}

type CommandDefinition = {
  name: string
  description?: string
  template: string
}

type PluginContext = {
  command: {
    transform: (cb: (editor: { add: (def: unknown) => void }) => void) => Promise<unknown>
  }
  skill: {
    transform: (cb: (editor: { add: (def: unknown) => void }) => void) => Promise<unknown>
  }
  session: {
    hook: (
      name: "context",
      cb: (event: { system: Array<{ type: "text"; text: string }> }) => void,
    ) => Promise<unknown>
    prompt: (input: { sessionID: string; text: string; delivery?: string; files?: unknown }) => Promise<unknown>
  }
}

export default {
  id: "ponytail",

  async setup(ctx: PluginContext) {
    const commandDir = path.join(VENDOR, "command")
    const skillsDir = path.join(VENDOR, "skills")

    // Register the slash commands. `ponytail` additionally persists the mode it
    // switches to; the V1 plugin did that in `command.execute.before`.
    const definitions: CommandDefinition[] = []
    try {
      for (const file of fs.readdirSync(commandDir).filter((f) => f.endsWith(".md"))) {
        const full = path.join(commandDir, file)
        const { description, body } = parseFrontmatter(fs.readFileSync(full, "utf8"))
        if (!body) continue
        definitions.push({ name: path.basename(file, ".md"), description, template: body })
      }
    } catch {
      // No command directory — the plugin still injects the ruleset.
    }

    await ctx.command.transform((editor) => {
      for (const def of definitions) {
        editor.add({
          name: def.name,
          description: def.description,
          execute: async ({ sessionID, prompt, delivery }: any) => {
            let text = prompt?.text ?? ""

            if (def.name === "ponytail") {
              const args = String(prompt?.arguments ?? "").trim()
              // $ARGUMENTS is the only substitution ponytail's command file uses.
              const substituted = text.replace(/\$\{?ARGUMENTS\}?/g, args)
              const mode = args ? normalizePersistedMode(args) : getDefaultMode()
              if (mode) writeMode(mode)
              text = substituted
            }

            await ctx.session.prompt({ ...(prompt ?? {}), sessionID, text, delivery })
          },
        })
      }
    })

    // Register the skills. V1 pushed the directory onto `config.skills.paths`;
    // V2 takes explicit definitions, so each SKILL.md is read and its
    // frontmatter name/description lifted out. The definition field is `path`
    // (not `location`).
    try {
      const skills = fs
        .readdirSync(skillsDir, { withFileTypes: true })
        .filter((e) => e.isDirectory())
        .map((e) => {
          const skillPath = path.join(skillsDir, e.name, "SKILL.md")
          if (!fs.existsSync(skillPath)) return null
          const raw = fs.readFileSync(skillPath, "utf8")
          const { description, body } = parseFrontmatter(raw)
          return { id: e.name, name: e.name, description: description ?? "", path: skillPath, content: body }
        })
        .filter((s): s is NonNullable<typeof s> => s !== null)

      if (skills.length > 0) {
        await ctx.skill.transform((editor) => {
          for (const skill of skills) editor.add(skill)
        })
      }
    } catch {
      // No skills directory — commands and injection still work.
    }

    // Append the ruleset to the agent loop's system prompt every request.
    await ctx.session.hook("context", (event) => {
      const mode = readMode()
      if (mode === "off") return
      const instructions = getPonytailInstructions(mode)
      if (!instructions) return
      event.system.push({ type: "text", text: instructions })
    })
  },
}
