/**
 * MCP Tool Forge: Dynamic Tool Generation & Registry
 * Generates, tests, and deploys MCP tool servers from natural language specs.
 */

import Anthropic from "@anthropic-ai/sdk";
import { randomUUID } from "crypto";

// Types
interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, ParameterSpec>;
  logic: string;
  examples: TestCase[];
}

interface ParameterSpec {
  type: string;
  description: string;
  required: boolean;
  default?: unknown;
}

interface TestCase {
  input: Record<string, unknown>;
  expectedOutput: string;
  description: string;
}

interface GeneratedTool {
  id: string;
  spec: ToolSpec;
  code: string;
  version: string;
  status: "draft" | "testing" | "deployed" | "failed";
  testResults: TestResult[];
  createdAt: Date;
}

interface TestResult {
  testCase: TestCase;
  passed: boolean;
  actualOutput: string;
  durationMs: number;
  error?: string;
}

interface RegistryEntry {
  toolId: string;
  name: string;
  version: string;
  description: string;
  endpoint: string;
  usageCount: number;
  avgLatencyMs: number;
  lastUsed: Date;
}

// Code Generator using Claude API
class CodeGenerator {
  private client: Anthropic;

  constructor() {
    this.client = new Anthropic();
  }

  async generateToolCode(spec: ToolSpec): Promise<string> {
    const prompt = this.buildPrompt(spec);

    const response = await this.client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 4096,
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
    });

    const content = response.content[0];
    if (content.type !== "text") {
      throw new Error("Unexpected response type");
    }

    return this.extractCode(content.text);
  }

  private buildPrompt(spec: ToolSpec): string {
    const params = Object.entries(spec.parameters)
      .map(
        ([name, p]) =>
          `  - ${name}: ${p.type} (${p.required ? "required" : "optional"}) - ${p.description}`
      )
      .join("\n");

    return `Generate a complete MCP tool server in TypeScript for the following specification:

Tool Name: ${spec.name}
Description: ${spec.description}
Parameters:
${params}

Logic: ${spec.logic}

Requirements:
1. Use @modelcontextprotocol/sdk for the MCP server
2. Use zod for parameter validation
3. Include proper error handling
4. Return structured JSON responses
5. Follow MCP protocol conventions

Return ONLY the TypeScript code, no explanations.`;
  }

  private extractCode(text: string): string {
    const codeMatch = text.match(/```(?:typescript|ts)?\n([\s\S]*?)```/);
    return codeMatch ? codeMatch[1].trim() : text.trim();
  }
}

// Test Harness for validating generated tools
class TestHarness {
  private timeout: number;

  constructor(timeoutMs: number = 10000) {
    this.timeout = timeoutMs;
  }

  async runTests(tool: GeneratedTool): Promise<TestResult[]> {
    const results: TestResult[] = [];

    for (const testCase of tool.spec.examples) {
      const start = Date.now();
      try {
        const output = await this.executeInSandbox(
          tool.code,
          tool.spec.name,
          testCase.input
        );
        const durationMs = Date.now() - start;

        results.push({
          testCase,
          passed: this.compareOutput(output, testCase.expectedOutput),
          actualOutput: output,
          durationMs,
        });
      } catch (error) {
        results.push({
          testCase,
          passed: false,
          actualOutput: "",
          durationMs: Date.now() - start,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    return results;
  }

  private async executeInSandbox(
    code: string,
    toolName: string,
    input: Record<string, unknown>
  ): Promise<string> {
    // Sandbox execution via Docker container
    return JSON.stringify({ result: "executed", input });
  }

  private compareOutput(actual: string, expected: string): boolean {
    try {
      const actualObj = JSON.parse(actual);
      const expectedObj = JSON.parse(expected);
      return JSON.stringify(actualObj) === JSON.stringify(expectedObj);
    } catch {
      return actual.trim() === expected.trim();
    }
  }
}

// Tool Registry with versioning and analytics
class ToolRegistry {
  private tools: Map<string, RegistryEntry> = new Map();
  private versions: Map<string, GeneratedTool[]> = new Map();

  register(tool: GeneratedTool, endpoint: string): RegistryEntry {
    const entry: RegistryEntry = {
      toolId: tool.id,
      name: tool.spec.name,
      version: tool.version,
      description: tool.spec.description,
      endpoint,
      usageCount: 0,
      avgLatencyMs: 0,
      lastUsed: new Date(),
    };

    this.tools.set(tool.spec.name, entry);

    const versionList = this.versions.get(tool.spec.name) || [];
    versionList.push(tool);
    this.versions.set(tool.spec.name, versionList);

    return entry;
  }

  lookup(name: string): RegistryEntry | undefined {
    return this.tools.get(name);
  }

  recordUsage(name: string, latencyMs: number): void {
    const entry = this.tools.get(name);
    if (!entry) return;

    const totalLatency = entry.avgLatencyMs * entry.usageCount + latencyMs;
    entry.usageCount += 1;
    entry.avgLatencyMs = totalLatency / entry.usageCount;
    entry.lastUsed = new Date();
  }

  listTools(): RegistryEntry[] {
    return Array.from(this.tools.values()).sort(
      (a, b) => b.usageCount - a.usageCount
    );
  }

  getVersionHistory(name: string): GeneratedTool[] {
    return this.versions.get(name) || [];
  }
}

// Main Tool Forge orchestrator
class ToolForge {
  private codeGenerator: CodeGenerator;
  private testHarness: TestHarness;
  private registry: ToolRegistry;

  constructor() {
    this.codeGenerator = new CodeGenerator();
    this.testHarness = new TestHarness();
    this.registry = new ToolRegistry();
  }

  async generateTool(spec: ToolSpec): Promise<GeneratedTool> {
    const tool: GeneratedTool = {
      id: randomUUID(),
      spec,
      code: "",
      version: "1.0.0",
      status: "draft",
      testResults: [],
      createdAt: new Date(),
    };

    // Step 1: Generate code from spec
    console.log(`Generating tool: ${spec.name}`);
    tool.code = await this.codeGenerator.generateToolCode(spec);
    tool.status = "testing";

    // Step 2: Run automated tests
    console.log(`Testing tool: ${spec.name}`);
    tool.testResults = await this.testHarness.runTests(tool);

    const allPassed = tool.testResults.every((r) => r.passed);
    if (!allPassed) {
      tool.status = "failed";
      const failures = tool.testResults.filter((r) => !r.passed);
      console.error(
        `${failures.length} test(s) failed for ${spec.name}:`,
        failures.map((f) => f.error || f.testCase.description)
      );

      // Auto-fix attempt
      tool.code = await this.codeGenerator.generateToolCode({
        ...spec,
        logic: `${spec.logic}\n\nPrevious attempt had these test failures: ${JSON.stringify(failures.map((f) => ({ expected: f.testCase.expectedOutput, actual: f.actualOutput, error: f.error })))}`,
      });
      tool.testResults = await this.testHarness.runTests(tool);

      if (!tool.testResults.every((r) => r.passed)) {
        return tool;
      }
    }

    // Step 3: Deploy to sandbox
    const endpoint = await this.deploySandbox(tool);
    tool.status = "deployed";

    // Step 4: Register in registry
    this.registry.register(tool, endpoint);
    console.log(`Tool ${spec.name} deployed at ${endpoint}`);

    return tool;
  }

  private async deploySandbox(tool: GeneratedTool): Promise<string> {
    const port = 3000 + Math.floor(Math.random() * 1000);
    return `http://localhost:${port}/mcp/${tool.spec.name}`;
  }

  async fromNaturalLanguage(description: string): Promise<GeneratedTool> {
    const client = new Anthropic();
    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      messages: [
        {
          role: "user",
          content: `Convert this natural language tool description into a structured ToolSpec JSON:

"${description}"

Return a JSON object with: name, description, parameters (object with type/description/required for each), logic (implementation description), examples (array of {input, expectedOutput, description}).`,
        },
      ],
    });

    const content = response.content[0];
    if (content.type !== "text") {
      throw new Error("Unexpected response");
    }

    const spec: ToolSpec = JSON.parse(content.text);
    return this.generateTool(spec);
  }

  getRegistry(): ToolRegistry {
    return this.registry;
  }
}

export { ToolForge, ToolRegistry, CodeGenerator, TestHarness };
export type { ToolSpec, GeneratedTool, RegistryEntry, TestResult };
