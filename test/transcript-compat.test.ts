import type { Context, Tool } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import {
	adaptContextForModule,
	adaptTranscriptContext,
	ensureTranscriptContext,
	supportsTranscriptSource,
} from "../extensions/codex-stream.ts";

describe("adaptTranscriptContext", () => {
	it("leaves classic context without system messages untouched", () => {
		const classicContext: Context = {
			systemPrompt: "You are a classic assistant.",
			tools: [
				{
					name: "read",
					description: "Read file",
					parameters: { type: "object", properties: {} },
				} as Tool,
			],
			messages: [{ role: "user", content: "Hello world", timestamp: 1000 }],
		};

		const result = adaptTranscriptContext(classicContext);
		expect(result).toBe(classicContext);
		expect(result.systemPrompt).toBe("You are a classic assistant.");
		expect(result.tools).toHaveLength(1);
		expect(result.messages).toHaveLength(1);
	});

	it("extracts systemPrompt and tools from pi 0.86.0 initial system message and filters system message", () => {
		const transcriptContext = {
			messages: [
				{
					role: "system",
					content: "Base preamble text",
					sections: {
						rules: "<rules>\n- Be concise\n</rules>",
						project: "<project_instructions>\nInstruction 1\n</project_instructions>",
					},
					toolsAdded: [
						{
							name: "read",
							description: "Read file",
							parameters: { type: "object", properties: {} },
						},
						{
							name: "bash",
							description: "Run bash",
							parameters: { type: "object", properties: {} },
						},
					],
					timestamp: 0,
				},
				{ role: "user", content: "What is this project?", timestamp: 1000 },
			],
		};

		const result = adaptTranscriptContext(transcriptContext);

		expect(result.systemPrompt).toContain("Base preamble text");
		expect(result.systemPrompt).toContain("<rules>\n- Be concise\n</rules>");
		expect(result.systemPrompt).toContain("<project_instructions>\nInstruction 1\n</project_instructions>");
		expect(result.tools).toHaveLength(2);
		expect(result.tools?.map((t) => t.name)).toEqual(["read", "bash"]);
		expect(result.messages).toHaveLength(1);
		expect(result.messages[0]).toEqual({ role: "user", content: "What is this project?", timestamp: 1000 });
	});

	it("handles mid-conversation sections and tool updates", () => {
		const contextWithUpdates = {
			messages: [
				{
					role: "system",
					content: "",
					sections: {
						rules: "Rule 1",
						doc: "Doc 1",
					},
					toolsAdded: [{ name: "toolA", description: "Tool A", parameters: {} }],
				},
				{ role: "user", content: "Use tool A", timestamp: 1000 },
				{
					role: "assistant",
					content: [{ type: "toolCall", id: "call_1", name: "toolA", arguments: {} }],
					api: "cliproxyapi-codex-responses",
					provider: "cliproxyapi",
					model: "test-model",
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: "toolUse",
					timestamp: 1001,
				},
				{
					role: "toolResult",
					toolCallId: "call_1",
					toolName: "toolA",
					content: [{ type: "text", text: "done" }],
					isError: false,
					timestamp: 1002,
				},
				{
					role: "system",
					sections: {
						doc: null, // deleted
						rules: "Rule 1 Updated", // updated
						extra: "Extra section", // added
					},
					toolsRemoved: [{ name: "toolA" }],
					toolsAdded: [{ name: "toolB", description: "Tool B", parameters: {} }],
					timestamp: 1003,
				},
				{ role: "user", content: "Use tool B", timestamp: 1004 },
			],
		};

		const result = adaptTranscriptContext(contextWithUpdates);

		expect(result.systemPrompt).toContain("Rule 1 Updated");
		expect(result.systemPrompt).toContain("Extra section");
		expect(result.systemPrompt).not.toContain("Doc 1");
		expect(result.tools).toHaveLength(1);
		expect(result.tools?.[0]?.name).toBe("toolB");
		expect(result.messages).toHaveLength(4);
		expect(result.messages.map((m) => m.role)).toEqual(["user", "assistant", "toolResult", "user"]);
	});

	it("handles empty or boundary inputs gracefully", () => {
		expect(adaptTranscriptContext({} as Context)).toEqual({});
		expect(adaptTranscriptContext({ messages: [] } as unknown as Context)).toEqual({ messages: [] });
	});

	it("is idempotent when called multiple times", () => {
		const transcriptContext = {
			messages: [
				{
					role: "system",
					content: "Preamble",
					toolsAdded: [{ name: "read", description: "Read", parameters: {} }],
				},
				{ role: "user", content: "Hi", timestamp: 1000 },
			],
		};

		const first = adaptTranscriptContext(transcriptContext);
		const second = adaptTranscriptContext(first);
		expect(second).toBe(first);
		expect(second.systemPrompt).toBe("Preamble");
		expect(second.tools).toHaveLength(1);
		expect(second.messages).toHaveLength(1);
	});
});

describe("ensureTranscriptContext", () => {
	it("leaves transcript context with system message untouched", () => {
		const transcriptContext = {
			messages: [
				{
					role: "system",
					content: "Preamble",
					toolsAdded: [{ name: "bash", description: "Run bash", parameters: {} }],
				},
				{ role: "user", content: "Hello", timestamp: 1000 },
			],
		};

		const result = ensureTranscriptContext(transcriptContext);
		expect(result).toBe(transcriptContext);
		expect(result.messages).toHaveLength(2);
		expect((result.messages[0] as { toolsAdded?: unknown[] }).toolsAdded).toHaveLength(1);
	});

	it("converts classic context into transcript context by prepending a system message", () => {
		const classicContext: Context = {
			systemPrompt: "Classic instructions",
			tools: [
				{
					name: "read",
					description: "Read file",
					parameters: { type: "object", properties: {} },
				} as Tool,
			],
			messages: [{ role: "user", content: "Hello world", timestamp: 1000 }],
		};

		const result = ensureTranscriptContext(classicContext);
		expect(result.messages).toHaveLength(2);
		const firstMsg = result.messages[0] as { role: string; content: string; toolsAdded?: Tool[] };
		expect(firstMsg.role).toBe("system");
		expect(firstMsg.content).toBe("Classic instructions");
		expect(firstMsg.toolsAdded).toHaveLength(1);
		expect(firstMsg.toolsAdded?.[0]?.name).toBe("read");
		expect(result.messages[1]).toEqual({ role: "user", content: "Hello world", timestamp: 1000 });
	});

	it("leaves empty context untouched", () => {
		expect(ensureTranscriptContext({} as Context)).toEqual({});
		expect(ensureTranscriptContext({ messages: [] } as unknown as Context)).toEqual({ messages: [] });
	});
});

describe("supportsTranscriptSource", () => {
	it("detects transcript capability from source code", () => {
		expect(supportsTranscriptSource("import { resolveTranscriptTools } from '../utils/transcript.js';")).toBe(true);
		expect(supportsTranscriptSource("function resolveTranscript(context) {}")).toBe(true);
		expect(supportsTranscriptSource("function buildRequestBody(model, context, options) {}")).toBe(false);
	});
});

describe("adaptContextForModule", () => {
	const transcriptContext = {
		messages: [
			{
				role: "system",
				content: "Base prompt",
				toolsAdded: [{ name: "bash", description: "Run bash", parameters: {} }],
			},
			{ role: "user", content: "Run pwd", timestamp: 1000 },
		],
	};

	it("preserves transcript context when host module supports transcript (Pi 0.86+)", () => {
		const result = adaptContextForModule(transcriptContext, true);
		expect(result).toBe(transcriptContext);
		expect(result.messages).toHaveLength(2);
		expect((result.messages[0] as { toolsAdded?: unknown[] }).toolsAdded).toBeDefined();
	});

	it("adapts transcript context to classic context when host module is legacy (Pi < 0.86)", () => {
		const result = adaptContextForModule(transcriptContext, false);
		expect(result.systemPrompt).toBe("Base prompt");
		expect(result.tools).toHaveLength(1);
		expect(result.tools?.[0]?.name).toBe("bash");
		expect(result.messages).toHaveLength(1);
		expect(result.messages[0]).toEqual({ role: "user", content: "Run pwd", timestamp: 1000 });
	});
});
