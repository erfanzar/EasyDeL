// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Check, Copy } from "lucide-react";
import { Fragment, useState, type ReactNode } from "react";

/*
 * A deliberately tiny Markdown subset for assistant replies: paragraphs,
 * headings, bullet and numbered lists, `inline code`, **bold** and fenced code
 * blocks. Everything is emitted as React text nodes -- a reply is model output,
 * and model output is never trusted as HTML.
 */

export type MarkdownBlock =
  | { kind: "paragraph"; lines: string[] }
  | { kind: "heading"; level: number; text: string }
  | { kind: "code"; language: string; code: string }
  | { kind: "list"; ordered: boolean; items: string[] };

const FENCE = /^\s*(```|~~~)\s*([\w#+.-]*)\s*$/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const ORDERED = /^\s*\d{1,3}[.)]\s+(.*)$/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const INLINE = /(`[^`\n]+`|\*\*[^*\n]+\*\*)/;

const startsBlock = (line: string) => FENCE.test(line) || HEADING.test(line) || BULLET.test(line) || ORDERED.test(line);

export function parseMarkdown(text: string): MarkdownBlock[] {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks: MarkdownBlock[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    const fence = FENCE.exec(line);
    if (fence) {
      const body: string[] = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== fence[1]) {
        body.push(lines[index]);
        index += 1;
      }
      // Skip the closing fence; an unterminated block runs to the end of the reply.
      index += 1;
      blocks.push({ kind: "code", language: fence[2], code: body.join("\n") });
      continue;
    }
    if (!line.trim()) {
      index += 1;
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({ kind: "heading", level: heading[1].length, text: heading[2].trim() });
      index += 1;
      continue;
    }
    if (BULLET.test(line) || ORDERED.test(line)) {
      const ordered = !BULLET.test(line);
      const pattern = ordered ? ORDERED : BULLET;
      const items: string[] = [];
      while (index < lines.length) {
        const item = pattern.exec(lines[index]);
        if (item) {
          items.push(item[1]);
          index += 1;
          continue;
        }
        // An indented line continues the item above it.
        if (items.length && /^\s{2,}\S/.test(lines[index]) && !startsBlock(lines[index])) {
          items[items.length - 1] += ` ${lines[index].trim()}`;
          index += 1;
          continue;
        }
        break;
      }
      blocks.push({ kind: "list", ordered, items });
      continue;
    }
    const paragraph: string[] = [];
    while (index < lines.length && lines[index].trim() && !startsBlock(lines[index])) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push({ kind: "paragraph", lines: paragraph });
  }
  return blocks;
}

/** `code` and **bold** spans; every other character stays literal text. */
export function renderInline(text: string): ReactNode[] {
  // A split on a single capture group puts the matches at the odd indices.
  return text.split(INLINE).map((part, index) => {
    if (index % 2 === 0) return <Fragment key={index}>{part}</Fragment>;
    return part.startsWith("`") ? (
      <code key={index}>{part.slice(1, -1)}</code>
    ) : (
      <strong key={index}>{part.slice(2, -2)}</strong>
    );
  });
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="md-code">
      <div className="md-code-bar">
        <span>{language || "code"}</span>
        <button
          className="md-copy"
          type="button"
          aria-label={copied ? "Copied" : "Copy code"}
          title="Copy code"
          onClick={() => {
            void globalThis.navigator?.clipboard
              ?.writeText(code)
              .then(() => {
                setCopied(true);
                globalThis.setTimeout(() => setCopied(false), 1600);
              })
              .catch(() => undefined);
          }}
        >
          {copied ? <Check size={12} aria-hidden="true" /> : <Copy size={12} aria-hidden="true" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function AssistantMarkdown({ text }: { text: string }) {
  return (
    <div className="md">
      {parseMarkdown(text).map((block, index) => {
        switch (block.kind) {
          case "code":
            return <CodeBlock key={index} language={block.language} code={block.code} />;
          case "heading":
            return (
              <p key={index} className={`md-heading md-heading-${Math.min(block.level, 3)}`}>
                {renderInline(block.text)}
              </p>
            );
          case "list": {
            const items = block.items.map((item, itemIndex) => <li key={itemIndex}>{renderInline(item)}</li>);
            return block.ordered ? <ol key={index}>{items}</ol> : <ul key={index}>{items}</ul>;
          }
          default:
            return (
              <p key={index}>
                {block.lines.map((line, lineIndex) => (
                  <Fragment key={lineIndex}>
                    {lineIndex > 0 && <br />}
                    {renderInline(line)}
                  </Fragment>
                ))}
              </p>
            );
        }
      })}
    </div>
  );
}
