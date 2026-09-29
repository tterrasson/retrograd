import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'

// Model and dataset text is untrusted. Raw HTML is never interpreted, and what
// markdown produces is sanitized again before it reaches the page.
const md = new MarkdownIt({ html: false, linkify: true, breaks: true })

const defaultLinkOpen =
  md.renderer.rules.link_open ??
  ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options))
md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  const token = tokens[idx]
  token?.attrSet('target', '_blank')
  token?.attrSet('rel', 'noopener noreferrer nofollow')
  return defaultLinkOpen(tokens, idx, options, env, self)
}

export function renderMarkdown(text: string): string {
  return DOMPurify.sanitize(md.render(text), {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ['style', 'img', 'iframe', 'form', 'input'],
    ADD_ATTR: ['target', 'rel'],
  })
}
