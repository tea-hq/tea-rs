//! Pure `Server-Sent-Events` parser: byte chunks -> completed `data:` payloads.
//!
//! `OpenAI` streams Chat Completions as `text/event-stream` with `data:` lines
//! terminated by blank lines, ending with `data: [DONE]`. This parser is
//! transport-agnostic so it is unit-tested with fixture byte chunks and reused
//! by the live `reqwest` stream.

use std::fmt;

/// One parsed SSE event.
#[derive(Clone, PartialEq, Eq)]
pub enum SseEvent {
    /// A `data:` payload (JSON text).
    Data(String),
    /// The terminal `data: [DONE]` sentinel.
    Done,
}

impl fmt::Debug for SseEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(payload) => formatter
                .debug_struct("Data")
                .field("payload_bytes", &payload.len())
                .finish_non_exhaustive(),
            Self::Done => formatter.write_str("Done"),
        }
    }
}

/// Incremental SSE parser fed arbitrary byte chunks.
#[derive(Default)]
pub struct SseParser {
    buffer: Vec<u8>,
    pending: Option<String>,
}

impl fmt::Debug for SseParser {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SseParser")
            .field("buffered_bytes", &self.buffer.len())
            .field("pending_bytes", &self.pending.as_ref().map(String::len))
            .finish()
    }
}

impl SseParser {
    /// Creates an empty parser.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds a byte chunk and returns any completed SSE events.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.buffer.drain(..=newline).collect::<Vec<u8>>();
            // Drop the trailing LF, then a trailing CR for CRLF streams.
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line_str = String::from_utf8_lossy(&line);
            self.process_line(&line_str, &mut events);
        }
        events
    }

    /// Flushes any trailing partial event without a final blank line.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let mut line = std::mem::take(&mut self.buffer);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(&String::from_utf8_lossy(&line), &mut events);
        }
        if let Some(data) = self.pending.take() {
            events.push(Self::classify(&data));
        }
        events
    }

    fn process_line(&mut self, line: &str, events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            if let Some(data) = self.pending.take() {
                events.push(Self::classify(&data));
            }
            return;
        }
        if line.starts_with(':') {
            return;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            let value = rest.strip_prefix(' ').unwrap_or(rest);
            match &self.pending {
                Some(existing) => {
                    let mut combined = existing.clone();
                    combined.push('\n');
                    combined.push_str(value);
                    self.pending = Some(combined);
                }
                None => self.pending = Some(value.to_owned()),
            }
        }
        // Other SSE fields (event:, id:, retry:) are ignored.
    }

    fn classify(data: &str) -> SseEvent {
        if data == "[DONE]" {
            SseEvent::Done
        } else {
            SseEvent::Data(data.to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPAQUE_SENTINEL: &str = "opaque-sse-payload-must-not-appear-in-debug";

    #[test]
    fn parses_data_lines_and_done() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: {\"a\":1}\n\ndata: [DONE]\n\n");
        assert_eq!(
            events,
            vec![SseEvent::Data("{\"a\":1}".to_owned()), SseEvent::Done]
        );
    }

    #[test]
    fn buffers_across_split_chunks() {
        let mut parser = SseParser::new();
        assert!(parser.feed(b"data: {\"a\"").is_empty());
        assert!(parser.feed(b":1}\n").is_empty());
        let events = parser.feed(b"\n");
        assert_eq!(events, vec![SseEvent::Data("{\"a\":1}".to_owned())]);
    }

    #[test]
    fn finish_flushes_an_unterminated_data_line() {
        let mut parser = SseParser::new();
        assert!(parser.feed(b"data: {\"a\":1}").is_empty());

        assert_eq!(
            parser.finish(),
            vec![SseEvent::Data("{\"a\":1}".to_owned())]
        );
    }

    #[test]
    fn finish_combines_pending_and_unterminated_data_lines() {
        let mut parser = SseParser::new();
        assert!(parser.feed(b"data: first\ndata: second").is_empty());

        assert_eq!(
            parser.finish(),
            vec![SseEvent::Data("first\nsecond".to_owned())]
        );
    }

    #[test]
    fn handles_crlf_line_endings() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: hi\r\n\r\n");
        assert_eq!(events, vec![SseEvent::Data("hi".to_owned())]);
    }

    #[test]
    fn ignores_comments_and_other_fields() {
        let mut parser = SseParser::new();
        let events = parser.feed(b": keepalive\n\nevent: ping\ndata: x\n\n");
        assert_eq!(events, vec![SseEvent::Data("x".to_owned())]);
    }

    #[test]
    fn sse_debug_redacts_provider_payloads() {
        let event = SseEvent::Data(OPAQUE_SENTINEL.to_owned());
        let mut parser = SseParser::new();
        let pending = format!("data: {OPAQUE_SENTINEL}\n");
        assert!(parser.feed(pending.as_bytes()).is_empty());
        assert!(parser.feed(OPAQUE_SENTINEL.as_bytes()).is_empty());

        let event_debug = format!("{event:?}");
        let parser_debug = format!("{parser:?}");
        assert!(!event_debug.contains(OPAQUE_SENTINEL), "{event_debug}");
        assert!(!parser_debug.contains(OPAQUE_SENTINEL), "{parser_debug}");
    }
}
