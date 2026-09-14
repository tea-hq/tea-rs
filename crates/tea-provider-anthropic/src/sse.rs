//! Pure Server-Sent Events parser used by the Anthropic stream adapter.

use std::fmt;

/// One parsed SSE payload.
#[derive(Clone, PartialEq, Eq)]
pub enum SseEvent {
    /// A complete JSON payload from one or more `data:` lines.
    Data(String),
    /// A terminal `[DONE]` sentinel accepted for compatible gateways.
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

    /// Feeds a byte chunk and returns completed events.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.buffer.drain(..=newline).collect::<Vec<u8>>();
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(&String::from_utf8_lossy(&line), &mut events);
        }
        events
    }

    /// Flushes a trailing event at the end of the byte stream.
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
            if let Some(existing) = &mut self.pending {
                existing.push('\n');
                existing.push_str(value);
            } else {
                self.pending = Some(value.to_owned());
            }
        }
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
    fn parses_anthropic_event_fields_and_split_data() {
        let mut parser = SseParser::new();
        assert!(
            parser
                .feed(b"event: message_start\ndata: {\"type\"")
                .is_empty()
        );
        let events = parser.feed(b":\"message_start\"}\n\n");
        assert_eq!(
            events,
            vec![SseEvent::Data("{\"type\":\"message_start\"}".to_owned())]
        );
    }

    #[test]
    fn finish_flushes_an_unterminated_data_line() {
        let mut parser = SseParser::new();
        assert!(parser.feed(b"data: {\"type\":\"message_stop\"}").is_empty());

        assert_eq!(
            parser.finish(),
            vec![SseEvent::Data("{\"type\":\"message_stop\"}".to_owned())]
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
