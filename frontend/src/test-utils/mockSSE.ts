/**
 * Mock SSE response helper for frontend tests.
 * Creates a mock ReadableStream that emits SSE-formatted events.
 */

export function mockSSEResponse(events: Array<Record<string, unknown>>): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder();
    return new ReadableStream({
        start(controller) {
            for (const event of events) {
                // SSE spec: event: <type>\ndata: <json>\n\n
                // The SSE parser in useRAGChatStream reads 'event:' lines FIRST
                // to determine currentEventType, then processes 'data:' lines.
                // Without 'event:', currentEventType stays empty/unknown and
                // handlers for 'sources', 'token_rollback', etc. never fire.
                if (event.type) {
                    controller.enqueue(encoder.encode(`event: ${event.type}\n`));
                }
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
            }
            controller.close();
        },
    });
}

/**
 * Create a mock SSE response string for testing.
 */
export function createSSEResponseString(events: Array<{ type: string; data: Record<string, unknown> }>): string {
    return events
        .map(({ type, data }) => `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`)
        .join('');
}

/**
 * Parse SSE response string into events array.
 */
export function parseSSEEvents(sseText: string): Array<Record<string, unknown>> {
    const events: Array<Record<string, unknown>> = [];
    const lines = sseText.split('\n');
    let currentEvent: Record<string, unknown> = {};
    let currentType = '';

    for (const line of lines) {
        if (line.startsWith('event: ')) {
            currentType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
            try {
                const data = JSON.parse(line.slice(6).trim());
                currentEvent = { type: currentType, ...data };
                events.push(currentEvent);
            } catch {
                // Ignore parse errors
            }
        }
    }

    return events;
}
