/* Component tests: stub fetch with a router over (method, URL, JSON body); every request is recorded. */
import { vi } from "vitest";

export interface Call {
  method: string;
  url: URL;
  body: unknown;
}
export type Route = (c: Call) => [number, unknown] | undefined;

export function stubFetch(route: Route): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal("fetch", async (input: string, init?: RequestInit) => {
    const c: Call = { method: (init?.method ?? "GET").toUpperCase(), url: new URL(input, "http://localhost"), body: init?.body ? JSON.parse(String(init.body)) : undefined };
    // the Forge session bootstrap (api/client.ts) is answered here, not recorded
    if (c.url.pathname === "/api/session") return new Response(JSON.stringify({ token: "test-session", header: "X-Forge-Session" }), { status: 200, headers: { "Content-Type": "application/json" } });
    calls.push(c);
    const [status, data] = route(c) ?? [404, { detail: `no stub for ${c.method} ${c.url.pathname}` }];
    return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
  });
  return calls;
}
