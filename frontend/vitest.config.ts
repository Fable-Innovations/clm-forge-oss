import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Component tests (jsdom): `npm test`.
export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", include: ["src/**/*.test.tsx", "src/**/*.test.ts"] },
});
