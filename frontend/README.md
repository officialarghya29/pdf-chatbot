# Unfold Frontend

React 18 + TypeScript + Vite + Tailwind CSS.

```bash
npm install
npm run dev      # http://localhost:5173 (proxies /api to backend :8000)
npm run build    # strict typecheck + production bundle to dist/
npm run preview  # serve the production build
```

Set `OPENAI_API_KEY` in `backend/.env` — the frontend needs no keys of its own.
