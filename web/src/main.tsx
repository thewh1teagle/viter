import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { applyTheme, useStore } from './store.ts'

// Apply the persisted theme (dark by default) before the first paint so the
// app never flashes light. `index.html` ships no theme class of its own.
applyTheme(useStore.getState().theme)
document.title = 'viter'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
