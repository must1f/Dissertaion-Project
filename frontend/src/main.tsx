import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// Initialize theme from localStorage — default to dark
const theme = localStorage.getItem('pinn-app-storage')
let prefersDark = true // default

if (theme) {
  try {
    const parsed = JSON.parse(theme)
    if (parsed.state?.theme === 'light') {
      prefersDark = false
    } else if (parsed.state?.theme === 'system') {
      prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    }
  } catch (e) {
    // Ignore parsing errors
  }
}

if (prefersDark) {
  document.documentElement.classList.add('dark')
} else {
  document.documentElement.classList.remove('dark')
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
