/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          blue:     '#2F80ED',
          teal:     '#2BBBAD',
          bg:       '#F7F9FC',
          text:     '#1F2937',
          muted:    '#6B7280',
          border:   '#E5E9F0',
          surface:  '#FFFFFF',
        }
      },
      fontFamily: {
        sans:    ['DM Sans', 'system-ui', 'sans-serif'],
        mono:    ['DM Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}