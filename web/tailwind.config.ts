import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: 'rgb(var(--ink-rgb) / <alpha-value>)',
        soil: 'rgb(var(--soil-rgb) / <alpha-value>)',
        line: 'var(--line)',
        paper: 'rgb(var(--paper-rgb) / <alpha-value>)',
        text: 'var(--text)',
        muted: 'var(--muted)',
        leaf: 'rgb(var(--leaf-rgb) / <alpha-value>)',
        turmeric: { DEFAULT: 'rgb(var(--turmeric-rgb) / <alpha-value>)', hover: 'var(--turmeric-hover)' },
        chlorosis: 'var(--chlorosis)',
        necrosis: 'var(--necrosis)',
        'dark-tissue': 'var(--dark-tissue)',
        mildew: 'var(--mildew)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        playfair: ['"Playfair Display"', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      maxWidth: { grid: '1360px' },
      transitionTimingFunction: { out: 'cubic-bezier(0.16,1,0.3,1)' },
    },
  },
  plugins: [],
} satisfies Config;
