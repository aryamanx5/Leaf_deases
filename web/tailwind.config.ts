import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: 'var(--ink)',
        soil: 'var(--soil)',
        line: 'var(--line)',
        paper: 'var(--paper)',
        text: 'var(--text)',
        muted: 'var(--muted)',
        leaf: 'var(--leaf)',
        turmeric: { DEFAULT: 'var(--turmeric)', hover: 'var(--turmeric-hover)' },
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
