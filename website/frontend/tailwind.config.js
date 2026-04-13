/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'bebas': ['Bebas Neue', 'cursive'],
        'dm-sans': ['DM Sans', 'sans-serif'],
      },
      transitionDuration: {
        '400': '400ms',
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.8s ease-out forwards',
        'scan-pulse': 'scanPulse 3s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'fade-float': 'fadeFloat 4s ease-in-out infinite',
        'sensor-pulse': 'sensorPulse 2s ease-in-out infinite',
        'ripple': 'ripple 2s ease-out infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': {
            opacity: '0',
            transform: 'translateY(30px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        scanPulse: {
          '0%, 100%': {
            opacity: '0.2',
            transform: 'scale(1)',
          },
          '50%': {
            opacity: '0.4',
            transform: 'scale(1.05)',
          },
        },
        float: {
          '0%, 100%': {
            transform: 'translateY(0px)',
          },
          '50%': {
            transform: 'translateY(-20px)',
          },
        },
        fadeFloat: {
          '0%, 100%': {
            opacity: '0.6',
            transform: 'translateY(0px)',
          },
          '50%': {
            opacity: '1',
            transform: 'translateY(-10px)',
          },
        },
        sensorPulse: {
          '0%, 100%': {
            opacity: '1',
            transform: 'scale(1)',
          },
          '50%': {
            opacity: '0.7',
            transform: 'scale(1.1)',
          },
        },
        ripple: {
          '0%': {
            opacity: '1',
            transform: 'scale(0.8)',
          },
          '100%': {
            opacity: '0',
            transform: 'scale(2)',
          },
        },
      },
    },
  },
  plugins: [],
}
