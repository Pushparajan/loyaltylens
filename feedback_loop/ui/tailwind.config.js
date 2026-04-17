/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          green: "#00704A",
          gold: "#CBA258",
          cream: "#F2F0EB",
        },
      },
    },
  },
  plugins: [],
};
