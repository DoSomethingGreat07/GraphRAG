import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#09111f",
        mist: "#f4efe4",
        ember: "#e86f2a",
        teal: "#1f8a8a",
        brass: "#b9975b",
        slate: "#2a3b52"
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        body: ["Space Grotesk", "system-ui", "sans-serif"]
      },
      boxShadow: {
        panel: "0 25px 80px rgba(9, 17, 31, 0.14)"
      },
      backgroundImage: {
        grain:
          "radial-gradient(circle at 20% 20%, rgba(232,111,42,0.14), transparent 35%), radial-gradient(circle at 80% 0%, rgba(31,138,138,0.2), transparent 30%), linear-gradient(135deg, rgba(9,17,31,0.96), rgba(26,42,63,0.92))"
      }
    }
  },
  plugins: []
} satisfies Config;
