import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { supabase } from "./supabase.js";

const shellStyle = {
  background: "linear-gradient(135deg, #0a2a43, #1c4e80, #3a7ca5)",
};

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const { error: err } = await supabase.auth.signInWithPassword({ email, password });
      if (err) setError(err.message);
      else navigate("/", { replace: true });
    } catch {
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="font-['DM_Sans'] text-white min-h-screen flex items-center justify-center px-6 pt-24 pb-12"
      style={shellStyle}
    >
      <div className="w-full max-w-md rounded-2xl bg-[#0a192f]/60 border border-white/10 p-8 backdrop-blur-sm">
        <h1 className="font-bebas text-4xl tracking-wide mb-2">Sign in</h1>
        <p className="text-white/60 text-sm mb-8">Use your Vector 4 account.</p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-[#7EC8E3] transition-colors"
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-[#7EC8E3] transition-colors"
            />
          </div>
          {error && <p className="text-sm text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-[#FF6B35] text-white font-semibold py-3 rounded-xl hover:bg-[#FF6B35]/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Signing in...
              </>
            ) : "Login"}
          </button>
        </form>
        <p className="mt-6 text-center text-white/60 text-sm">
          No account?{" "}
          <Link to="/signup" className="text-[#7EC8E3] hover:text-white transition-colors">
            Sign up
          </Link>
        </p>
        <p className="mt-4 text-center">
          <Link to="/" className="text-sm text-white/50 hover:text-white transition-colors">
            ← Back home
          </Link>
        </p>
      </div>
    </div>
  );
}
