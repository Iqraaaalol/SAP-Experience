import { useState } from "react";
import { Link } from "react-router-dom";
import { supabase } from "./supabase.js";

const shellStyle = {
  background: "linear-gradient(135deg, #0a2a43, #1c4e80, #3a7ca5)",
};

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [enquiryNature, setEnquiryNature] = useState("");
  const professionOptions = [
    "Student",
    "Researcher",
    "Aviation Operations",
    "Airport Management",
    "Airline Staff",
    "Security/Compliance",
    "Technology Partner",
    "Other",
  ];
  const [profession, setProfession] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  function handleResetSurvey() {
    setEnquiryNature("");
    setProfession("");
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const { error: err } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            enquiry_nature: enquiryNature,
            client_profession: profession,
          },
        },
      });
      if (err) setError(err.message);
      else setSuccess(true);
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
        <h1 className="font-bebas text-4xl tracking-wide mb-2">Create account</h1>
        <p className="text-white/60 text-sm mb-8">Join Vector 4.</p>

        {success ? (
          <div className="text-center py-6">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-[#7EC8E3]/15 flex items-center justify-center text-3xl">✓</div>
            <h2 className="text-xl font-semibold text-[#7EC8E3] mb-3">Account Created</h2>
            <p className="text-white/70 text-sm mb-6">
              Check your email at <span className="text-white font-medium">{email}</span> to verify your account before signing in.
            </p>
            <Link
              to="/login"
              className="inline-block bg-[#FF6B35] text-white font-semibold px-8 py-3 rounded-xl hover:bg-[#FF6B35]/90 transition-all"
            >
              Go to Login
            </Link>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">
                Nature of enquiry
              </label>
              <select
                value={enquiryNature}
                onChange={(e) => setEnquiryNature(e.target.value)}
                required
                className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white focus:outline-none focus:border-[#7EC8E3] transition-colors"
              >
                <option value="" className="text-black">Select one</option>
                <option value="General information" className="text-black">General information</option>
                <option value="Product demo request" className="text-black">Product demo request</option>
                <option value="Research collaboration" className="text-black">Research collaboration</option>
                <option value="Implementation/technical support" className="text-black">Implementation/technical support</option>
                <option value="Partnership enquiry" className="text-black">Partnership enquiry</option>
              </select>
            </div>

            <div>
              <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">
                Client profession
              </label>
              <select
                value={profession}
                onChange={(e) => setProfession(e.target.value)}
                required
                className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white focus:outline-none focus:border-[#7EC8E3] transition-colors"
              >
                <option value="" className="text-black">Select profession</option>
                {professionOptions.map((option) => (
                  <option key={option} value={option} className="text-black">{option}</option>
                ))}
              </select>
            </div>

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
                minLength={6}
                className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-[#7EC8E3] transition-colors"
                placeholder="Min 6 characters"
              />
            </div>
            {error && <p className="text-sm text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{error}</p>}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                type="button"
                onClick={handleResetSurvey}
                className="w-full bg-transparent border border-white/25 text-white font-semibold py-3 rounded-xl hover:border-[#FFB08A] hover:text-[#FFB08A] transition-all"
              >
                Reset survey
              </button>
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
                    Creating...
                  </>
                ) : "Sign up"}
              </button>
            </div>
          </form>
        )}

        {!success && (
          <>
            <p className="mt-6 text-center text-white/60 text-sm">
              Already have an account?{" "}
              <Link to="/login" className="text-[#7EC8E3] hover:text-white transition-colors">Sign in</Link>
            </p>
            <p className="mt-4 text-center">
              <Link to="/" className="text-sm text-white/50 hover:text-white transition-colors">← Back home</Link>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
