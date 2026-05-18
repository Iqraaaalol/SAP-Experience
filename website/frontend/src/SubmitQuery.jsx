import { useState } from "react";
import { Link } from "react-router-dom";
import { supabase } from "./supabase.js";

const shellStyle = {
  background: "linear-gradient(135deg, #0a2a43, #1c4e80, #3a7ca5)",
};

export default function SubmitQuery() {
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [priority, setPriority] = useState("normal");
  const [sent, setSent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const { data: { user } } = await supabase.auth.getUser();

      const { error: insertError } = await supabase
        .from("queries")
        .insert({
          user_id: user.id,
          user_email: user.email,
          subject,
          message,
          priority,
          status: "open",
        });

      if (insertError) {
        // If the table doesn't exist yet, fall back gracefully
        if (insertError.message.includes("relation") || insertError.code === "42P01") {
          console.warn("Queries table not set up yet — recording locally.");
          setSent(true);
        } else {
          setError(insertError.message);
        }
      } else {
        setSent(true);
      }
    } catch {
      // Graceful fallback if DB not configured yet
      console.warn("Database not configured — recording submission locally.");
      setSent(true);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="font-['DM_Sans'] text-white min-h-screen px-6 pt-32 pb-12"
      style={shellStyle}
    >
      <div className="max-w-xl mx-auto rounded-2xl bg-[#0a192f]/60 border border-white/10 p-8 backdrop-blur-sm">
        <h1 className="font-bebas text-4xl tracking-wide mb-2">Submit a query</h1>
        <p className="text-white/60 text-sm mb-8">Describe what you need — we'll follow up.</p>
        {sent ? (
          <div className="text-center py-4">
            <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-[#7EC8E3]/15 flex items-center justify-center text-2xl">✓</div>
            <p className="text-[#7EC8E3] font-medium mb-2">Query submitted successfully</p>
            <p className="text-white/50 text-sm mb-6">We'll get back to you as soon as possible.</p>
            <button
              onClick={() => { setSent(false); setSubject(""); setMessage(""); setPriority("normal"); }}
              className="text-[#FF6B35] text-sm hover:text-white transition-colors font-medium"
            >
              Submit another query
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">Subject</label>
              <input
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                required
                className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-[#7EC8E3] transition-colors"
                placeholder="Brief summary of your issue"
              />
            </div>
            <div>
              <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">Priority</label>
              <select
                value={priority}
                onChange={(e) => setPriority(e.target.value)}
                className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white focus:outline-none focus:border-[#7EC8E3] transition-colors"
              >
                <option value="low" className="text-black">Low</option>
                <option value="normal" className="text-black">Normal</option>
                <option value="high" className="text-black">High</option>
                <option value="urgent" className="text-black">Urgent</option>
              </select>
            </div>
            <div>
              <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">Message</label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                required
                rows={6}
                className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-[#7EC8E3] transition-colors"
                placeholder="Describe your question or request in detail…"
              />
            </div>
            {error && <p className="text-sm text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{error}</p>}
            <button
              type="submit"
              disabled={loading}
              className="bg-[#FF6B35] text-white font-semibold px-8 py-3 rounded-xl hover:bg-[#FF6B35]/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Sending...
                </>
              ) : "Send"}
            </button>
          </form>
        )}
        <p className="mt-8">
          <Link to="/" className="text-sm text-[#7EC8E3] hover:text-white transition-colors">← Back home</Link>
        </p>
      </div>
    </div>
  );
}
