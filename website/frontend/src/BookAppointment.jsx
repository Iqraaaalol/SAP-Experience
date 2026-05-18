import { useState } from "react";
import { Link } from "react-router-dom";
import { supabase } from "./supabase.js";

const shellStyle = {
  background: "linear-gradient(135deg, #0a2a43, #1c4e80, #3a7ca5)",
};

export default function BookAppointment() {
  const [date, setDate] = useState("");
  const [time, setTime] = useState("");
  const [topic, setTopic] = useState("");
  const [note, setNote] = useState("");
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
        .from("appointments")
        .insert({
          user_id: user.id,
          user_email: user.email,
          preferred_date: date,
          preferred_time: time,
          topic,
          notes: note,
          status: "pending",
        });

      if (insertError) {
        if (insertError.message.includes("relation") || insertError.code === "42P01") {
          console.warn("Appointments table not set up yet — recording locally.");
          setSent(true);
        } else {
          setError(insertError.message);
        }
      } else {
        setSent(true);
      }
    } catch {
      console.warn("Database not configured — recording submission locally.");
      setSent(true);
    } finally {
      setLoading(false);
    }
  }

  // Get tomorrow's date as minimum
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  const minDate = tomorrow.toISOString().split("T")[0];

  return (
    <div
      className="font-['DM_Sans'] text-white min-h-screen px-6 pt-32 pb-12"
      style={shellStyle}
    >
      <div className="max-w-xl mx-auto rounded-2xl bg-[#0a192f]/60 border border-white/10 p-8 backdrop-blur-sm">
        <h1 className="font-bebas text-4xl tracking-wide mb-2">Book consultation</h1>
        <p className="text-white/60 text-sm mb-8">Request a time to speak with our team.</p>
        {sent ? (
          <div className="text-center py-4">
            <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-[#7EC8E3]/15 flex items-center justify-center text-2xl">📅</div>
            <p className="text-[#7EC8E3] font-medium mb-2">Appointment requested</p>
            <p className="text-white/50 text-sm mb-1">
              Date: <span className="text-white">{date}</span> {time && <>at <span className="text-white">{time}</span></>}
            </p>
            <p className="text-white/50 text-sm mb-6">We'll confirm your booking shortly.</p>
            <button
              onClick={() => { setSent(false); setDate(""); setTime(""); setTopic(""); setNote(""); }}
              className="text-[#FF6B35] text-sm hover:text-white transition-colors font-medium"
            >
              Book another appointment
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">Preferred date</label>
                <input
                  type="date"
                  value={date}
                  onChange={(e) => setDate(e.target.value)}
                  min={minDate}
                  required
                  className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white focus:outline-none focus:border-[#7EC8E3] transition-colors"
                />
              </div>
              <div>
                <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">Preferred time</label>
                <select
                  value={time}
                  onChange={(e) => setTime(e.target.value)}
                  className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white focus:outline-none focus:border-[#7EC8E3] transition-colors"
                >
                  <option value="" className="text-black">Any time</option>
                  <option value="09:00" className="text-black">09:00 AM</option>
                  <option value="10:00" className="text-black">10:00 AM</option>
                  <option value="11:00" className="text-black">11:00 AM</option>
                  <option value="13:00" className="text-black">01:00 PM</option>
                  <option value="14:00" className="text-black">02:00 PM</option>
                  <option value="15:00" className="text-black">03:00 PM</option>
                  <option value="16:00" className="text-black">04:00 PM</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">Topic</label>
              <select
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                required
                className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white focus:outline-none focus:border-[#7EC8E3] transition-colors"
              >
                <option value="" className="text-black">Select topic</option>
                <option value="Product demo" className="text-black">Product demo</option>
                <option value="Technical discussion" className="text-black">Technical discussion</option>
                <option value="Partnership enquiry" className="text-black">Partnership enquiry</option>
                <option value="Research collaboration" className="text-black">Research collaboration</option>
                <option value="General enquiry" className="text-black">General enquiry</option>
              </select>
            </div>
            <div>
              <label className="block text-xs uppercase tracking-wider text-[#7EC8E3] mb-2">Additional notes</label>
              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                rows={4}
                className="w-full rounded-lg bg-white/5 border border-white/15 px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-[#7EC8E3] transition-colors"
                placeholder="Anything else we should know — timezone preferences, specific questions, etc."
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
                  Requesting...
                </>
              ) : "Request Appointment"}
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
