import { Link } from "react-router-dom";
import { supabase } from "./supabase.js";
import { useEffect, useState } from "react";

const shellStyle = {
  background: "linear-gradient(135deg, #0a2a43, #1c4e80, #3a7ca5)",
};

export default function Profile() {
  const [user, setUser] = useState(null);
  const [editing, setEditing] = useState(false);
  const [enquiryNature, setEnquiryNature] = useState("");
  const [profession, setProfession] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const professionOptions = [
    "Student", "Researcher", "Aviation Operations", "Airport Management",
    "Airline Staff", "Security/Compliance", "Technology Partner", "Other",
  ];

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user: u } }) => {
      setUser(u);
      if (u?.user_metadata) {
        setEnquiryNature(u.user_metadata.enquiry_nature || "");
        setProfession(u.user_metadata.client_profession || "");
      }
    });
  }, []);

  async function handleSave() {
    setSaving(true);
    setSaved(false);
    try {
      const { data, error } = await supabase.auth.updateUser({
        data: {
          enquiry_nature: enquiryNature,
          client_profession: profession,
        },
      });
      if (!error && data?.user) {
        setUser(data.user);
        setSaved(true);
        setEditing(false);
        setTimeout(() => setSaved(false), 3000);
      }
    } catch {
      // silent fallback
    } finally {
      setSaving(false);
    }
  }

  const meta = user?.user_metadata || {};

  return (
    <div
      className="font-['DM_Sans'] text-white min-h-screen px-6 pt-32 pb-12"
      style={shellStyle}
    >
      <div className="max-w-lg mx-auto rounded-2xl bg-[#0a192f]/60 border border-white/10 p-8 backdrop-blur-sm">
        <h1 className="font-bebas text-4xl tracking-wide mb-6">Profile</h1>

        {/* Avatar + email */}
        <div className="flex items-center gap-4 mb-8">
          <div className="w-14 h-14 rounded-full bg-gradient-to-br from-[#FF6B35] to-[#7EC8E3] flex items-center justify-center text-xl font-bold">
            {user?.email?.[0]?.toUpperCase() || "?"}
          </div>
          <div>
            <p className="text-lg font-medium">{user?.email ?? "—"}</p>
            <p className="text-white/50 text-xs">
              Joined {user?.created_at ? new Date(user.created_at).toLocaleDateString() : "—"}
            </p>
          </div>
        </div>

        {saved && (
          <div className="mb-4 px-4 py-2 rounded-lg bg-[#7EC8E3]/10 border border-[#7EC8E3]/20 text-[#7EC8E3] text-sm">
            Profile updated successfully.
          </div>
        )}

        {/* Info fields */}
        <div className="space-y-4">
          <div className="rounded-lg bg-white/5 border border-white/10 p-4">
            <p className="text-xs uppercase tracking-wider text-[#7EC8E3] mb-1">Nature of Enquiry</p>
            {editing ? (
              <select
                value={enquiryNature}
                onChange={(e) => setEnquiryNature(e.target.value)}
                className="w-full rounded-lg bg-white/5 border border-white/15 px-3 py-2 text-white text-sm focus:outline-none focus:border-[#7EC8E3] transition-colors mt-1"
              >
                <option value="" className="text-black">Select one</option>
                <option value="General information" className="text-black">General information</option>
                <option value="Product demo request" className="text-black">Product demo request</option>
                <option value="Research collaboration" className="text-black">Research collaboration</option>
                <option value="Implementation/technical support" className="text-black">Implementation/technical support</option>
                <option value="Partnership enquiry" className="text-black">Partnership enquiry</option>
              </select>
            ) : (
              <p className="text-white/80">{meta.enquiry_nature || "Not specified"}</p>
            )}
          </div>

          <div className="rounded-lg bg-white/5 border border-white/10 p-4">
            <p className="text-xs uppercase tracking-wider text-[#7EC8E3] mb-1">Profession</p>
            {editing ? (
              <select
                value={profession}
                onChange={(e) => setProfession(e.target.value)}
                className="w-full rounded-lg bg-white/5 border border-white/15 px-3 py-2 text-white text-sm focus:outline-none focus:border-[#7EC8E3] transition-colors mt-1"
              >
                <option value="" className="text-black">Select profession</option>
                {professionOptions.map((o) => (
                  <option key={o} value={o} className="text-black">{o}</option>
                ))}
              </select>
            ) : (
              <p className="text-white/80">{meta.client_profession || "Not specified"}</p>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-3 mt-6">
          {editing ? (
            <>
              <button
                onClick={handleSave}
                disabled={saving}
                className="bg-[#FF6B35] text-white font-semibold px-6 py-2.5 rounded-xl hover:bg-[#FF6B35]/90 transition-all disabled:opacity-50 text-sm flex items-center gap-2"
              >
                {saving ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Saving...
                  </>
                ) : "Save Changes"}
              </button>
              <button
                onClick={() => {
                  setEditing(false);
                  setEnquiryNature(meta.enquiry_nature || "");
                  setProfession(meta.client_profession || "");
                }}
                className="bg-transparent border border-white/25 text-white font-semibold px-6 py-2.5 rounded-xl hover:border-white/50 transition-all text-sm"
              >
                Cancel
              </button>
            </>
          ) : (
            <button
              onClick={() => setEditing(true)}
              className="bg-transparent border border-white/25 text-white font-semibold px-6 py-2.5 rounded-xl hover:border-[#7EC8E3] hover:text-[#7EC8E3] transition-all text-sm"
            >
              Edit Profile
            </button>
          )}
        </div>

        <div className="flex flex-wrap gap-4 mt-8 pt-6 border-t border-white/10">
          <Link to="/" className="text-sm text-[#7EC8E3] hover:text-white transition-colors">← Back home</Link>
          <Link to="/logout" className="text-sm text-[#FF6B35] hover:text-white transition-colors">Logout</Link>
        </div>
      </div>
    </div>
  );
}
