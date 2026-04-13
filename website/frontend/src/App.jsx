import { Routes, Route, Link, useLocation } from "react-router-dom";
import { useEffect, useState, useRef } from "react";
import Login from "./Login.jsx";
import Signup from "./Signup.jsx";
import SubmitQuery from "./SubmitQuery.jsx";
import BookAppointment from "./BookAppointment.jsx";
import Logout from "./Logout.jsx";
import ProtectedRoute from "./ProtectedRoute.jsx";
import Profile from "./Profile.jsx";
import { supabase } from "./supabase.js";
import "./App.css";

const shellStyle = {
  background: "linear-gradient(135deg, #0a2a43, #1c4e80, #3a7ca5)",
};

/* ───────── scroll-reveal hook ───────── */
function useReveal() {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setVisible(true); obs.unobserve(el); } },
      { threshold: 0.12 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return [ref, visible];
}

function RevealSection({ children, className = "", delay = 0 }) {
  const [ref, visible] = useReveal();
  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

/* ───────── Navbar ───────── */
function Navbar() {
  const [user, setUser] = useState(null);
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setUser(session?.user ?? null));
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_e, s) => setUser(s?.user ?? null));
    return () => subscription.unsubscribe();
  }, []);

  useEffect(() => { setOpen(false); }, [location]);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <nav className={`fixed top-0 inset-x-0 z-50 px-[5%] py-3 flex items-center justify-between transition-all duration-300 ${scrolled ? "bg-[#0a192f]/95 backdrop-blur-md shadow-lg border-b border-white/10" : "bg-transparent"}`}>
      <Link to="/" className="flex items-center gap-3 no-underline">
        <img
          src="/logo.png"
          alt=""
          width={40}
          height={40}
          className="h-10 w-10 object-contain rounded-full shrink-0"
          decoding="async"
        />
        <span className="font-bebas text-xl tracking-widest text-white">VECTOR <span className="text-[#FF6B35]">4</span></span>
      </Link>

      {/* desktop links */}
      <div className="hidden md:flex items-center gap-6 text-sm font-medium">
        <Link to="/" className="text-white/80 hover:text-[#7EC8E3] transition-colors">Home</Link>
        <Link to="/about" className="text-white/80 hover:text-[#7EC8E3] transition-colors">About</Link>
        <Link to="/why-choose-us" className="text-white/80 hover:text-[#7EC8E3] transition-colors">Why Us</Link>
        <Link to="/portfolio" className="text-white/80 hover:text-[#7EC8E3] transition-colors">Portfolio</Link>
        <Link to="/support" className="text-white/80 hover:text-[#7EC8E3] transition-colors">Support</Link>
        {user ? (
          <>
            <Link to="/profile" className="text-white/80 hover:text-[#7EC8E3] transition-colors">Profile</Link>
            <Link to="/logout" className="bg-[#FF6B35] text-white px-5 py-2 rounded-lg font-semibold hover:bg-[#FF6B35]/85 transition-all">Logout</Link>
          </>
        ) : (
          <>
            <Link to="/login" className="text-white/80 hover:text-[#7EC8E3] transition-colors">Login</Link>
            <Link to="/signup" className="bg-[#FF6B35] text-white px-5 py-2 rounded-lg font-semibold hover:bg-[#FF6B35]/85 transition-all">Sign Up</Link>
          </>
        )}
      </div>

      {/* mobile hamburger */}
      <button className="md:hidden text-white text-2xl" onClick={() => setOpen(!open)} aria-label="Toggle menu">
        {open ? "✕" : "☰"}
      </button>

      {/* mobile drawer */}
      {open && (
        <div className="absolute top-full left-0 right-0 bg-[#0a192f]/98 backdrop-blur-md border-b border-white/10 flex flex-col gap-3 p-6 md:hidden text-sm font-medium">
          <Link to="/" className="text-white/80 hover:text-[#7EC8E3] py-1">Home</Link>
          <Link to="/about" className="text-white/80 hover:text-[#7EC8E3] py-1">About</Link>
          <Link to="/why-choose-us" className="text-white/80 hover:text-[#7EC8E3] py-1">Why Us</Link>
          <Link to="/portfolio" className="text-white/80 hover:text-[#7EC8E3] py-1">Portfolio</Link>
          <Link to="/support" className="text-white/80 hover:text-[#7EC8E3] py-1">Support</Link>
          {user ? (
            <>
              <Link to="/profile" className="text-white/80 hover:text-[#7EC8E3] py-1">Profile</Link>
              <Link to="/logout" className="text-[#FF6B35] py-1">Logout</Link>
            </>
          ) : (
            <>
              <Link to="/login" className="text-white/80 hover:text-[#7EC8E3] py-1">Login</Link>
              <Link to="/signup" className="text-[#FF6B35] py-1">Sign Up</Link>
            </>
          )}
        </div>
      )}
    </nav>
  );
}

/* ───────── App shell ───────── */
export default function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/why-choose-us" element={<WhyChooseUsPage />} />
        <Route path="/portfolio" element={<PortfolioPage />} />
        <Route path="/support" element={<SupportPage />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route
          path="/submitquery"
          element={<ProtectedRoute><SubmitQuery /></ProtectedRoute>}
        />
        <Route
          path="/bookappointment"
          element={<ProtectedRoute><BookAppointment /></ProtectedRoute>}
        />
        <Route
          path="/profile"
          element={<ProtectedRoute><Profile /></ProtectedRoute>}
        />
        <Route path="/logout" element={<Logout />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </>
  );
}

/* ───────── 404 ───────── */
function NotFoundPage() {
  return (
    <div className="font-['DM_Sans'] text-white min-h-screen flex flex-col items-center justify-center px-6" style={shellStyle}>
      <h1 className="font-bebas text-8xl tracking-wide text-[#FF6B35] mb-4">404</h1>
      <p className="text-white/70 text-lg mb-8">The page you're looking for doesn't exist.</p>
      <Link to="/" className="bg-[#FF6B35] text-white px-8 py-3 rounded-xl font-semibold hover:bg-[#FF6B35]/90 transition-all">
        Back to Home
      </Link>
    </div>
  );
}

/* ═══════════════════════ HOME PAGE ═══════════════════════ */
function HomePage() {
  return (
    <div className="font-['DM_Sans'] text-white min-h-screen" style={shellStyle}>
      <Hero />
      <StatsSection />
      <FeaturesSection />
      <ProcessSection />
      <PartnersSection />
      <SafetySection />
      <TestimonialSection />
      <ImportantPagesSection />
      <CTASection />
      <Footer />
    </div>
  );
}

function Hero() {
  return (
    <section className="min-h-[90vh] flex items-center px-[5%] pt-24 pb-10 relative overflow-hidden">
      {/* grid overlay */}
      <div className="absolute inset-0 pointer-events-none" style={{
        backgroundImage: "linear-gradient(rgba(126,200,227,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(126,200,227,0.04) 1px, transparent 1px)",
        backgroundSize: "60px 60px"
      }} />
      {/* radial glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[140%] h-[60%] pointer-events-none" style={{
        background: "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(126,200,227,0.12) 0%, transparent 60%)"
      }} />

      <div className="max-w-3xl relative z-10">
        <RevealSection>
          <span className="inline-flex items-center gap-2 text-xs font-semibold text-[#7EC8E3] tracking-widest uppercase mb-6">
            <span className="w-2 h-2 bg-[#7EC8E3] rounded-full animate-pulse" />
            Next-Gen Aviation Intelligence
          </span>
        </RevealSection>
        <RevealSection delay={100}>
          <h1 className="font-bebas text-5xl sm:text-6xl md:text-8xl leading-[0.95] tracking-wide mb-6">
            REDEFINING <span className="text-[#FF6B35]">PASSENGER</span> EXPERIENCE WITH{" "}
            <span className="text-[#7EC8E3]">AI</span>
          </h1>
        </RevealSection>
        <RevealSection delay={200}>
          <p className="text-lg sm:text-xl text-white/70 max-w-3xl mb-10 leading-relaxed">
            Advanced computer vision and multi-sensor AI that transforms facial expressions, body
            language, audio patterns, and thermal data into actionable insights for unparalleled
            customer experience and security.
          </p>
        </RevealSection>
        <RevealSection delay={300}>
          <div className="flex flex-wrap gap-4">
            <Link
              to="/signup"
              className="bg-[#FF6B35] text-white px-8 py-4 font-bold tracking-wide rounded-xl hover:bg-[#FF6B35]/90 hover:-translate-y-0.5 transition-all shadow-lg shadow-[#FF6B35]/20"
            >
              Get Started →
            </Link>
            <Link
              to="/portfolio"
              className="bg-transparent text-white px-8 py-4 font-bold tracking-wide rounded-xl border border-white/30 hover:border-[#7EC8E3] hover:text-[#7EC8E3] hover:-translate-y-0.5 transition-all"
            >
              Explore Portfolio
            </Link>
          </div>
        </RevealSection>
      </div>
    </section>
  );
}

function StatsSection() {
  const stats = [
    { label: "Detection Accuracy", value: "98.7%" },
    { label: "Passengers Analyzed", value: "50M+" },
    { label: "Aircraft Deployed", value: "200+" },
    { label: "Response Time", value: "< 50ms" },
  ];

  return (
    <section className="px-[5%] py-8 border-y border-white/10 bg-[#0a192f]/30">
      <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((item, i) => (
          <RevealSection key={item.label} delay={i * 80}>
            <div className="rounded-xl bg-[#0a192f]/55 border border-white/10 p-5 text-center hover:border-[#7EC8E3]/30 transition-colors">
              <p className="text-2xl sm:text-3xl font-bold text-[#7EC8E3]">{item.value}</p>
              <p className="text-xs sm:text-sm text-white/70 mt-1">{item.label}</p>
            </div>
          </RevealSection>
        ))}
      </div>
    </section>
  );
}

function FeaturesSection() {
  const features = [
    {
      icon: "👤",
      title: "Facial Expression Analysis",
      text: "Real-time recognition of micro-expressions and emotional states using deep learning models trained on diverse facial datasets.",
    },
    {
      icon: "🧍",
      title: "Body Language Decoding",
      text: "Comprehensive posture and gesture analysis that identifies passenger comfort levels, potential distress, and behavioral anomalies.",
    },
    {
      icon: "🎤",
      title: "Audio Intelligence",
      text: "Advanced acoustic processing that detects vocal stress patterns, anomalous sounds, and ambient audio signatures for enhanced situational awareness.",
    },
    {
      icon: "🌡️",
      title: "Thermal Sensing",
      text: "Infrared thermal imaging that monitors passenger health indicators, identifies elevated temperatures, and supports contactless wellness screening.",
    },
  ];

  return (
    <section className="px-[5%] py-16">
      <div className="max-w-7xl mx-auto">
        <RevealSection>
          <span className="text-xs font-semibold text-[#FF6B35] tracking-widest uppercase block mb-3">Core Capabilities</span>
          <h2 className="font-bebas text-4xl sm:text-5xl tracking-wide mb-4">
            Multi-Sensor <span className="text-[#FFB08A]">Intelligence</span>
          </h2>
          <p className="text-white/70 max-w-3xl mb-10">
            Four fusion layers working in concert to deliver a unified intelligence picture of the cabin environment.
          </p>
        </RevealSection>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, i) => (
            <RevealSection key={feature.title} delay={i * 100} className="h-full">
              <article className="h-full group rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 hover:border-[#7EC8E3]/50 transition-all hover:-translate-y-1 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-[#FF6B35] to-[#7EC8E3] scale-x-0 group-hover:scale-x-100 transition-transform origin-left" />
                <span className="text-3xl block mb-4">{feature.icon}</span>
                <h3 className="text-lg font-semibold mb-3 text-[#7EC8E3]">{feature.title}</h3>
                <p className="text-white/70 text-sm leading-relaxed">{feature.text}</p>
              </article>
            </RevealSection>
          ))}
        </div>
      </div>
    </section>
  );
}

function ProcessSection() {
  const steps = [
    { title: "Capture", desc: "Collect facial, movement, thermal, and audio signals at cabin touchpoints via embedded sensor arrays." },
    { title: "Process", desc: "Fuse the multi-modal signal stack through AI inference layers for real-time classification and scoring." },
    { title: "Act", desc: "Generate role-based insights and route prioritized alerts to cabin crew, ground teams, and operations." },
  ];

  return (
    <section className="px-[5%] py-16 bg-[#0a192f]/35 border-y border-white/10">
      <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-10 items-start">
        <RevealSection>
          <span className="text-xs font-semibold text-[#FF6B35] tracking-widest uppercase block mb-3">Pipeline</span>
          <h2 className="font-bebas text-4xl sm:text-5xl tracking-wide mb-4">
            How It <span className="text-[#FFB08A]">Works</span>
          </h2>
          <p className="text-white/70">
            A simple, reliable data-to-decision pipeline designed for the constraints of in-flight environments — low latency, high reliability, no connectivity dependency.
          </p>
        </RevealSection>
        <ol className="space-y-4">
          {steps.map((step, index) => (
            <RevealSection key={step.title} delay={index * 120}>
              <li className="rounded-xl bg-[#0b2135]/75 border border-white/10 px-5 py-4 hover:border-[#7EC8E3]/30 transition-colors">
                <span className="text-[#FF6B35] font-bold mr-2">0{index + 1}.</span>
                <span className="text-[#7EC8E3] font-semibold">{step.title}</span>
                <span className="text-white/80 ml-1">— {step.desc}</span>
              </li>
            </RevealSection>
          ))}
        </ol>
      </div>
    </section>
  );
}

function PartnersSection() {
  const partners = [
    { icon: "✈️", title: "Airline Operators", text: "Partnering with leading carriers to pilot and scale AI-driven passenger experience and safety programs across global routes." },
    { icon: "🛰️", title: "Avionics & OEMs", text: "Working with aircraft manufacturers and avionics providers to embed intelligence at the hardware and systems level." },
    { icon: "🔐", title: "Security & Compliance", text: "Aligning with regulatory bodies and security partners to ensure responsible, compliant, and auditable AI deployments." },
  ];

  return (
    <section className="px-[5%] py-16">
      <div className="max-w-7xl mx-auto">
        <RevealSection>
          <span className="text-xs font-semibold text-[#FF6B35] tracking-widest uppercase block mb-3">Partnerships</span>
          <h2 className="font-bebas text-4xl sm:text-5xl tracking-wide mb-4 text-center">
            Built With <span className="text-[#FFB08A]">Industry Leaders</span>
          </h2>
          <p className="text-white/70 max-w-2xl mx-auto text-center mb-10">
            We collaborate with aviation, security, and technology partners to ensure our solutions meet the highest operational standards.
          </p>
        </RevealSection>
        <div className="grid sm:grid-cols-3 gap-6">
          {partners.map((p, i) => (
            <RevealSection key={p.title} delay={i * 100}>
              <article className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 hover:border-[#7EC8E3]/40 transition-all hover:-translate-y-1">
                <span className="text-3xl block mb-4">{p.icon}</span>
                <h3 className="text-lg font-semibold mb-2 text-[#7EC8E3]">{p.title}</h3>
                <p className="text-white/70 text-sm leading-relaxed">{p.text}</p>
              </article>
            </RevealSection>
          ))}
        </div>
      </div>
    </section>
  );
}

function SafetySection() {
  const items = [
    { icon: "🔒", title: "Data Privacy by Design", text: "Anonymization, on-device processing options, and strict access controls to protect passenger identity and data." },
    { icon: "🧾", title: "Regulatory Alignment", text: "Architected to support compliance with aviation, data protection, and security regulations across multiple jurisdictions." },
    { icon: "🛡️", title: "Secure Infrastructure", text: "End-to-end encryption, hardened deployment environments, and continuous monitoring to safeguard critical systems." },
  ];

  return (
    <section className="px-[5%] py-16 bg-[#0a192f]/35 border-y border-white/10">
      <div className="max-w-7xl mx-auto">
        <RevealSection>
          <span className="text-xs font-semibold text-[#FF6B35] tracking-widest uppercase block mb-3">Safety & Confidentiality</span>
          <h2 className="font-bebas text-4xl sm:text-5xl tracking-wide mb-4 text-center">
            Designed for <span className="text-[#FFB08A]">Trust</span>
          </h2>
          <p className="text-white/70 max-w-2xl mx-auto text-center mb-10">
            Every layer of the platform is engineered with privacy, security, and regulatory alignment at its core.
          </p>
        </RevealSection>
        <div className="grid sm:grid-cols-3 gap-6">
          {items.map((s, i) => (
            <RevealSection key={s.title} delay={i * 100}>
              <article className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 text-center hover:border-[#7EC8E3]/40 transition-all hover:-translate-y-1">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-[#FF6B35]/15 to-[#7EC8E3]/15 flex items-center justify-center text-2xl">{s.icon}</div>
                <h3 className="text-lg font-semibold mb-2 text-white">{s.title}</h3>
                <p className="text-white/65 text-sm leading-relaxed">{s.text}</p>
              </article>
            </RevealSection>
          ))}
        </div>
      </div>
    </section>
  );
}

function TestimonialSection() {
  return (
    <section className="px-[5%] py-16">
      <div className="max-w-7xl mx-auto">
        <RevealSection>
          <h2 className="font-bebas text-4xl sm:text-5xl tracking-wide mb-2">
            Projected <span className="text-[#FFB08A]">Use-Case Scenarios</span>
          </h2>
          <p className="text-white/50 text-sm mb-8">These represent anticipated outcomes based on simulation data and project research.</p>
        </RevealSection>
        <div className="grid md:grid-cols-2 gap-6">
          <RevealSection delay={0}>
            <blockquote className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-8">
              <p className="text-white/80 leading-relaxed italic">
                "Vector 4 helped us identify high-friction service moments earlier and improve
                passenger satisfaction in under one quarter."
              </p>
              <footer className="mt-4 text-[#7EC8E3] text-sm">— Projected: Operations Lead, Regional Airport</footer>
            </blockquote>
          </RevealSection>
          <RevealSection delay={100}>
            <blockquote className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-8">
              <p className="text-white/80 leading-relaxed italic">
                "The multi-sensor architecture gave us stronger confidence in interventions without
                disrupting normal operations."
              </p>
              <footer className="mt-4 text-[#7EC8E3] text-sm">— Projected: Safety Program Manager</footer>
            </blockquote>
          </RevealSection>
        </div>
      </div>
    </section>
  );
}

function ImportantPagesSection() {
  const pages = [
    { title: "About Us", to: "/about", text: "Meet the team, mission, and the vision behind this project." },
    { title: "Why Choose Us", to: "/why-choose-us", text: "What sets the platform apart from conventional analytics." },
    { title: "Our Portfolio", to: "/portfolio", text: "Demonstrations, implementation snapshots, and AI capability previews." },
    { title: "Customer Support", to: "/support", text: "FAQs, direct support actions, and consultation booking." },
  ];

  return (
    <section className="px-[5%] py-16 bg-[#0a192f]/35 border-y border-white/10">
      <div className="max-w-7xl mx-auto">
        <RevealSection>
          <h2 className="font-bebas text-4xl sm:text-5xl tracking-wide mb-8">
            Explore <span className="text-[#FFB08A]">More</span>
          </h2>
        </RevealSection>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {pages.map((page, i) => (
            <RevealSection key={page.title} delay={i * 80}>
              <Link
                to={page.to}
                className="block rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 hover:border-[#7EC8E3]/60 hover:-translate-y-1 transition-all"
              >
                <h3 className="text-xl font-semibold text-[#7EC8E3] mb-3">{page.title}</h3>
                <p className="text-white/70 text-sm leading-relaxed">{page.text}</p>
              </Link>
            </RevealSection>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className="px-[5%] py-20 relative text-center">
      <div className="absolute inset-0 pointer-events-none" style={{
        background: "radial-gradient(ellipse at 30% 50%, rgba(255,107,53,0.1) 0%, transparent 50%), radial-gradient(ellipse at 70% 50%, rgba(126,200,227,0.08) 0%, transparent 50%)"
      }} />
      <RevealSection>
        <div className="max-w-2xl mx-auto relative z-10">
          <h2 className="font-bebas text-4xl sm:text-5xl tracking-wide mb-4">
            Ready to Redefine Your Passenger Experience?
          </h2>
          <p className="text-white/70 text-lg mb-8">
            Discover how Vector 4 can transform cabins into intelligent, responsive environments.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link to="/bookappointment" className="bg-[#FF6B35] text-white px-8 py-4 font-bold rounded-xl hover:bg-[#FF6B35]/90 hover:-translate-y-0.5 transition-all shadow-lg shadow-[#FF6B35]/20">
              Request a Demo
            </Link>
            <Link to="/about" className="bg-transparent text-white px-8 py-4 font-bold rounded-xl border border-white/30 hover:border-[#7EC8E3] hover:text-[#7EC8E3] transition-all">
              View Capabilities
            </Link>
          </div>
        </div>
      </RevealSection>
    </section>
  );
}

function Footer() {
  return (
    <footer className="px-[5%] py-10 border-t border-white/10 bg-[#081523]/55">
      <div className="max-w-7xl mx-auto">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <img
                src="/logo.png"
                alt=""
                width={32}
                height={32}
                className="h-8 w-8 object-contain rounded-full shrink-0"
                decoding="async"
              />
              <span className="font-bebas text-lg tracking-widest">VECTOR <span className="text-[#FF6B35]">4</span></span>
            </div>
            <p className="text-white/50 text-sm leading-relaxed">
              AI-powered aviation intelligence for passenger experience, safety, and operational awareness.
            </p>
          </div>
          <div>
            <h4 className="font-bebas text-base tracking-wider mb-3">Solutions</h4>
            <div className="flex flex-col gap-2 text-sm text-white/60">
              <Link to="/about" className="hover:text-white transition-colors">Cabin Intelligence</Link>
              <Link to="/why-choose-us" className="hover:text-white transition-colors">Decision Support</Link>
              <Link to="/portfolio" className="hover:text-white transition-colors">Integrations</Link>
            </div>
          </div>
          <div>
            <h4 className="font-bebas text-base tracking-wider mb-3">Company</h4>
            <div className="flex flex-col gap-2 text-sm text-white/60">
              <Link to="/about" className="hover:text-white transition-colors">About</Link>
              <Link to="/support" className="hover:text-white transition-colors">Support</Link>
              <Link to="/submitquery" className="hover:text-white transition-colors">Contact</Link>
            </div>
          </div>
          <div>
            <h4 className="font-bebas text-base tracking-wider mb-3">Account</h4>
            <div className="flex flex-col gap-2 text-sm text-white/60">
              <Link to="/login" className="hover:text-white transition-colors">Login</Link>
              <Link to="/signup" className="hover:text-white transition-colors">Sign Up</Link>
              <Link to="/profile" className="hover:text-white transition-colors">Profile</Link>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-4 pt-6 border-t border-white/10">
          <p className="text-white/40 text-xs">© 2026 Vector 4 Solutions. All rights reserved.</p>
          <p className="text-white/40 text-xs">Final Year Project — CSIT321</p>
        </div>
      </div>
    </footer>
  );
}

/* ═══════════════════════ INTERNAL PAGES ═══════════════════════ */
function InternalPageShell({ title, subtitle, children }) {
  return (
    <div className="font-['DM_Sans'] text-white min-h-screen" style={shellStyle}>
      <main className="px-[5%] pt-28 pb-14">
        <div className="max-w-6xl mx-auto">
          <RevealSection>
            <h1 className="font-bebas text-5xl sm:text-6xl tracking-wide mb-3 text-white/95">{title}</h1>
            <p className="text-white/70 mb-10 max-w-3xl">{subtitle}</p>
          </RevealSection>
          {children}
          <div className="mt-10">
            <Link
              to="/"
              className="inline-block bg-transparent text-white px-6 py-3 font-semibold rounded-xl border border-white/30 hover:border-[#7EC8E3] hover:text-[#7EC8E3] transition-all"
            >
              ← Back to Home
            </Link>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

function AboutPage() {
  return (
    <InternalPageShell
      title="About Us"
      subtitle="Vector 4 is a final year project focused on improving passenger experience through applied AI and multimodal analytics — using computer vision and microphones to track and control lighting, temperature, and create alerts for on-flight crew."
    >
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <RevealSection>
          <div className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 h-full">
            <h2 className="text-2xl text-[#7EC8E3] mb-3">Our Mission</h2>
            <p className="text-white/75 leading-relaxed">
              Build a practical, transparent AI system that helps cabin crew understand passenger sentiment and respond faster to emerging issues. We combine facial expression analysis, body language decoding, audio intelligence, and thermal sensing into a single actionable platform.
            </p>
          </div>
        </RevealSection>
        <RevealSection delay={100}>
          <div className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 h-full">
            <h2 className="text-2xl text-[#7EC8E3] mb-3">Our Team</h2>
            <p className="text-white/75 leading-relaxed">
              A student-led group combining skills in computer vision, full-stack development, and human-centered design. We are building Vector 4 as our CSIT321 final year project, targeting real-world deployment scenarios in commercial aviation.
            </p>
          </div>
        </RevealSection>
      </div>
      <RevealSection delay={200}>
        <div className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6">
          <h2 className="text-2xl text-[#7EC8E3] mb-3">The Technology</h2>
          <p className="text-white/75 leading-relaxed mb-4">
            Our platform integrates four sensor modalities into a unified intelligence layer. Cameras capture facial micro-expressions and body posture. Microphone arrays detect vocal stress patterns and ambient anomalies. Thermal sensors monitor passenger health indicators. All streams are fused through real-time AI inference and delivered as prioritized alerts and dashboards for cabin crew and ground operations.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
            {[
              { icon: "👤", label: "Facial Analysis" },
              { icon: "🧍", label: "Body Language" },
              { icon: "🎤", label: "Audio Processing" },
              { icon: "🌡️", label: "Thermal Sensing" },
            ].map((m) => (
              <div key={m.label} className="rounded-lg bg-white/5 border border-white/10 p-4 text-center">
                <span className="text-2xl block mb-2">{m.icon}</span>
                <span className="text-sm text-[#7EC8E3]">{m.label}</span>
              </div>
            ))}
          </div>
        </div>
      </RevealSection>
    </InternalPageShell>
  );
}

function WhyChooseUsPage() {
  const reasons = [
    { num: "01", title: "End-to-End Cabin Intelligence", text: "Fuse visual, audio, and thermal signals into a unified intelligence layer that continuously monitors cabin sentiment and safety." },
    { num: "02", title: "Real-Time Decision Support", text: "Deliver live alerts and recommendations to crew and ground teams, enabling proactive interventions instead of reactive responses." },
    { num: "03", title: "Seamless Integration", text: "Integrate with existing aircraft systems, airline CRMs, and security workflows through secure APIs and modular deployment options." },
    { num: "04", title: "Scalable Across Fleets", text: "Deploy across narrow-body, wide-body, and regional fleets with flexible hardware and cloud or edge-based processing." },
    { num: "05", title: "Explainable AI Outputs", text: "Every alert and recommendation comes with reasoning context so that crew can make informed decisions, not blindly follow automated instructions." },
  ];

  return (
    <InternalPageShell
      title="Why Choose Us"
      subtitle="Our platform is engineered to be actionable, reliable, and aligned with real service operations — from enhancing passenger comfort to strengthening security protocols."
    >
      <div className="space-y-4">
        {reasons.map((reason, i) => (
          <RevealSection key={reason.num} delay={i * 80}>
            <div className="rounded-xl bg-[#0a192f]/60 border border-white/10 px-5 py-5 flex gap-4 items-start hover:border-[#7EC8E3]/30 transition-colors group">
              <span className="text-2xl font-bebas text-[#FF6B35] opacity-70 group-hover:opacity-100 transition-opacity">{reason.num}</span>
              <div>
                <h3 className="text-lg font-semibold text-[#7EC8E3] mb-1">{reason.title}</h3>
                <p className="text-white/70 text-sm leading-relaxed">{reason.text}</p>
              </div>
            </div>
          </RevealSection>
        ))}
      </div>
    </InternalPageShell>
  );
}

function PortfolioPage() {
  const cards = [
    { title: "Passenger Sentiment Monitoring", desc: "Live dashboard showing real-time emotional state classification across cabin zones using facial expression and posture data." },
    { title: "Thermal Stress Pattern Analysis", desc: "Infrared-based heat mapping that identifies passengers showing elevated temperature or physiological stress indicators." },
    { title: "Service Escalation Early Warning", desc: "Predictive alert system that flags high-friction interactions before they escalate, giving crew time to intervene." },
    { title: "AI Dashboard for Team Leads", desc: "Operational summary view with aggregated sentiment scores, incident history, and actionable recommendations per flight segment." },
  ];

  return (
    <InternalPageShell
      title="Our Portfolio"
      subtitle="Browse the core demonstrations and outputs developed throughout this project lifecycle. Each module represents a distinct AI capability."
    >
      <div className="grid sm:grid-cols-2 gap-6">
        {cards.map((card, i) => (
          <RevealSection key={card.title} delay={i * 100}>
            <article className="group rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 hover:border-[#7EC8E3]/60 transition-all hover:-translate-y-1 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-[#FF6B35] to-[#7EC8E3] scale-x-0 group-hover:scale-x-100 transition-transform origin-left" />
              <h2 className="text-xl text-[#7EC8E3] mb-3">{card.title}</h2>
              <p className="text-white/70 text-sm leading-relaxed">{card.desc}</p>
              <p className="text-xs text-white/40 mt-4 italic">Module can be expanded with live screenshots, metrics, and methodology notes.</p>
            </article>
          </RevealSection>
        ))}
      </div>
    </InternalPageShell>
  );
}

function SupportPage() {
  return (
    <InternalPageShell
      title="Customer Support"
      subtitle="Get support, raise concerns, and find answers. We offer multiple channels depending on your needs."
    >
      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <RevealSection>
          <article className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 h-full">
            <h2 className="text-xl text-[#7EC8E3] mb-3">Help Center</h2>
            <p className="text-white/70 text-sm mb-4">
              Check frequently asked questions and standard troubleshooting steps for common issues.
            </p>
          </article>
        </RevealSection>
        <RevealSection delay={100}>
          <article className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 h-full">
            <h2 className="text-xl text-[#7EC8E3] mb-3">Submit a Ticket</h2>
            <p className="text-white/70 text-sm mb-4">
              Send your issue details and expected outcomes through the query form.
            </p>
            <Link to="/submitquery" className="text-[#FF6B35] text-sm hover:text-white transition-colors font-medium">
              Go to Submit Query →
            </Link>
          </article>
        </RevealSection>
        <RevealSection delay={200}>
          <article className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6 h-full">
            <h2 className="text-xl text-[#7EC8E3] mb-3">Speak to Team</h2>
            <p className="text-white/70 text-sm mb-4">
              Need direct guidance? Book a consultation with our project team.
            </p>
            <Link to="/bookappointment" className="text-[#FF6B35] text-sm hover:text-white transition-colors font-medium">
              Book Appointment →
            </Link>
          </article>
        </RevealSection>
      </div>
      <RevealSection delay={300}>
        <div className="rounded-2xl bg-[#0a192f]/60 border border-white/10 p-6">
          <h2 className="text-xl text-[#7EC8E3] mb-4">Frequently Asked Questions</h2>
          <div className="space-y-4">
            {[
              { q: "What data does Vector 4 collect?", a: "The system processes facial expressions, body posture, audio patterns, and thermal signatures. All data is anonymized and processed in real-time with no persistent passenger identification." },
              { q: "Is the system always recording?", a: "The sensors capture data continuously during active monitoring periods, but all processing happens on-device or at the edge. Raw data is not stored beyond the current flight session." },
              { q: "How does the crew receive alerts?", a: "Alerts are delivered through a tablet-based dashboard in the cabin and via the ground operations interface. Priority levels determine notification urgency." },
            ].map((faq) => (
              <div key={faq.q} className="border-b border-white/10 pb-4 last:border-0">
                <h3 className="text-sm font-semibold text-white/90 mb-1">{faq.q}</h3>
                <p className="text-sm text-white/60 leading-relaxed">{faq.a}</p>
              </div>
            ))}
          </div>
        </div>
      </RevealSection>
    </InternalPageShell>
  );
}
